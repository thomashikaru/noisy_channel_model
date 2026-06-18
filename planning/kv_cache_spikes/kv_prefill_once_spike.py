"""Spike 3: the ACTUAL integration shape for manual_subflip_move's prefix-KV scorer.

Risks the earlier two spikes did NOT cover:
  (1) PREFILL ONCE, reuse the prefix cache across K candidate tails (kv_spike's score_one
      re-prefills per candidate -- that defeats the whole saving). Do prefill once -> frozen vars,
      then bind those vars K times (unfreeze_as_copy=True), rewind, feed each candidate's tail.
  (2) Run the whole thing UNDER jax.jit (production manual_subflip_move lives inside the jitted
      _aligned_window_move_fn). The earlier spikes ran eager.

We reproduce manual_subflip_move's `chain[P,K]` (suffix sum over [posc, i_len), incl. the posc
candidate prior) via the cache and compare to the uncached [P*K, M] full forward.

    NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python planning/kv_cache_spikes/kv_prefill_once_spike.py
"""
import numpy as np
import jax
import jax.numpy as jnp
from penzai import pz
from penzai.models.transformer import sampling_mode

from src.genjax_port import lm_penzai as L
from src.genjax_port.tokenizer import encode

L.load_model()
model = L._model
PAD = -1
M = 8           # cache_len (buffer bucket)
MAX_TAIL = 4    # lookback + 1

# Two particles; revisit a MIDDLE word so the suffix term is real.
b0 = [L.EOS_ID] + [int(i) for i in encode(" he too the dog")]   # [EOS,he,too,the,dog] i_len=5
b1 = [L.EOS_ID] + [int(i) for i in encode(" i went to bed")]    # [EOS,i,went,to,bed]  i_len=5
ilen = np.array([5, 5], np.int32)
posc = np.array([2, 3], np.int32)                                # flip "too"(p0) / "to"(p1)

# K candidates for the flipped position (shared id table, as in the production move).
cand_strs = [" too", " to", " two", " toe"]
cand_x = np.array([int(encode(s)[0]) for s in cand_strs], np.int32)   # [K]
K = cand_x.shape[0]

def pad(xs, n):
    return xs + [PAD] * (n - len(xs))

bufs = jnp.asarray([pad(b0, M), pad(b1, M)], jnp.int32)          # [P, M]
ilen = jnp.asarray(ilen); posc = jnp.asarray(posc)
cand_x = jnp.asarray(cand_x)
P = bufs.shape[0]

caching = sampling_mode.KVCachingTransformerLM.from_uncached(
    model, cache_len=M, batch_axes={}, pad_id=PAD)
stateless, vars0 = pz.unbind_variables(caching, freeze=True)


def kv_chain(buf, i_len, posc_p):
    """chain[K]: for each candidate, prior P_LM(cand|prefix) + sum log P(continuation | cand-context).

    prefill the buffer (positions >= i_len masked to PAD so they don't enter the cache), read the
    candidate prior from prefill logits at posc-1, rewind to posc, feed each candidate's tail
    [cand, real continuation] and sum the continuation log-probs.
    """
    idx = jnp.arange(M)
    buf_pf = jnp.where(idx < i_len, buf, PAD)                    # mask padding tail to -1
    pf = pz.bind_variables(stateless, vars0, unfreeze_as_copy=True)
    pf_out = pf(pz.nx.wrap(buf_pf).tag("seq")).untag("seq", "vocabulary").unwrap()  # [M, V]
    pf_lp = jax.nn.log_softmax(pf_out, axis=-1)
    _, prefix_vars = pz.unbind_variables(pf, freeze=True)
    prior = pf_lp[posc_p - 1][cand_x]                           # [K]  P_LM(cand|prefix)

    # tail target tokens = real continuation buf[posc+1 .. i_len-1], gathered at fixed MAX_TAIL width
    j = jnp.arange(MAX_TAIL)
    tail_pos = posc_p + j                                        # [MAX_TAIL]
    cont = jnp.where(tail_pos < i_len, buf[jnp.clip(tail_pos, 0, M - 1)], PAD)  # tail[j]=buf[posc+j]

    def one_cand(cx):
        tail = cont.at[0].set(cx)                               # tail[0]=cand at posc
        b = pz.bind_variables(stateless, prefix_vars, unfreeze_as_copy=True)
        b.cache_end_index.value = posc_p                       # REWIND (prefix [0,posc) preserved)
        out = b(pz.nx.wrap(tail).tag("seq")).untag("seq", "vocabulary").unwrap()  # [MAX_TAIL, V]
        lp = jax.nn.log_softmax(out, axis=-1)
        # tail logits at j predict tail[j+1] = buf[posc+1+j]; valid while posc+1+j < i_len
        tgt = jnp.where((posc_p + 1 + j) < i_len, buf[jnp.clip(posc_p + 1 + j, 0, M - 1)], 0)
        step_lp = jnp.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]   # [MAX_TAIL]
        valid = (posc_p + 1 + j) < i_len
        suffix_cont = jnp.sum(jnp.where(valid, step_lp, 0.0))
        return suffix_cont

    suffix = jax.vmap(one_cand)(cand_x)                         # [K]
    return prior + suffix


def chain_all(bufs, ilen, posc):
    return jax.vmap(kv_chain)(bufs, ilen, posc)                 # [P, K]


# --- uncached reference: the current manual_subflip_move chain computation ---
def uncached_chain(bufs, ilen, posc):
    rows = jnp.arange(P)
    kr = jnp.arange(K)
    flatbufs = jnp.broadcast_to(bufs[:, None, :], (P, K, M))
    flatbufs = flatbufs.at[rows[:, None], kr[None, :], posc[:, None]].set(
        jnp.broadcast_to(cand_x[None, :], (P, K)))
    flat = jnp.where(flatbufs == PAD, L.EOS_ID, flatbufs).reshape(P * K, M)
    raw = L._raw_logits(flat)
    lp = jax.nn.log_softmax(raw, axis=-1)
    tok = jnp.concatenate([flat[:, 1:], flat[:, -1:]], axis=1)
    g = jnp.take_along_axis(lp, tok[:, :, None], axis=-1)[..., 0]
    prev = jnp.concatenate([g[:, :1], g[:, :-1]], axis=1).reshape(P, K, M)
    idx = jnp.arange(M)
    mask = (idx[None, None, :] >= posc[:, None, None]) & (idx[None, None, :] < ilen[:, None, None])
    return jnp.sum(jnp.where(mask, prev, 0.0), axis=2)         # [P, K]


ref = uncached_chain(bufs, ilen, posc)

print("=== eager ===")
got_eager = chain_all(bufs, ilen, posc)
e1 = float(jnp.max(jnp.abs(ref - got_eager)))
print(f"eager  max|kv-uncached| = {e1:.2e}")

print("=== jit ===")
try:
    got_jit = jax.jit(chain_all)(bufs, ilen, posc)
    e2 = float(jnp.max(jnp.abs(ref - got_jit)))
    print(f"jit    max|kv-uncached| = {e2:.2e}")
except Exception as ex:
    print("JIT FAILED:", type(ex).__name__, str(ex)[:500])
    e2 = 1.0

worst = max(e1, e2)
print("ref chain[0]:", np.asarray(ref[0]))
print("kv  chain[0]:", np.asarray(got_eager[0]))
print("PASS" if worst < 5e-3 else "FAIL", f"(worst={worst:.2e})")
