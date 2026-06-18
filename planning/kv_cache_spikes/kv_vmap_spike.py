"""Make-or-break spike: drive penzai's KV cache UNDER jax.vmap with a PER-PARTICLE split point.

Approach (the "rewind" design): build the caching model single-sequence (batch_axes={}), vmap over
particles. Per particle: prefill the full buffer, then SET cache_end_index = posc (per-particle, so
under vmap it's a [P] value), then feed the candidate + real continuation as a fixed-length tail.
Read the tail logits and compare to the uncached full forward of the buffer-with-candidate at the
same absolute positions. Two particles use DIFFERENT posc -> proves per-particle splits work.

NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python /tmp/kv_vmap_spike.py
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
M = 8
PAD = -1
MAX_TAIL = 3

# Two particles, different buffers AND different flip positions posc.
b0 = [L.EOS_ID] + [int(i) for i in encode(" he too the")]    # [EOS,he,too,the] i_len=4, revisit posc=1
b1 = [L.EOS_ID] + [int(i) for i in encode(" i went to")]     # [EOS,i,went,to]  i_len=4, revisit posc=2
ilen = 4
posc = np.array([1, 2], np.int32)
candA = int(encode(" she")[0]);  candB = int(encode(" go")[0])
cand = np.array([candA, candB], np.int32)

def pad(xs, n):
    return xs + [PAD] * (n - len(xs))

bufs = np.array([pad(b0, M), pad(b1, M)], np.int32)               # [P, M]
# tail tokens fed after rewind: [cand, real continuation up to i_len), padded to MAX_TAIL
tails = []
for p in range(2):
    cont = list(bufs[p, posc[p] + 1:ilen])                        # tokens after the flip slot
    tails.append(pad([int(cand[p])] + cont, MAX_TAIL))
tails = np.array(tails, np.int32)                                 # [P, MAX_TAIL]
tail_valid = np.array([[1] * (ilen - posc[p]) + [0] * (MAX_TAIL - (ilen - posc[p]))
                       for p in range(2)], bool)                  # which tail slots are real

# ---- caching model, single-sequence (batch_axes={}); vmap supplies the particle axis ----
caching = sampling_mode.KVCachingTransformerLM.from_uncached(
    model, cache_len=M, batch_axes={}, pad_id=PAD)
stateless, vars0 = pz.unbind_variables(caching, freeze=True)

def score_one(buf, posc_p, tail):
    bound = pz.bind_variables(stateless, vars0, unfreeze_as_copy=True)
    bound(pz.nx.wrap(buf).tag("seq"))                            # prefill full buffer
    bound.cache_end_index.value = posc_p                        # rewind to the per-particle split
    out = bound(pz.nx.wrap(tail).tag("seq"))                    # feed cand + continuation
    return out.untag("seq", "vocabulary").unwrap()             # [MAX_TAIL, V]

try:
    cached = jax.vmap(score_one, in_axes=(0, 0, 0))(
        jnp.asarray(bufs), jnp.asarray(posc), jnp.asarray(tails))   # [P, MAX_TAIL, V]
    cached = jax.nn.log_softmax(cached, axis=-1)
    print("VMAP RAN. cached shape:", cached.shape)
except Exception as e:
    print("VMAP FAILED:", type(e).__name__, str(e)[:400])
    raise SystemExit(1)

# uncached reference: place cand at posc, full forward, compare logits at positions [posc, posc+valid)
worst = 0.0
for p in range(2):
    bwc = bufs[p].copy(); bwc[posc[p]] = cand[p]
    bwc = np.where(bwc == PAD, L.EOS_ID, bwc)                   # uncached uses EOS padding
    unc = jax.nn.log_softmax(L._raw_logits(jnp.asarray(bwc)[None]), axis=-1)[0]  # [M, V]
    n = int(tail_valid[p].sum())
    ref = unc[posc[p]: posc[p] + n]                            # [n, V]
    got = cached[p, :n]
    e = float(jnp.max(jnp.abs(ref - got)))
    print(f"  particle {p}: posc={posc[p]} valid_tail={n}  max|cached-uncached|={e:.2e}")
    worst = max(worst, e)
print("PASS" if worst < 5e-3 else "FAIL", f"(worst={worst:.2e})")
