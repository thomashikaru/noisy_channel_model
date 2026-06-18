"""De-risking spike for the prefix-KV-cache rejuvenation scorer.

Goal: prove we can (a) drive penzai's KVCachingTransformerLM so incremental cached logits match the
uncached full forward, and (b) FORK the cache state (a plain pytree via unbind_variables(freeze=True))
to score K different candidate tokens at the same position without recomputing the shared prefix.

Run eager (no jit) for speed. NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python /tmp/kv_spike.py
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
PAD = -1  # NOT EOS_ID(0): keep the BOS seed a real token, nothing treated as padding

ids = encode(" he too the")            # 3 single tokens after the EOS seed
buf = [L.EOS_ID] + [int(i) for i in ids]      # [EOS, he, too, the]
ilen = len(buf)
print("buf:", buf, "ilen:", ilen)

# --- uncached full-forward reference: log-probs at every position ---
arr = jnp.array([buf + [L.EOS_ID] * (M - ilen)], jnp.int32)   # [1, M]
unc = jax.nn.log_softmax(L._raw_logits(arr), axis=-1)[0]      # [M, V]

# --- caching model ---
caching = sampling_mode.KVCachingTransformerLM.from_uncached(
    model, cache_len=M, batch_axes={"batch": 1}, pad_id=PAD)
stateless, vars0 = pz.unbind_variables(caching, freeze=True)   # vars0: frozen pytree = empty cache


def run(frozen_vars, tok_ids):
    """Feed a chunk of token ids [S] into the cache; return (logits[S, V], new_frozen_vars)."""
    bound = pz.bind_variables(stateless, frozen_vars, unfreeze_as_copy=True)
    toks = pz.nx.wrap(jnp.asarray(tok_ids, jnp.int32)[None, :]).tag("batch", "seq")
    out = bound(toks)                                          # NamedArray [batch, seq, vocab]
    logits = out.untag("batch", "seq", "vocabulary").unwrap()[0]   # [S, V]
    _, new_vars = pz.unbind_variables(bound, freeze=True)
    return jax.nn.log_softmax(logits, axis=-1), new_vars


# (a) incremental == full forward. Prefill [EOS, he], then feed 'too', then 'the'.
lp_pre, v_pre = run(vars0, buf[:2])                # positions 0,1
lp_too, v_too = run(v_pre, [buf[2]])               # position 2 (after EOS,he,too)
lp_the, v_the = run(v_too, [buf[3]])               # position 3
err2 = float(jnp.max(jnp.abs(lp_too[0] - unc[2])))
err3 = float(jnp.max(jnp.abs(lp_the[0] - unc[3])))
print(f"(a) incremental vs full: err@pos2={err2:.2e}  err@pos3={err3:.2e}")

# (b) FORK: from the [EOS, he] prefix, score two different candidates at position 2 by reusing
# the SAME cached prefix (v_pre). Compare each to an uncached forward with that candidate placed.
candA, candB = buf[2], int(encode(" to")[0])       # 'too' vs 'to'
lpA, _ = run(v_pre, [candA])
lpB, _ = run(v_pre, [candB])
# uncached references with the candidate at position 2:
def unc_at2(cand):
    b = jnp.array([[L.EOS_ID, buf[1], cand] + [L.EOS_ID] * (M - 3)], jnp.int32)
    return jax.nn.log_softmax(L._raw_logits(b), axis=-1)[0, 2]
eA = float(jnp.max(jnp.abs(lpA[0] - unc_at2(candA))))
eB = float(jnp.max(jnp.abs(lpB[0] - unc_at2(candB))))
print(f"(b) fork candA('too') err={eA:.2e}   candB('to') err={eB:.2e}")
print("PASS" if max(err2, err3, eA, eB) < 1e-3 else "FAIL")
