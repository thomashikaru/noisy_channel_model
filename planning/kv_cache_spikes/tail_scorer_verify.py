"""R3 de-risk: confirm lm_penzai.batch_tail_logprobs (KV and uncached) computes chain-rule
log P(tail | ctx) with the convention `tail = continuation AFTER ctx[:ctx_len]` -- i.e. it matches a
hand chain-rule via next_token_logprobs. If KV==uncached==manual within ~5e-3, the rejuvenation sweep
can drop its (n_out+1) whole-sentence rescore for this suffix scorer (the prefix LM cancels in the
conditional)."""
import numpy as np
import jax
import jax.numpy as jnp

from genjax_port import lm_penzai as L
from genjax_port.tokenizer import encode

L.load_model()

# ctx = ". the cat", tails = K candidate continuations "sat on mat" / "ran on mat" ...
CACHE_LEN = 12
ctx_ids = [L.EOS_ID] + [int(i) for i in encode(". the cat")]
ctx_len = len(ctx_ids)
ctx_buf = ctx_ids + [L.CACHE_PAD] * (CACHE_LEN - len(ctx_ids))

cont_strs = [" sat on mat", " ran on mat", " cat on mat"]
MAX_TAIL = 4
tails, tail_lens = [], []
for s in cont_strs:
    t = [int(i) for i in encode(s)][:MAX_TAIL]
    tail_lens.append(len(t))
    tails.append(t + [L.CACHE_PAD] * (MAX_TAIL - len(t)))

ctx_bufs = jnp.asarray([ctx_buf], jnp.int32)              # [1, CACHE_LEN]
ctx_lens = jnp.asarray([ctx_len], jnp.int32)              # [1]
tails_a = jnp.asarray([tails], jnp.int32)                 # [1, K, MAX_TAIL]
tail_lens_a = jnp.asarray([tail_lens], jnp.int32)         # [1, K]
K = len(cont_strs)


def manual_chain(ctx_buf, ctx_len, tail, tlen):
    """Hand chain-rule via next_token_logprobs: sum_i log P(tail[i] | ctx + tail[:i])."""
    buf = list(np.asarray(ctx_buf))
    buf = [b if b != L.CACHE_PAD else L.EOS_ID for b in buf]
    sc = 0.0
    cur = list(buf[:int(ctx_len)])
    for i in range(int(tlen)):
        padded = (cur + [L.EOS_ID] * (CACHE_LEN + MAX_TAIL))[:CACHE_LEN + MAX_TAIL]
        lp = L.next_token_logprobs(jnp.asarray([padded], jnp.int32), jnp.asarray([len(cur)], jnp.int32))
        sc += float(lp[0, int(tail[i])])
        cur.append(int(tail[i]))
    return sc


manual = np.array([[manual_chain(ctx_buf, ctx_len, tails[k], tail_lens[k]) for k in range(K)]])
unc = np.asarray(L.batch_tail_logprobs(ctx_bufs, ctx_lens, tails_a, tail_lens_a, use_kv=False))
kv = np.asarray(L.batch_tail_logprobs(ctx_bufs, ctx_lens, tails_a, tail_lens_a, use_kv=True))

print("manual  :", manual[0])
print("uncached:", unc[0])
print("kv      :", kv[0])
print(f"max|manual-uncached| = {np.max(np.abs(manual-unc)):.2e}")
print(f"max|manual-kv|       = {np.max(np.abs(manual-kv)):.2e}")
print(f"max|uncached-kv|     = {np.max(np.abs(unc-kv)):.2e}")
ok = np.max(np.abs(manual - unc)) < 5e-3 and np.max(np.abs(manual - kv)) < 5e-3
print("PASS" if ok else "FAIL")
