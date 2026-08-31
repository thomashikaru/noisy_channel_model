"""§3.5 probe (NOISY_CHANNEL_HARNESS_IMPLEMENTATION_PLAN.md): context-prime compile cost.

Each distinct prime length gives a new seed_len, hence a new LCTX, hence new XLA compile shapes
for the forward. Question for the Phase-3 worker: how expensive is the first call at a new
(seed_len, M), and does the second call at the SAME (seed_len, M) reuse the compile (which is what
makes sorting a shard by (seed_len, M) sufficient, without lctx_round bucketing)?

One item, three seed lengths, first vs second call (different PRNG key -- keys never retrace).

Run:  conda run -n ncgenjax python -u planning/lctx_compile_probe.py
"""
import time

import jax

from genjax_port import lm_penzai
from genjax_port import pythia_word_caprop as pwc

lm_penzai.load_model()

OBS = "the boy gave the ball to the girl."
PRIMES = [
    ".",                                                            # the default (seed_len 2)
    "The waiter was very attentive tonight.",                       # short context
    "The waiter was very attentive tonight. He brought everything quickly and "
    "checked on the table twice.",                                  # long context
]

t_all = time.time()
for prime in PRIMES:
    for rep in (1, 2):
        t0 = time.time()
        _st, _lw, z, sl = pwc.run(OBS, jax.random.PRNGKey(rep), P=64, rejuv="off", prime=prime)
        print(f"[{time.time()-t_all:6.1f}s] seed_len={sl:3d} call={rep}  "
              f"t={time.time()-t0:6.1f}s  logZ={z:.2f}", flush=True)
print(f"[{time.time()-t_all:6.1f}s] done", flush=True)
