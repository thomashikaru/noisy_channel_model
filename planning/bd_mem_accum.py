"""Accumulation test: run SEVERAL sentences sequentially in ONE process (as a cluster shard does:
model loaded once, 2 items x 5 seeds = 10 pwc.run calls) and print peak RSS after each run. If peak
grows monotonically beyond any single run's need, in-process accumulation (JIT cache / unfreed
buffers) is confirmed as the OOM driver -- not a single sentence's working set.

Usage: python bd_mem_accum.py <P>
"""
import sys, time, resource, gc
import jax
from genjax_port import pythia_word_caprop as pwc

P = int(sys.argv[1])
# Mix of lengths + the suspect items (idx 22,23,34,35,60,61 from the battery) to vary shape.
SENTS = [
    "He is very tall.",                          # short, clean
    "We saw a a movie.",                         # short, doubled-word insertion (idx 61)
    "The chef seasoned the soup.",               # idx 22 (OOM'd shard)
    "The mother gave the candle a daughter.",    # idx 23 (OOM'd shard)
    "The astronomer photographed the comet.",    # medium, clean
    "The baker iced the cake the children.",     # idx 35 (OOM'd shard)
]
for k, s in enumerate(SENTS, 1):
    t0 = time.time()
    pwc.run(s, jax.random.PRNGKey(0), P=P, band=2, max_dist=2, rejuv="gibbs+bd",
            rejuv_lookback=6, channel="align", bd_mode="gibbs", bd_funcwords=True, dedup=True)
    gc.collect()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # macOS bytes
    print(f"ACCUM run={k}/{len(SENTS)} P={P} peak_rss_GB={peak:.2f} dt={time.time()-t0:.1f}s  {s!r}",
          flush=True)
