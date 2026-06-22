"""Phase-2 gate 4 (live, throwaway): the INS-01 duplicate `the boy handed handed the pencil to the girl`.
gibbs+bd's informed q_del should REMOVE the doubled `handed` -> top decode == `...handed the pencil...`.
Compares rejuv in {gibbs, gibbs+bd} on pythia-70m (align channel default). Also a SHORT warm-up case first
to confirm the path runs + give an early timing read before the long sentence.

This is the FIRST live exercise of the Phase-2 birth/death path -- the bd move does O(Wmax^2) un-jitted
score_fn calls (suffix-tail KV sharing is the deferred perf win), so it can be slow on long inputs; timed
per phase so we can judge whether the perf optimization is needed before gate 5.

Run (redirect, never pipe):  python -u -m planning.bd_live_gate > planning/bd_live_gate.log 2>&1
"""
import os, time
os.environ.setdefault("NC_LM", "EleutherAI/pythia-70m")
import jax
from genjax_port import lm_penzai
from genjax_port.pythia_word_caprop import run, decode, _norm

import os as _os
CASES = [
    ("warmup short dup", "the the dog ran", "the dog ran", 64),
    ("INS-01 handed handed", "the boy handed handed the pencil to the girl",
     "the boy handed the pencil to the girl", 96),
]
if _os.environ.get("BD_WARMUP_ONLY") == "1":
    CASES = CASES[:1]

print("loading pythia-70m...", flush=True)
t0 = time.time()
lm_penzai.load_model()
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

for name, obs, target, P in CASES:
    print(f"\n=== {name}  P={P} ===\n  obs   : {obs!r}\n  target: {target!r}", flush=True)
    for rejuv in ["gibbs", "gibbs+bd"]:
        ts = time.time()
        st, lw, logZ, sl = run(obs, jax.random.PRNGKey(0), P=P, rejuv=rejuv)
        top = decode(st, lw, skip=sl, top=4)
        dt = time.time() - ts
        hit = _norm(top[0][0]) == _norm(target)
        print(f"  [{dt:6.1f}s] rejuv={rejuv:9s} logZ={float(logZ):8.2f}  top1={'HIT' if hit else 'miss'}",
              flush=True)
        for s, m in top:
            mark = "  <-- target" if _norm(s) == _norm(target) else ""
            print(f"        {m:6.3f}  {s!r}{mark}", flush=True)
print("\nDONE", flush=True)
