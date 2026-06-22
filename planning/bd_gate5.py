"""Phase-2.5 GATE 5 (live, throwaway): no-regression of birth/death + the LM-bridge pool fix.

Gate-5 (first run) was a SPLIT: gibbs+bd removed the doubled `on` in INS-02a (0->0.985) but REGRESSED
DEL-of-01a (restore dropped `of`: 0.047->0.000). Root cause: the bd birth pool was OBSERVED surfaces only,
so a birth could not propose `of` (not an observed word). Phase 2.5 enriches the pool with the SAME top-J
LM bridges the forward filter uses (`bd_bridge_j`>0). This harness re-runs the gate comparing:
  * gibbs                       -- substitution-only sweep baseline
  * gibbs+bd bridges OFF (j=0)  -- reproduces the observed-only regression
  * gibbs+bd bridges ON  (j=3)  -- the fix; should restore `of` AND keep the dedup wins
on three items: DEL-of-01a (insertion restore), INS-02a (deletion of duplicate), INS-02b (clean keep guard
-- bridges must NOT induce a spurious birth on an input that needs no edit). Timed per phase so the cost
delta of the bigger pool is measured (the move's `_ins_logq` is O(Kc) score_fn calls).

Run (redirect, never pipe):  python -u -m planning.bd_gate5 > planning/bd_gate5.log 2>&1
"""
import os, time
os.environ.setdefault("NC_LM", "EleutherAI/pythia-70m")
import jax
from genjax_port import lm_penzai
from genjax_port.pythia_word_caprop import run, decode, _norm

P = 128
# (name, observed, target)
CASES = [
    ("DEL-of-01a restore 'of'", "this is one the best", "this is one of the best"),
    ("INS-02a remove 'on on'", "the cat sat on on the mat", "the cat sat on the mat"),
    ("INS-02b clean keep", "the cat sat on the mat", "the cat sat on the mat"),
]
# (label, rejuv, kwargs). Bridges OFF (j=0) to isolate the §11 STAY branch: stay must fix the INS-02b
# clean-keep disaster (0.000) and the DEL-of regression WITHOUT bridges, while preserving the INS-02a dedup.
# (stay=0.0 was the broken baseline -- DEL 0.000 / INS-02a 0.985 / INS-02b 0.000; not re-run here.)
CONFIGS = [
    ("gibbs", "gibbs", {}),
    ("bd stay=0.1", "gibbs+bd", dict(bd_bridge_j=0, bd_p_stay=0.1)),
    ("bd stay=0.3", "gibbs+bd", dict(bd_bridge_j=0, bd_p_stay=0.3)),
]

print("loading pythia-70m...", flush=True)
t0 = time.time()
lm_penzai.load_model()
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

for name, obs, target in CASES:
    print(f"\n=== {name}  P={P} ===\n  obs   : {obs!r}\n  target: {target!r}", flush=True)
    base = None
    for label, rejuv, kw in CONFIGS:
        ts = time.time()
        st, lw, logZ, sl = run(obs, jax.random.PRNGKey(0), P=P, rejuv=rejuv, **kw)
        top = decode(st, lw, skip=sl, top=4)
        dt = time.time() - ts
        tmass = sum(m for s, m in top if _norm(s) == _norm(target))
        hit = _norm(top[0][0]) == _norm(target)
        reg = ""
        if label == "gibbs":
            base = tmass
        elif base is not None:
            reg = ("  REGRESSION vs gibbs" if tmass + 1e-6 < base
                   else "  ok vs gibbs")
        print(f"  [{dt:6.1f}s] {label:18s} logZ={float(logZ):8.2f}  "
              f"top1={'HIT' if hit else 'miss'}  target_mass={tmass:.3f}{reg}", flush=True)
        for s, m in top:
            mark = "  <-- target" if _norm(s) == _norm(target) else ""
            print(f"        {m:6.3f}  {s!r}{mark}", flush=True)
print("\nDONE", flush=True)
