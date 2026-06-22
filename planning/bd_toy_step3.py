"""Phase-2 Step 3 (throwaway): does NEAR-TERMINAL bd gating (bd_min_done) cut the logZ-depression / weight
variance while keeping the dedup behavioral win? Toy `the cat cat sat`; compare gibbs (baseline) and gibbs+bd
at bd_min_done in {0.0 (every event), 0.5, 0.9 (only when >=90% done)}.
"""
import time
import jax
from genjax_port import pairhmm_smc
from genjax_port.tests.test_pairhmm_exact import _toy_model, WDEL, WINS
from genjax_port.tests.toy_bigram import lm_logits

OBS, DEDUP, DUP = "the cat cat sat", "the cat sat", "the cat cat sat"
P = 3000
model = _toy_model(lm_logits)
key = jax.random.PRNGKey(0)

def show(tag, rejuv, **kw):
    ts = time.time()
    st, lw, logZ, sl = pairhmm_smc.run(OBS, key, model, P=P, wdel=WDEL, wins=WINS, band=None,
                                       rejuv=rejuv, **kw)
    dd = dict(pairhmm_smc.decode(st, lw, model, top=8))
    print(f"[{time.time()-ts:5.1f}s] {tag:22s} logZ={float(logZ):8.3f}  "
          f"dedup={dd.get(DEDUP,0.0):.3f}  dup={dd.get(DUP,0.0):.3f}", flush=True)
    return float(logZ)

zg = show("gibbs", "gibbs")
for thr in (0.0, 0.5, 0.9):
    show(f"gibbs+bd min_done={thr}", "gibbs+bd", bd_min_done=thr)
print(f"\n(baseline gibbs logZ={zg:.3f}; want gibbs+bd dedup high + logZ near gibbs)", flush=True)
print("DONE", flush=True)
