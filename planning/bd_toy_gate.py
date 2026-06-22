"""Phase-2 gate 3 (throwaway): toy duplicate `the cat cat sat`. The target prefers the deduped `the cat sat`
(a spurious-word/insertion reading). Compare rejuv in {off, gibbs, gibbs+bd}: gibbs+bd's informed
near-conditional q_del should REMOVE the duplicate -> deduped-reading mass >= gibbs, and logZ must NOT be
depressed (the Phase-1 uniform move depressed it ~1.3 nats = weight-variance blowup).

Run:  python -u -m planning.bd_toy_gate   (from repo src/ ; or adjust sys.path)
"""
import sys, time
import jax

from genjax_port import pairhmm_smc
from genjax_port.tests.test_pairhmm_exact import _toy_model, WDEL, WINS
from genjax_port.tests.toy_bigram import lm_logits

OBS = "the cat cat sat"
DEDUP = "the cat sat"
P = 3000
model = _toy_model(lm_logits)
key = jax.random.PRNGKey(0)

t0 = time.time()
results = {}
for rejuv in ["off", "gibbs", "gibbs+bd"]:
    ts = time.time()
    state, log_w, logZ, sl = pairhmm_smc.run(OBS, key, model, P=P, wdel=WDEL, wins=WINS,
                                             band=None, rejuv=rejuv)
    top = pairhmm_smc.decode(state, log_w, model, top=6)
    dd = dict(top)
    dedup_mass = dd.get(DEDUP, 0.0)
    results[rejuv] = (float(logZ), dedup_mass)
    print(f"[{time.time()-ts:5.1f}s] rejuv={rejuv:9s}  logZ={float(logZ):8.3f}  "
          f"P({DEDUP!r})={dedup_mass:.3f}", flush=True)
    for s, m in top:
        mark = "  <-- deduped" if s == DEDUP else ""
        print(f"        {m:6.3f}  {s!r}{mark}", flush=True)

print(f"\nTotal {time.time()-t0:.1f}s", flush=True)
zg, dg = results["gibbs"]
zb, db = results["gibbs+bd"]
print(f"\nGATE 3:")
print(f"  deduped-mass  gibbs={dg:.3f}  gibbs+bd={db:.3f}   -> {'PASS' if db >= dg - 1e-6 else 'FAIL'} (bd >= gibbs)")
print(f"  logZ          gibbs={zg:.3f}  gibbs+bd={zb:.3f}   -> {'PASS' if zb >= zg - 0.3 else 'FAIL'} (bd not depressed)")
