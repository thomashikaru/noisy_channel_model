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

# (label, rejuv, kwargs) -- p_stay=0 reproduces the always-move move; p_stay=0.5 adds the §11 STAY branch.
CONFIGS = [
    ("off", "off", {}),
    ("gibbs", "gibbs", {}),
    ("bd stay=0.0", "gibbs+bd", dict(bd_p_stay=0.0)),
    ("bd stay=0.5", "gibbs+bd", dict(bd_p_stay=0.5)),
]
t0 = time.time()
results = {}
for label, rejuv, kw in CONFIGS:
    ts = time.time()
    state, log_w, logZ, sl = pairhmm_smc.run(OBS, key, model, P=P, wdel=WDEL, wins=WINS,
                                             band=None, rejuv=rejuv, **kw)
    top = pairhmm_smc.decode(state, log_w, model, top=6)
    dd = dict(top)
    dedup_mass = dd.get(DEDUP, 0.0)
    results[label] = (float(logZ), dedup_mass)
    print(f"[{time.time()-ts:5.1f}s] {label:12s}  logZ={float(logZ):8.3f}  "
          f"P({DEDUP!r})={dedup_mass:.3f}", flush=True)
    for s, m in top:
        mark = "  <-- deduped" if s == DEDUP else ""
        print(f"        {m:6.3f}  {s!r}{mark}", flush=True)

print(f"\nTotal {time.time()-t0:.1f}s", flush=True)
zg, dg = results["gibbs"]
print(f"\nGATE 3 (dedup must hold; stay must not depress logZ further):")
for label in ("bd stay=0.0", "bd stay=0.5"):
    zb, db = results[label]
    print(f"  {label:12s}  dedup={db:.3f} (gibbs {dg:.3f}, {'PASS' if db >= dg - 1e-6 else 'FAIL'})  "
          f"logZ={zb:.3f} (gibbs {zg:.3f})", flush=True)
