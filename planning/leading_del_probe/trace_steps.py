import sys, json, numpy as np, jax
sys.path.insert(0, "src")
from genjax_port import pythia_word_caprop as pwc
text = "The mother gave the candle the daughter."
key = jax.random.fold_in(jax.random.PRNGKey(0), 0)
tr = []
st, lw, logZ, sl = pwc.run(text, key, P=64, band=2, max_dist=2, rejuv="off", rejuv_lookback=6, dedup=True,
                           channel="align", prime=".", trace=tr)
conv = lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
print("dist entry example:", json.dumps(tr[1]["dist"][:1], default=conv)[:300])
for e in tr:
    fr = " ".join(f"pos{int(p)}:{float(m):.2f}" for p, m in e["frontier"])
    print(f"--- step {e['t']:2d} ess={e['ess']:5.1f} resampled={str(e['resampled']):5s} final={e['final']} n_unique={e['n_unique']:3d} n_done={e['n_done']:2d} logZ={e['logZ']:.2f} frontier[{fr}]")
    for x in e["dist"][:3]:
        print("      ", json.dumps(x, default=conv)[:150])
