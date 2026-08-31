import sys, time, numpy as np, jax
sys.path.insert(0, "src")
from genjax_port import pythia_word_caprop as pwc, word_stats as WS
text = "The mother gave the candle the daughter."
LIT = (380, 3101, 3534, 253, 28725, 253, 6122, 15)
def one(label, k=0, **kw):
    key = jax.random.fold_in(jax.random.PRNGKey(k), 0); ws, dg = {}, {}
    base = dict(P=64, band=2, max_dist=2, rejuv="off", rejuv_lookback=6, dedup=True, channel="align", prime=".")
    base.update(kw)
    st, lw, logZ, sl = pwc.run(text, key, word_stats=ws, diag=dg, **base)
    w = np.asarray(jax.nn.softmax(lw)); n = np.asarray(st[2]); wsf = np.asarray(st[4])
    lit = sum(w[i] for i in range(w.shape[0]) if n[i] == 8 and tuple(wsf[i, :8].tolist()) == LIT)
    post = WS.alignment_posteriors(st, lw, dg); top = pwc.decode(st, lw, skip=sl, top=1)[0]
    print(f"{label:34s} key={k}: logZ={float(logZ):7.2f} w(plain literal)={lit:.3f} del_before(The)={post['e_del_gap'][0]:.2f} "
          f"wdel_p(mean)={float(np.mean(np.asarray(dg['wdel_p']))):.2f} top={top[0]!r}@{top[1]:.2f}", flush=True)
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--lookahead-only", action="store_true",
                help="run only the lookahead arm (LOOKAHEAD_CHARGE_PLAN gate 3), 3 keys")
args = ap.parse_args()
if not args.lookahead_only:
    one("align default a=(200,2,2)", 0)
    one("align a=(200,2,0.02) p_del~1e-4", 0, action_alpha=(200.0, 2.0, 0.02))
    one("align a=(200,2,0.02) p_del~1e-4", 1, action_alpha=(200.0, 2.0, 0.02))
    one("align default band=1", 0, band=1)
    one("align default band=0", 0, band=0)
    one("char_copy (wdel -9 flat)", 0, channel="char_copy")
    one("char_copy (wdel -9 flat)", 1, channel="char_copy")
# Lookahead-charge arm (planning/LOOKAHEAD_CHARGE_PLAN.md gate 3): the deployed config with the
# APF twist ON. Expect w(plain literal) >= ~0.9 (from 0.000) and logZ ~ -53 (from -63.58).
for k in (0, 1, 2):
    one("align default + lookahead", k, lookahead=True)
