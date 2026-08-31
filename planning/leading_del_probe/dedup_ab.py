import sys, time, numpy as np, jax
sys.path.insert(0, "src")
from genjax_port import pythia_word_caprop as pwc, word_stats as WS
text = "The mother gave the candle the daughter."
LIT = (380, 3101, 3534, 253, 28725, 253, 6122, 15)
def one(P, dedup, k):
    key = jax.random.fold_in(jax.random.PRNGKey(k), 0)
    ws, dg = {}, {}
    t = time.time()
    st, lw, logZ, sl = pwc.run(text, key, P=P, band=2, max_dist=2, rejuv="off", rejuv_lookback=6, dedup=dedup,
                               channel="align", prime=".", word_stats=ws, diag=dg)
    w = np.asarray(jax.nn.softmax(lw)); n = np.asarray(st[2]); wsf = np.asarray(st[4])
    lit = sum(w[i] for i in range(P) if n[i] == 8 and tuple(wsf[i, :8].tolist()) == LIT)
    post = WS.alignment_posteriors(st, lw, dg)
    distinct = len({tuple(wsf[i, :n[i]].tolist()) for i in range(P) if w[i] > 0})
    top = pwc.decode(st, lw, skip=sl, top=1)[0]
    print(f"P={P:3d} dedup={str(dedup):5s} key={k}: logZ={float(logZ):7.2f} w(plain literal)={lit:.3f} del_before(The)={post['e_del_gap'][0]:.2f} "
          f"distinct={distinct:3d} nonzero={int((w>0).sum()):3d} top={top[0]!r}@{top[1]:.2f} [{time.time()-t:.0f}s]", flush=True)
for k in (0, 1, 2):
    for dd in (True, False):
        one(64, dd, k)
one(16, True, 0); one(256, True, 0); one(256, False, 0)
