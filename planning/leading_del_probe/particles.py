import sys, numpy as np, jax, collections
sys.path.insert(0, "src")
from genjax_port import pythia_word_caprop as pwc, word_stats as WS, pairhmm_smc
text = "The mother gave the candle the daughter."
key = jax.random.fold_in(jax.random.PRNGKey(0), 0)          # the worker's item_key for idx 0, seed 0
kw = dict(P=64, band=2, max_dist=2, wdel=None, wins=None, rejuv="off", rejuv_lookback=6, trace=None,
          dedup=True, lm_temp=1.0, ins_rate=0.02, uniform_ins=False, action_alpha=None, channel="align",
          align_slope=None, bd_p_stay=0.0, bd_mode="gibbs", bd_attempts=1, bd_funcwords=True, prime=".")
ws, dg = {}, {}
st, lw, logZ, sl = pwc.run(text, key, word_stats=ws, diag=dg, **kw)
post = WS.alignment_posteriors(st, lw, dg)
print(f"sl={sl} M={dg['M']} logZ={float(logZ):.3f} e_del_gap={np.round(post['e_del_gap'],3).tolist()} p_err_pos={np.round(post['p_err_positional'],2).tolist()}")
print("decode top3:", pwc.decode(st, lw, skip=sl, top=3))
model = pwc._pythia_model(pwc.PRIME)
ctx_buf, ctx_len, n_words, word_len, word_surf, log_alpha, done = st
w = np.asarray(jax.nn.softmax(lw)); n = np.asarray(n_words); wl = np.asarray(word_len); wsf = np.asarray(word_surf)
pp = post["per_particle"]
groups = collections.OrderedDict()
for p in range(w.shape[0]):
    one = tuple(a[p:p+1] for a in st)
    s = pairhmm_smc.decode(one, lw[p:p+1], model, skip=sl, top=1)[0][0]
    k = (s, int(n[p]), tuple(wsf[p, :n[p]].tolist()), tuple(wl[p, :n[p]].tolist()))
    g = groups.setdefault(k, {"w": 0.0, "ps": [], "edel0": pp["e_del_gap"][p, 0]})
    g["w"] += w[p]; g["ps"].append(p)
for (s, nn, surf, lens), g in sorted(groups.items(), key=lambda kv: -kv[1]["w"])[:6]:
    print(f"w={g['w']:.3f} n_words={nn} edel0={g['edel0']:.2f} ctx_len={int(np.asarray(ctx_len)[g['ps'][0]])} decode={s!r}")
    print(f"      surf={surf}\n      tok_len={lens}")
print("obs_words:", dg.get("obs_words"))
