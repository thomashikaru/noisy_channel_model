"""Measure cloud degeneracy (unique particle states U vs P) at the gibbs+bd indel move. If U/P is
small, deduplicating the candidate scoring over unique particles alone collapses the Wmax*Kc grid's
memory + runtime by P/U -- possibly without the full suffix-tail rewrite. If U/P ~ 1 (diverse cloud),
dedup won't help and tail scoring is the robust fix.

Usage: python bd_degeneracy_probe.py "<sentence>" <P>
"""
import sys, numpy as np, jax
from genjax_port import pythia_word_caprop as pwc
from genjax_port import pairhmm_rejuv as RJ

sentence, P = sys.argv[1], int(sys.argv[2])
_orig = RJ.make_gibbs_indel_sweep
seen = []
def _wrap(ctx, *cands, **kw):
    sweep = _orig(ctx, *cands, **kw)
    def wrapped(key, ctx_buf, ctx_len, word_len, word_surf, done, theta_costs=None):
        cb = np.asarray(ctx_buf); wl = np.asarray(word_len); ws = np.asarray(word_surf)
        dn = np.asarray(done); cl = np.asarray(ctx_len)
        keys = set()
        for r in range(cb.shape[0]):
            keys.add(cb[r, :cl[r]].tobytes() + b"|" + wl[r].tobytes() + b"|" +
                     ws[r].tobytes() + b"|" + dn[r].tobytes())
        seen.append((cb.shape[0], len(keys), int(dn.sum())))
        return sweep(key, ctx_buf, ctx_len, word_len, word_surf, done, theta_costs)
    return wrapped
RJ.make_gibbs_indel_sweep = _wrap

pwc.run(sentence, jax.random.PRNGKey(0), P=P, band=2, max_dist=2, rejuv="gibbs+bd",
        rejuv_lookback=6, channel="align", bd_mode="gibbs", bd_funcwords=True, dedup=True)
for (p, u, nd) in seen:
    print(f"DEGEN sentence={sentence!r} P={p} unique_states={u} U/P={u/p:.3f} done={nd}", flush=True)
