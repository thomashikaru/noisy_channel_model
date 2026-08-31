"""Memory probe for the gibbs+bd blow-up. Runs ONE (sentence, P) in this process and reports
the candidate-pool size Kc (the lax.map width over which the full-vocab [P,LCTX,V] forward is
repeated) and the process peak RSS. Run one config per process so ru_maxrss is clean.

Usage: python bd_mem_probe.py "<sentence>" <P> [rejuv] [context]
Matches the cluster config: channel=align, rejuv=gibbs+bd, band=2, max_dist=2, rejuv_lookback=6,
bd_mode=gibbs, bd_funcwords=on, bd_bridge_j=0 (default -> bridges OFF), dedup=on. A non-empty
``context`` (4th arg) is fed as the LM prime, exactly as the harness worker does for chen2023 --
it grows LCTX, which is part of what the probe must measure.
"""
import os, sys, time, resource

# Importable from any cwd / caller (planning/ -> repo root -> src), like slurm/run_nc_batch.py.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import jax
from genjax_port import pythia_word_caprop as pwc
from genjax_port import pairhmm_rejuv as RJ

sentence = sys.argv[1]
P = int(sys.argv[2])
REJUV = sys.argv[3] if len(sys.argv) > 3 else "gibbs+bd"   # "off" / "gibbs" / "gibbs+bd"
CONTEXT = sys.argv[4].strip() if len(sys.argv) > 4 else ""   # harness context -> LM prime

# Capture Kc by wrapping the sweep factory (called once per run from inside pairhmm_smc.run).
_orig = RJ.make_gibbs_indel_sweep
captured = {}
def _wrap(ctx, cand_tok, cand_len, cand_surf, *a, **k):
    captured["Kc"] = int(cand_surf.shape[0])
    captured["Wmax"] = int(ctx.Wmax)
    captured["seed_len"] = int(ctx.seed_len)
    captured["t_max"] = int(ctx.t_max)
    return _orig(ctx, cand_tok, cand_len, cand_surf, *a, **k)
RJ.make_gibbs_indel_sweep = _wrap

t0 = time.time()
st, lw, logZ, sl = pwc.run(sentence, jax.random.PRNGKey(0), P=P, band=2, max_dist=2,
                           rejuv=REJUV, rejuv_lookback=6, channel="align",
                           bd_mode="gibbs", bd_funcwords=True, dedup=True,
                           prime=(CONTEXT or pwc.PRIME))
dt = time.time() - t0
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # macOS: bytes
Kc = captured.get("Kc", -1); Wmax = captured.get("Wmax", -1)
LCTX = captured.get("seed_len", -1) + Wmax * captured.get("t_max", -1) + 1
print(f"RESULT sentence={sentence!r} P={P} ctx_words={len(CONTEXT.split())} Kc={Kc} "
      f"Wmax={Wmax} LCTX~{LCTX} grid_forwards={Wmax*Kc} runtime_s={dt:.1f} "
      f"peak_rss_GB={peak_gb:.2f}", flush=True)
