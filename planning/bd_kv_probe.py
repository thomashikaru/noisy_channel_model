"""Timing probe: split the gibbs+bd (indel move) per-item wall-clock into COMPILE vs STEADY-STATE EXEC.

Purpose (before committing to the KV suffix-tail rewrite): the slowdown report estimated the indel
move's cost from a "full-forward" count, but the KV feed over a FULL-length exact tail costs about as
much as a full forward -- so the exact rewrite is only ~1.5-3x, not 70-80%. If the real per-item cost is
dominated instead by the per-run RECOMPILE of the indel sweep (make_gibbs_indel_sweep is NOT lru_cached,
unlike make_sweep -- report Cause 3), then the exact Cause-3 fix is the higher-ROI lever and the KV
rewrite barely moves wall-clock. This probe settles which one dominates.

Method: run the SAME sentence (same shape) several times in ONE process under rejuv=gibbs+bd.
  * make_gibbs_indel_sweep rebuilds a fresh @jax.jit `_logits` every run() -> a new function object ->
    JAX recompiles. So if the sweep recompiles per run, run #2 ~= run #1 (both pay compile).
  * If run #2 << run #1, compile was amortized and run #2 ~ steady-state EXEC.
Comparisons:
  A#1 cold  vs  A#2 / A#3 warm (same shape)  -> compile-per-run (Cause 3) if A#2 ~= A#1
  A gibbs+bd (warm)  -  A gibbs-only         -> the indel move's marginal steady-state cost (Cause 1)
  B (new length)     -  A#2                   -> new-shape compile spike

Deployment config mirrored from calibration_word_action_smc.evaluate: channel=align, P=256, band=2,
align defaults, dedup=True, bd_mode=gibbs, bd_attempts=1, bd_funcwords=True.
"""
import os, sys, time

import jax

from genjax_port import pythia_word_caprop as W
from genjax_port import lm_penzai
from genjax_port.pythia_word_caprop import ALIGN_ALPHA_DEFAULT

P = int(os.environ.get("NC_P", "256"))
SEED = int(os.environ.get("NC_SEED", "0"))

# Same-shape sentence run repeatedly (doubled function word -> the indel move has real work: it scores
# insertions at every gap + deletions). B is a DIFFERENT length to expose the new-shape compile spike.
A = "the cat sat on on the warm mat"                          # 8 observed words
B = "the quick brown fox jumped over over the lazy dog again"  # 11 observed words


def _wellform(s):
    s = s.strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?":
        s = s + "."
    return s


def timed(label, observed, rejuv):
    t = time.time()
    st, lw, logZ, sl = W.run(_wellform(observed), jax.random.PRNGKey(SEED), P=P, band=2,
                             action_alpha=ALIGN_ALPHA_DEFAULT, rejuv=rejuv, dedup=True,
                             channel="align", bd_mode="gibbs", bd_attempts=1, bd_funcwords=True)
    jax.block_until_ready((lw, logZ, st))     # force async dispatch to finish before stopping the clock
    dt = time.time() - t
    n = len(observed.split())
    print(f"[{time.strftime('%H:%M:%S')}] {label:26s} rejuv={rejuv:8s} len={n:2d}  ->  {dt:7.1f}s", flush=True)
    return dt


def main():
    print(f"=== bd_kv_probe: compile-vs-exec split for the indel move (P={P}, seed={SEED}) ===", flush=True)
    print("Loading pythia-70m ...", flush=True)
    t0 = time.time()
    lm_penzai.load_model()
    print(f"model loaded in {time.time()-t0:.1f}s. Estimated total probe time ~10-20 min "
          f"(5 heavy SMC runs; the first ~compile-heavy).\n", flush=True)

    a1 = timed("A #1 (cold)", A, "gibbs+bd")
    a2 = timed("A #2 (warm, same shape)", A, "gibbs+bd")
    a3 = timed("A #3 (warm, same shape)", A, "gibbs+bd")
    g = timed("A  gibbs-only", A, "gibbs")
    off = timed("A  off", A, "off")
    b = timed("B  (new length)", B, "gibbs+bd")

    print("\n--- interpretation ---", flush=True)
    print(f"A#1={a1:.1f}s  A#2={a2:.1f}s  A#3={a3:.1f}s  gibbs={g:.1f}s  off={off:.1f}s  B={b:.1f}s", flush=True)
    print(f"recompile-per-run (A#1 - A#2)          = {a1-a2:7.1f}s"
          f"   [~0 and A#2~=A#1 -> sweep RECOMPILES every run: Cause 3 dominates]", flush=True)
    print(f"steady indel exec (A#2_gibbs+bd - A_gibbs) = {a2-g:7.1f}s"
          f"   [the part the KV suffix-tail rewrite (~1.5-3x) could cut: Cause 1]", flush=True)
    print(f"gibbs-over-off (A_gibbs - A_off)       = {g-off:7.1f}s", flush=True)
    print(f"new-length compile (B - A#2)           = {b-a2:7.1f}s", flush=True)
    verdict = ("RECOMPILE (Cause 3) dominates -> fix the lru_cache/traced-args first; KV rewrite is minor"
               if (a1 - a2) > (a2 - g) else
               "STEADY EXEC (Cause 1) dominates -> the KV suffix-tail rewrite's ~1.5-3x is the lever")
    print(f"\nVERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
