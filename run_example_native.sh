#!/usr/bin/env bash
# Run the GENJAX-NATIVE noisy-channel model WITH INTERLEAVED REJUVENATION on one sentence and print
# the inferred alternatives + runtime. This is `run.py --filter native --conditional_rejuv`
# (rejuv_bridge.run_smc_conditional_rejuv): the word-scan SMC filtering sweep with surprisal-gated
# rejuvenation interleaved after every word's resample, VECTORIZED over particles (one vmapped MH
# substitution-flip move re-decides earlier words using later context). See
# src/genjax_port/VECTORIZED_REJUV_PLAN.md and MIGRATION_PLAN.md.
#
# Uses pythia-70m (fast iteration). For the plain sweep without rejuvenation, drop --conditional_rejuv
# (see the elif branches in run.py) or run ./run_example.sh (the hand-rolled reference filter).
#
# v1 REJUVENATION SCOPE: single-token observed words, substitution-only. A sentence with a multi-token
# word (e.g. "experimemt") gracefully SKIPS rejuvenation and runs the plain substitution filter
# (which still corrects it) -- never an error, never less capable than --filter native. Use
# single-token words to actually exercise the rejuvenation move; multi-token rejuv is Phase 2 / R2.
#
#   Usage:  ./run_example_native.sh [particle_count] [max_edit_distance]
#   e.g.    ./run_example_native.sh 128
#           ./run_example_native.sh 64 3
#   Tune the rejuvenation trigger / lookback / sweeps via the variables below.

# ---- edit these ----
SENTENCE="he wants too go home"   # v1 rejuv scope: SINGLE-TOKEN words only (multi-token -> error)
PARTICLES="${1:-64}"              # number of SMC particles
MAX_DIST="${2:-2}"               # max char edit distance for word-substitution candidates (SymSpell)

# interleaved rejuvenation (surprisal-gated, vectorized over particles)
LOOKBACK=4            # words of context to revisit on each rejuvenation event
LOGPROB_THRESH=5.0    # surprisal-gate CENTER (the trigger): higher => rejuvenate less often
LOGPROB_SPREAD=1.0    # surprisal-gate STEEPNESS (larger => sharper on/off around the center)
REJUV_SWEEPS=2        # MH sweeps over the lookback window per rejuvenation event
# --------------------

cd "$(dirname "$0")"
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh
conda activate ncgenjax
export TOKENIZERS_PARALLELISM=false

SECONDS=0
NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python -m src.genjax_port.run \
  --filter native --conditional_rejuv \
  --sentence "$SENTENCE" --particles "$PARTICLES" --max_dist "$MAX_DIST" \
  --lookback "$LOOKBACK" --logprob_thresh "$LOGPROB_THRESH" \
  --logprob_spread "$LOGPROB_SPREAD" --rejuv_sweeps "$REJUV_SWEEPS" \
  2>&1 | grep -vEi "warning|tqdm|fork|tokenizers|avoid using|explicitly set|sub-SMC|word/s"
echo "runtime: ${SECONDS}s"
