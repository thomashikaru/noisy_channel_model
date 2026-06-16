#!/usr/bin/env bash
# Run the GENJAX-NATIVE noisy-channel model with its current best configuration on one sentence and
# print the inferred alternatives + runtime. This is
#   run.py --filter native --conditional_rejuv
# (rejuv_bridge.run_smc_conditional_rejuv_aligned): the word-scan SMC filtering sweep does
# copy / substitution / deletion / insertion, and after each word's resample a surprisal-gated,
# vectorized SUBSTITUTION rejuvenation pass revises recent words using later context. The add/delete
# capability lives in the forward filter; the interleaved rejuvenation is substitution-only (shape
# preserving) and locates each word's token via the per-particle alignment, so forward
# deletions/insertions don't misalign it. See src/genjax_port/MIGRATION_PLAN.md, R2_PLAN.md.
#
# Uses pythia-70m for fast iteration (set NC_LM=EleutherAI/pythia-410m for the stronger LM). NB: 70m
# is weak enough that it substitutes short words fairly freely.
#
#   Usage:  ./run_example_native.sh [particle_count] [max_edit_distance]
#   e.g.    ./run_example_native.sh 128
#           ./run_example_native.sh 64 3

# ---- edit these ----
SENTENCE="The little boy licked the ball into the net."      # licked->kicked (sub) + omitted 'the' x2 (deletion recovers them)
PARTICLES="${1:-64}"             # number of SMC particles
MAX_DIST="${2:-2}"               # max char edit distance for word-substitution candidates (SymSpell)
MAX_DELETIONS=1                  # forward omitted-word reconstructions per gap (0 disables deletion)

# interleaved substitution rejuvenation (surprisal-gated, vectorized over particles)
LOOKBACK=2                       # words of context to revisit on each rejuvenation event
LOGPROB_THRESH=3.0               # gate CENTER on (contextual - unigram) surprisal: higher => less often
LOGPROB_SPREAD=1.0               # surprisal-gate STEEPNESS
REJUV_SWEEPS=2                   # MH sweeps over the lookback window per rejuvenation event
NC_LM="${NC_LM:-EleutherAI/pythia-70m}"    # set NC_LM=EleutherAI/pythia-410m for the stronger LM
# --------------------

cd "$(dirname "$0")"
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh
conda activate ncgenjax
export TOKENIZERS_PARALLELISM=false

SECONDS=0
NC_LM="$NC_LM" PYTHONPATH=. python -m src.genjax_port.run \
  --filter native \
  --sentence "$SENTENCE" --particles "$PARTICLES" --max_dist "$MAX_DIST" \
  --max_deletions "$MAX_DELETIONS" --lookback "$LOOKBACK" \
  --logprob_thresh "$LOGPROB_THRESH" --logprob_spread "$LOGPROB_SPREAD" \
  --rejuv_sweeps "$REJUV_SWEEPS" \
  2>&1 | grep -vEi "warning|tqdm|fork|tokenizers|avoid using|explicitly set|sub-SMC|word/s"
echo "runtime: ${SECONDS}s"
