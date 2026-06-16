#!/usr/bin/env bash
# Run the GENJAX-NATIVE noisy-channel model with its MAXIMAL inference + rejuvenation machinery on one
# sentence and print the inferred alternatives + runtime. This is
#   run.py --filter native --add_delete --rejuvenate
# (rejuv_bridge.run_smc_add_delete with sub_flip=True): the word-scan SMC filtering sweep, followed by
# a single POST-SWEEP reanalysis pass that revises BOTH substitutions (R1 substitution-flip) AND
# add/deletes (R2 trans-dimensional move) on each word using full-sentence context, vectorized over
# particles. See src/genjax_port/R2_PLAN.md, MIGRATION_PLAN.md.
#
# Uses pythia-70m for fast iteration (set NC_LM=EleutherAI/pythia-410m for the stronger LM). NB: 70m
# is weak enough that with substitution on (max_dist > 0) it tends to substitute short words away,
# e.g. go->to, home->come; pass max_dist 0 to isolate the add/delete (omitted-word) behavior.
#
# v1 REJUVENATION SCOPE: single-token observed words. A sentence with a multi-token word (e.g.
# "experimemt") gracefully falls back to the native filter (which still corrects it) -- never an
# error, never less capable than --filter native. Use single-token words to exercise the reanalysis.
#
#   Usage:  ./run_example_native.sh [particle_count] [max_edit_distance]
#   e.g.    ./run_example_native.sh 128
#           ./run_example_native.sh 64 0      # max_dist 0: pure add/delete, no substitution

# ---- edit these ----
SENTENCE="The little boy kicked ball into net."      # 'to' omitted before 'go'; all single-token
PARTICLES="${1:-64}"             # number of SMC particles
MAX_DIST="${2:-2}"               # max char edit distance for word-substitution candidates (SymSpell)
REJUV_SWEEPS=2                   # MH sweeps of the post-sweep reanalysis (each: add/delete + sub-flip per word)
                                 # NB: the post-sweep cost ~ sweeps x words^2 LM forwards (each move
                                 # re-scores the sentence via the @gen trace edit); long sentences /
                                 # many sweeps are slow. max_dist 0 (no sub-flip work) is much cheaper.
NC_LM="${NC_LM:-EleutherAI/pythia-70m}"    # set NC_LM=EleutherAI/pythia-410m for the stronger LM
# --------------------

cd "$(dirname "$0")"
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh
conda activate ncgenjax
export TOKENIZERS_PARALLELISM=false

SECONDS=0
NC_LM="$NC_LM" PYTHONPATH=. python -m src.genjax_port.run \
  --filter native --add_delete --rejuvenate \
  --sentence "$SENTENCE" --particles "$PARTICLES" --max_dist "$MAX_DIST" \
  --rejuv_sweeps "$REJUV_SWEEPS" \
  2>&1 | grep -vEi "warning|tqdm|fork|tokenizers|avoid using|explicitly set|sub-SMC|word/s"
echo "runtime: ${SECONDS}s"
