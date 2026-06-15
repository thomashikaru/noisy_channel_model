#!/usr/bin/env bash
# Run the GENJAX-NATIVE noisy-channel filter (the migration port) on one sentence and print the
# inferred alternatives + runtime. This is `smc_substitution.run_smc_substitution` (run.py
# --filter native): a word-scan SMC built on the genjax @gen Switch word model, handling copy /
# substitution (incl. BPE-token-count typos like "inflection"->"infection") / deletion / insertion.
# It is the filter the rejuvenation work (rejuvenation.py) extends; see src/genjax_port/MIGRATION_PLAN.md.
#
# Counterpart: ./run_example.sh runs the original hand-rolled REFERENCE filter (--filter unified),
# which additionally dedups LM forwards. Behavior matches within Monte-Carlo noise; the native
# filter has no dedup and recompiles per new sentence length (see the latency note in MIGRATION_PLAN).
#
#   Usage:  ./run_example_native.sh [particle_count] [max_edit_distance]
#   e.g.    ./run_example_native.sh 200        # 200 particles, default edit distance 2
#           ./run_example_native.sh 64 3       # 64 particles, substitution candidates up to dist 3
#   For finer control (deletion budget, disabling insertion) call run.py directly:
#           python -m src.genjax_port.run --filter native --max_deletions 2 --no_insertion ...

# ---- edit these ----
SENTENCE="The little boy licked the big round ball into the net."
PARTICLES="${1:-32}"          # default 32
MAX_DIST="${2:-3}"            # max char edit distance for word-substitution candidates
# --------------------

cd "$(dirname "$0")"
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh
conda activate ncgenjax
export TOKENIZERS_PARALLELISM=false

SECONDS=0
PYTHONPATH=. python -m src.genjax_port.run \
  --filter native --sentence "$SENTENCE" --particles "$PARTICLES" --max_dist "$MAX_DIST" \
  2>&1 | grep -vEi "warning|tqdm|fork|tokenizers|avoid using|explicitly set|sub-SMC|word/s"
echo "runtime: ${SECONDS}s"
