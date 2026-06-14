#!/usr/bin/env bash
# Run the unified Genjax noisy-channel model on one sentence and print the inferred
# alternatives + runtime.  The unified word-scan filter handles copy / substitution
# (including BPE-token-count typos like "inflection"->"infection") / insertion / deletion
# in one pass, with dedup LM forwards.
#   Usage:  ./run_example.sh [particle_count] [max_edit_distance]
#   e.g.    ./run_example.sh 200        # 200 particles, default edit distance 2
#           ./run_example.sh 64 3       # 64 particles, substitution candidates up to dist 3

# ---- edit these ----
SENTENCE="The medics treated the wound to prevent an inflection."
PARTICLES="${1:-32}"          # default 32
MAX_DIST="${2:-2}"            # max char edit distance for word-substitution candidates
# --------------------

cd "$(dirname "$0")"
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh
conda activate ncgenjax
export TOKENIZERS_PARALLELISM=false

SECONDS=0
PYTHONPATH=. python -m src.genjax_port.run \
  --sentence "$SENTENCE" --particles "$PARTICLES" --max_dist "$MAX_DIST" \
  2>&1 | grep -vEi "warning|tqdm|fork|tokenizers|avoid using|explicitly set|unified-SMC|word/s"
echo "runtime: ${SECONDS}s"
