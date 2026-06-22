#!/bin/zsh
# Full 40-item calibration subsample validation of the finalist align config, comparable to the
# prior ALIGN_PHASE4_RESULTS table (which used this same 40-item set). 70m, gibbs, P=128.
# Args: K alpha seed outfile
set -e
cd /Users/thomasclark/mit/noisy_channel_model
K=${1:?K} ALPHA=${2:?alpha} SEED=${3:-0} OUT=${4:?outfile}
SUB=$(cat planning/wa_alpha_subsample.txt)
PY=/Users/thomasclark/mit/openrouter/.conda/envs/ncgenjax/bin/python
echo "### $(date +%H:%M:%S) START full40 K=$K alpha=$ALPHA seed=$SEED -> $OUT"
NC_CHANNEL=align NC_REJUV=gibbs NC_ALIGN_SLOPE=$K NC_ALPHA=$ALPHA PYTHONPATH=src PYTHONUNBUFFERED=1 \
  $PY -u -m genjax_port.calibration_word_action_smc 128 $SEED ${=SUB} > $OUT 2>&1
echo "### $(date +%H:%M:%S) DONE full40 -> $OUT"
