#!/bin/zsh
# K-sweep for the error-correction calibration goal (align channel, deployment config).
# Phase 4 only swept K in [-4.1, -2.7] (the garage-flip window); the word_action-equivalent
# substitution threshold is K=-8.56. The user's goal drops the garage constraint, so we probe
# the unexplored sharp-K range to suppress substitution over-editing / restructure junk.
# 12-item bimodal subset; rejuv=gibbs (deployment); P=128; pythia-70m; alpha=(200,1,1).
set -e
cd /Users/thomasclark/mit/noisy_channel_model
SUB=$(cat planning/align_opt_subset.txt)
PY=/Users/thomasclark/mit/openrouter/.conda/envs/ncgenjax/bin/python  # env python direct (conda run buffers stdout)
run() {  # $1=K  $2=outfile
  echo "### $(date +%H:%M:%S) START K=$1 -> $2"
  NC_CHANNEL=align NC_REJUV=gibbs NC_ALIGN_SLOPE=$1 PYTHONPATH=src PYTHONUNBUFFERED=1 \
    $PY -u -m genjax_port.calibration_word_action_smc 128 0 ${=SUB} \
    > $2 2>&1
  echo "### $(date +%H:%M:%S) DONE  K=$1"
}
run -4.5 planning/align_opt_K4.5_gibbs.txt
run -5.5 planning/align_opt_K5.5_gibbs.txt
run -8.0 planning/align_opt_K8.0_gibbs.txt
echo "### ALL DONE $(date +%H:%M:%S)"
