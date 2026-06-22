#!/bin/zsh
# Map the ins/del-alpha Pareto on the full 40-item set at the chosen K=-4.5 (70m, gibbs, P=128).
# a4 already run; add a1 (isolates the K effect) and a2 (moderate) to bracket the knee.
set -e
cd /Users/thomasclark/mit/noisy_channel_model
SUB=$(cat planning/wa_alpha_subsample.txt)
PY=/Users/thomasclark/mit/openrouter/.conda/envs/ncgenjax/bin/python
run() {  # $1=alpha  $2=out
  echo "### $(date +%H:%M:%S) START full40 K=-4.5 alpha=$1 -> $2"
  NC_CHANNEL=align NC_REJUV=gibbs NC_ALIGN_SLOPE=-4.5 NC_ALPHA=$1 PYTHONPATH=src PYTHONUNBUFFERED=1 \
    $PY -u -m genjax_port.calibration_word_action_smc 128 0 ${=SUB} > $2 2>&1
  echo "### $(date +%H:%M:%S) DONE  alpha=$1"
}
run 200,1,1 planning/align_opt_full40_K4.5_a1.txt
run 200,2,2 planning/align_opt_full40_K4.5_a2.txt
echo "### ALL DONE $(date +%H:%M:%S)"
