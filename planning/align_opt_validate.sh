#!/bin/zsh
# Seed-robustness + alpha-push validation at K=-4.5 (align, gibbs, 70m, P=128).
# The K=-4.5 + alpha=(200,4,4) combined=0.624 headline rests on two high-variance keeps
# (INS-01b, DELTO-02b) recovering at seed 0 -- validate it holds across seeds before trusting it.
# Also probe alpha=(200,8,8) for the residual DEL-restore under-edits (DEL-the, DELTO-02a still ~0).
set -e
cd /Users/thomasclark/mit/noisy_channel_model
SUB=$(cat planning/align_opt_subset.txt)
PY=/Users/thomasclark/mit/openrouter/.conda/envs/ncgenjax/bin/python
run() {  # $1=alpha  $2=seed  $3=outfile
  echo "### $(date +%H:%M:%S) START alpha=$1 seed=$2 -> $3"
  NC_CHANNEL=align NC_REJUV=gibbs NC_ALIGN_SLOPE=-4.5 NC_ALPHA=$1 PYTHONPATH=src PYTHONUNBUFFERED=1 \
    $PY -u -m genjax_port.calibration_word_action_smc 128 $2 ${=SUB} > $3 2>&1
  echo "### $(date +%H:%M:%S) DONE  alpha=$1 seed=$2"
}
run 200,4,4 1 planning/align_opt_a4_s1.txt
run 200,4,4 2 planning/align_opt_a4_s2.txt
run 200,1,1 1 planning/align_opt_a1_s1.txt
run 200,8,8 0 planning/align_opt_a8_s0.txt
echo "### ALL DONE $(date +%H:%M:%S)"
