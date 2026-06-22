#!/bin/zsh
# Alpha (ins/del concentration) sweep at a fixed K, align channel, deployment config.
# K controls SUBSTITUTION edits (SUBW/SUBN); alpha's ins/del entries control DELETION/INSERTION edits
# (DEL*/INS* restoration vs spurious-ins/del junk on keeps). Two independent knobs for the two junk
# sources. Symmetric (200, a, a): a-up = more willing to restore dropped / remove doubled words
# (helps under-editing) but more keep over-edit; a-down = the reverse.  Arg1 = K (align_slope).
set -e
cd /Users/thomasclark/mit/noisy_channel_model
K=${1:?usage: align_opt_alpha_sweep.sh K}
SUB=$(cat planning/align_opt_subset.txt)
PY=/Users/thomasclark/mit/openrouter/.conda/envs/ncgenjax/bin/python
run() {  # $1=alpha-triple  $2=outfile
  echo "### $(date +%H:%M:%S) START alpha=$1 K=$K -> $2"
  NC_CHANNEL=align NC_REJUV=gibbs NC_ALIGN_SLOPE=$K NC_ALPHA=$1 PYTHONPATH=src PYTHONUNBUFFERED=1 \
    $PY -u -m genjax_port.calibration_word_action_smc 128 0 ${=SUB} > $2 2>&1
  echo "### $(date +%H:%M:%S) DONE  alpha=$1"
}
run 200,0.5,0.5 planning/align_opt_a0.5_gibbs.txt
run 200,2,2     planning/align_opt_a2_gibbs.txt
run 200,4,4     planning/align_opt_a4_gibbs.txt
echo "### ALL DONE $(date +%H:%M:%S)"
