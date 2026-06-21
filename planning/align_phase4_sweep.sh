#!/bin/zsh
# Phase 4 (planning/ALIGN_ACTION_CHANNEL_PLAN.md sec 6): align battery regression + K-sweep.
# Run from repo root with the ncgenjax env active. Outputs land in planning/.
#   (1) FULL 40-item subsample under align at the default K=log(1/26), rejuv=gibbs -- the apples-to-apples
#       L-retention / E-correction check vs the word_action gibbs@200 baseline (wa_alpha_sweep_gibbs_200_1_1_1.txt).
#   (2) K-SWEEP on the SUB items (the items align actually moves) at K in {log1/8, log1/15, log1/26, log1/40}.
set -e
SUB="SUBW-01a SUBW-01b SUBW-02a SUBW-02b SUBN-01a SUBN-01b SUBN-02a SUBN-02b"
P=128
export NC_LM=EleutherAI/pythia-70m NC_REJUV=gibbs NC_CHANNEL=align

run() { PYTHONPATH=src python -u -m genjax_port.calibration_word_action_smc "$@"; }

echo "### (1) full subsample, align K=log(1/26) default, gibbs"
SUBSAMPLE=$(cat planning/wa_alpha_subsample.txt)
run $P 0 ${=SUBSAMPLE} > planning/align_subsample_K26_gibbs.txt 2>&1

echo "### (2) K-sweep on SUB items"
for KV in -2.0794 -2.7081 -3.2581 -3.6889; do
  KN=$(echo $KV | tr -d '-.' )
  NC_ALIGN_SLOPE=$KV run $P 0 ${=SUB} > planning/align_sweep_K${KN}_gibbs.txt 2>&1
done
echo "### done"
