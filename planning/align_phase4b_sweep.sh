#!/bin/zsh
# Phase 4b (ALIGN_ACTION_CHANNEL_PLAN sec 6): does a SHARPER K recover the battery KEEP-retention the
# K=log(1/26) run lost (L 0.99->0.76, junk 0->9) while keeping the garage-class fix? Run from repo root,
# ncgenjax active. K=log(1/26)=-3.258 is the over-editing point; sharper = more negative.
set -e
P=128
export NC_LM=EleutherAI/pythia-70m
runc() { PYTHONPATH=src python -u -m genjax_port.calibration_word_action_smc "$@"; }
SUBSAMPLE=$(cat planning/wa_alpha_subsample.txt)
SUB="SUBW-01a SUBW-01b SUBW-02a SUBW-02b SUBN-01a SUBN-01b SUBN-02a SUBN-02b"

# (A) garage flip-survival across sharpening K (single sentence; fast). Does the garage fix survive a
# sharper K that we'd need to curb over-editing? Each prints the top reading.
echo "### (A) garage at sharpening K"
for KV in -3.2581 -3.6889 -4.0943 -4.6052; do
  echo "--- K=$KV ---"
  PYTHONPATH=src python -u -m genjax_port.pythia_word_caprop \
    --sentence "The garage needs to be tossed out." --particles 128 --rejuv gibbs \
    --channel align --align_slope $KV --top 2 2>/dev/null | grep -E "p=|inferred"
done

# (B) full subsample at the SHARPEST plan K (log 1/40) -- the retention-recovery test vs the K26 run.
echo "### (B) full subsample K=log(1/40)=-3.6889"
NC_CHANNEL=align NC_REJUV=gibbs NC_ALIGN_SLOPE=-3.6889 runc $P 0 ${=SUBSAMPLE} \
  > planning/align_subsample_K40_gibbs.txt 2>&1

# (C) K-sweep on the SUB items (E/L trade-off as K sharpens); labels fixed (avoid tr).
echo "### (C) SUB K-sweep"
for pair in "15:-2.7081" "26:-3.2581" "40:-3.6889" "60:-4.0943"; do
  KN=${pair%%:*}; KV=${pair##*:}
  NC_CHANNEL=align NC_REJUV=gibbs NC_ALIGN_SLOPE=$KV runc $P 0 ${=SUB} \
    > planning/align_sweep_K${KN}_gibbs.txt 2>&1
done
echo "### done"
