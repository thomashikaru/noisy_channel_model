#!/usr/bin/env bash
# Posterior-stability benchmark driver for the noisy-channel model.
# Runs a list of configs sequentially via the resume-aware slurm/run_nc_batch.py worker (NO slurm).
# Each config -> its own config-encoded results dir under $ROOT; per-seed + evidence-merged records.
# Cheap configs first so early signal arrives fast. Resume-safe: re-running skips finished items.
#
# Usage:  bash planning/bench_driver.sh <phaseA|phaseB|phaseC>
set -u
cd /Users/thomasclark/mit/noisy_channel_model
PY=/Users/thomasclark/mit/openrouter/.conda/envs/ncgenjax/bin/python
INPUT=planning/bench_sentences.txt
ROOT=planning/bench_results
NSEEDS=6
COMMON="--input $INPUT --results-root $ROOT --shard-index 0 --shard-size 8 --sort-by-length \
  --channel align --no-viz --top 8 --n-seeds $NSEEDS"
mkdir -p "$ROOT/logs"

# run_cfg LM P REJUV LOOKBACK  -> one config (all 3 sentences x NSEEDS), logged per-config
run_cfg() {
  local lm="$1" P="$2" rej="$3" lb="$4"
  local tag="lm-$(basename $lm)__P${P}__${rej}__lb${lb}"
  local log="$ROOT/logs/${tag}.log"
  echo "[$(date +%H:%M:%S)] START $tag" | tee -a "$ROOT/logs/master.log"
  NC_LM="$lm" PYTHONPATH=src $PY -u slurm/run_nc_batch.py $COMMON \
    --particles "$P" --rejuv "$rej" --rejuv-lookback "$lb" > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END   $tag (rc=$rc)" | tee -a "$ROOT/logs/master.log"
}

P70=EleutherAI/pythia-70m
P160=EleutherAI/pythia-160m

case "${1:-}" in
  phaseA)
    # Core 2D grid: P x rejuv, lookback=6, pythia-70m. Cheap (off) -> expensive (gibbs+bd).
    for P in 16 32 64 128; do run_cfg $P70 $P off       6; done
    for P in 16 32 64 128; do run_cfg $P70 $P gibbs     6; done
    for P in 16 32 64 128; do run_cfg $P70 $P gibbs+bd  6; done
    ;;
  phaseB)
    # Lookback spur at P=64, gibbs+bd (lb=6 already in phaseA grid).
    run_cfg $P70 64 gibbs+bd 2
    run_cfg $P70 64 gibbs+bd 12
    ;;
  phaseC)
    # LM spur: pythia-160m at P=64.
    run_cfg $P160 64 off      6
    run_cfg $P160 64 gibbs+bd 6
    ;;
  *) echo "usage: $0 <phaseA|phaseB|phaseC>"; exit 2 ;;
esac
echo "[$(date +%H:%M:%S)] ALL DONE for ${1}" | tee -a "$ROOT/logs/master.log"
