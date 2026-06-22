#!/bin/zsh
# Signal-quality test: align channel on pythia-410m (vs 70m default). The prior diagnosis is that the
# over-editing (restructure junk) AND under-editing (DEL-determiner items the 70m LM doesn't strongly
# prefer to fix) are 70m signal-quality problems. A stronger LM should raise genuine-correction gain
# without raising spurious-edit gain -- helping BOTH metrics, unlike the linear K/alpha tradeoffs.
# Slower (410m ~6x params). 12-item subset only. Args: K [alpha-triple].
set -e
cd /Users/thomasclark/mit/noisy_channel_model
K=${1:?usage: align_opt_410m.sh K [alpha]}
ALPHA=${2:-200,1,1}
SUB=$(cat planning/align_opt_subset.txt)
PY=/Users/thomasclark/mit/openrouter/.conda/envs/ncgenjax/bin/python
OUT=planning/align_opt_410m_K${K}_a${ALPHA}.txt
echo "### $(date +%H:%M:%S) START 410m K=$K alpha=$ALPHA -> $OUT"
NC_LM=EleutherAI/pythia-410m NC_CHANNEL=align NC_REJUV=gibbs NC_ALIGN_SLOPE=$K NC_ALPHA=$ALPHA \
  PYTHONPATH=src PYTHONUNBUFFERED=1 \
  $PY -u -m genjax_port.calibration_word_action_smc 128 0 ${=SUB} > $OUT 2>&1
echo "### $(date +%H:%M:%S) DONE 410m -> $OUT"
