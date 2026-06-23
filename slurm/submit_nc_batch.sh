#!/bin/bash
# Submit a noisy-channel batch as a throttled, requeue-able SLURM job array.
#
# One run = one (input file x config). The input's sentences are split into shards of
# SENTENCES_PER_SHARD; each shard is one array task that loads the model once and processes its
# sentences serially (amortizing the ~minutes of model-load + JIT). Results land in a
# config-encoded directory so different configs never collide and are trivial to compare later.
#
# Everything below is overridable from the environment, e.g.:
#   INPUT=data/battery.txt CHANNEL=align REJUV=gibbs+bd PARTICLES=128 MAX_PARALLEL=20 \
#       bash slurm/submit_nc_batch.sh
#
# Re-running the SAME command resumes: finished items are skipped, only shards with remaining work
# are submitted. Set OVERWRITE=1 to recompute everything. Set DRYRUN=1 to print the generated sbatch
# script and the plan WITHOUT submitting (no SLURM needed -- good for inspecting a config first).
set -euo pipefail

# Resolve paths, then load your PRIVATE, gitignored cluster settings (slurm/cluster.env) if present.
# cluster.env supplies the org-specific values (partitions, modules, paths) WITHOUT committing them to
# the public repo -- copy slurm/cluster.env.example to slurm/cluster.env and fill it in. Precedence:
# explicit command-line env (FOO=bar bash submit_nc_batch.sh) > cluster.env > the defaults below.
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SLURM_DIR")"
RUNNER="$SLURM_DIR/run_nc_batch.py"
[ -f "$SLURM_DIR/cluster.env" ] && source "$SLURM_DIR/cluster.env"

# ============================================================================================
# 1. CLUSTER ENVIRONMENT  --  set these in slurm/cluster.env (NOT here -- keep them out of the public repo)
#    Nothing is preinstalled, so this is the part most likely to need tweaking. The defaults assume a
#    conda/mamba env named $CONDA_ENV created by slurm/setup_env.sh.
# ============================================================================================
PARTITIONS="${PARTITIONS:?set PARTITIONS in slurm/cluster.env to your GPU partition names, comma-separated, or pass PARTITIONS=...}"
GRES="${GRES:-gpu:1}"                 # pythia-70m is tiny: any single GPU. Pin a type only if required, e.g. gpu:a100:1
CPUS="${CPUS:-4}"
CONDA_ENV="${CONDA_ENV:-ncgenjax}"    # the CLUSTER-side env name (NOT the local arm64 env, which won't exist on x86 nodes)
CONDA_BASE="${CONDA_BASE:-}"          # e.g. $HOME/miniforge3 ; empty -> discovered via `conda info --base`
MODULE_PURGE="${MODULE_PURGE:-module purge}"          # set to ':' to disable
MODULE_LOADS="${MODULE_LOADS:-}"      # e.g. 'module load miniforge' ; semicolon-separate multiples

# ============================================================================================
# 2. MODEL CONFIG  --  the experiment knobs (each distinct value -> its own results directory)
# ============================================================================================
INPUT="${INPUT:?set INPUT=path/to/sentences.txt (one observed sentence per line)}"
NC_LM="${NC_LM:-EleutherAI/pythia-70m}"   # also selects the LM inside the model (read from this env var)
CHANNEL="${CHANNEL:-align}"
REJUV="${REJUV:-gibbs+bd}"
REJUV_LOOKBACK="${REJUV_LOOKBACK:-6}"
PARTICLES="${PARTICLES:-128}"
BAND="${BAND:-2}"
MAX_DIST="${MAX_DIST:-2}"
SEED="${SEED:-0}"
LM_TEMP="${LM_TEMP:-1.0}"
INS_RATE="${INS_RATE:-0.02}"
TOP="${TOP:-5}"
# Optional overrides -- leave empty to use the model/channel defaults (they then stay out of the dir name):
WDEL="${WDEL:-}"                # missing-word penalty (default -9.0)
WINS="${WINS:-}"               # flat spurious-word cost override
ALIGN_SLOPE="${ALIGN_SLOPE:-}"  # align K (default -4.5)
ACTION_ALPHA="${ACTION_ALPHA:-}"  # 'copy,sub,ins,del' or 'align,ins,del'
UNIFORM_INS="${UNIFORM_INS:-0}" # 1 -> legacy flat -log(vocab) insertion cost
NO_DEDUP="${NO_DEDUP:-0}"       # 1 -> disable the exact post-resample dedup (only to A/B cost)
BD_P_STAY="${BD_P_STAY:-0.0}"
BD_MODE="${BD_MODE:-gibbs}"
BD_ATTEMPTS="${BD_ATTEMPTS:-1}"
NO_BD_FUNCWORDS="${NO_BD_FUNCWORDS:-0}"

# ============================================================================================
# 3. BATCH / SLURM EXECUTION
# ============================================================================================
RESULTS_ROOT="${RESULTS_ROOT:-results_nc}"
SENTENCES_PER_SHARD="${SENTENCES_PER_SHARD:-8}"   # sentences per array task (amortizes model load)
MAX_PARALLEL="${MAX_PARALLEL:-20}"                # array throttle: never more than this many tasks at once
MEM="${MEM:-12G}"                 # host RAM. MEASURE with `seff` on the first shard and tighten (see README)
SECONDS_PER_ITEM="${SECONDS_PER_ITEM:-240}"       # used to auto-size --time
MODEL_OVERHEAD_S="${MODEL_OVERHEAD_S:-900}"       # model load + JIT compile budget added to --time
MAX_TIME="${MAX_TIME:-3:59:00}"                   # cap on --time (match your partition's short-QOS limit)
WRITE_VIZ="${WRITE_VIZ:-1}"       # 1 -> also write the heavy directly-viz-loadable trace json per item
OVERWRITE="${OVERWRITE:-0}"
SKIP_ERRORS="${SKIP_ERRORS:-0}"   # 1 -> do not retry items that previously errored
DRYRUN="${DRYRUN:-0}"
PREFLIGHT_PYTHON="${PREFLIGHT_PYTHON:-python3}"   # any python3 (the --plan mode is stdlib-only)

# ============================================================================================
# (no edits needed below)
# ============================================================================================
# absolutize the paths SLURM tasks will reference (their cwd is undefined)
INPUT_ABS="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
case "$RESULTS_ROOT" in /*) RR_ABS="$RESULTS_ROOT";; *) RR_ABS="$REPO/$RESULTS_ROOT";; esac
[ -f "$INPUT_ABS" ] || { echo "INPUT not found: $INPUT_ABS" >&2; exit 1; }

# Fixed config args, shared verbatim by the preflight (--plan) and the per-shard run so the
# resolved output directory and the resume bookkeeping match exactly. All values are space-free.
CFG="--channel $CHANNEL --particles $PARTICLES --band $BAND --max-dist $MAX_DIST"
CFG="$CFG --rejuv $REJUV --rejuv-lookback $REJUV_LOOKBACK --seed $SEED"
CFG="$CFG --lm-temp $LM_TEMP --ins-rate $INS_RATE --top $TOP"

# Optional / behavior flags -- also shared by plan and run.
EXTRA=""
[ -n "$WDEL" ]          && EXTRA="$EXTRA --wdel $WDEL"
[ -n "$WINS" ]          && EXTRA="$EXTRA --wins $WINS"
[ -n "$ALIGN_SLOPE" ]   && EXTRA="$EXTRA --align-slope $ALIGN_SLOPE"
[ -n "$ACTION_ALPHA" ]  && EXTRA="$EXTRA --action-alpha $ACTION_ALPHA"
[ "$UNIFORM_INS" = 1 ]  && EXTRA="$EXTRA --uniform-ins"
[ "$NO_DEDUP" = 1 ]     && EXTRA="$EXTRA --no-dedup"
[ "$BD_P_STAY" != "0.0" ] && EXTRA="$EXTRA --bd-p-stay $BD_P_STAY"
[ "$BD_MODE" != "gibbs" ] && EXTRA="$EXTRA --bd-mode $BD_MODE"
[ "$BD_ATTEMPTS" != "1" ] && EXTRA="$EXTRA --bd-attempts $BD_ATTEMPTS"
[ "$NO_BD_FUNCWORDS" = 1 ] && EXTRA="$EXTRA --no-bd-funcwords"
[ "$WRITE_VIZ" = 1 ] || EXTRA="$EXTRA --no-viz"
[ "$OVERWRITE" = 1 ]    && EXTRA="$EXTRA --overwrite"
[ "$SKIP_ERRORS" = 1 ]  && EXTRA="$EXTRA --skip-errors"

# ---- Preflight: write the manifest + find which shards still have work (resume-aware) ----------
echo "Preflight (resume-aware plan)..."
PLAN="$("$PREFLIGHT_PYTHON" "$RUNNER" --plan \
        --input "$INPUT_ABS" --results-root "$RR_ABS" --shard-size "$SENTENCES_PER_SHARD" \
        $CFG $EXTRA)"
echo "$PLAN" | sed 's/^/  /'
OUTPUT_DIR="$(sed -n 's/^OUTPUT_DIR=//p'    <<<"$PLAN")"
NUM_SHARDS="$(sed -n 's/^NUM_SHARDS=//p'    <<<"$PLAN")"
REMAINING="$(sed -n 's/^REMAINING_ITEMS=//p' <<<"$PLAN")"
SHARDS="$(sed -n 's/^SHARDS_WITH_WORK=//p'  <<<"$PLAN")"
LOGS_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOGS_DIR"

if [ -z "${SHARDS:-}" ]; then
    echo "Nothing to do: all ${NUM_SHARDS} shards complete for this config (use OVERWRITE=1 to redo)."
    echo "Results dir: $OUTPUT_DIR"
    exit 0
fi

# Array spec: only the shards with remaining work, throttled to MAX_PARALLEL.
ARRAY_SPEC="${SHARDS}%${MAX_PARALLEL}"

# ---- Auto-size --time from the shard size, capped at MAX_TIME ----------------------------------
to_secs() { awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else if (NF==2) print $1*60+$2; else print $1 }' <<<"$1"; }
EST=$(( MODEL_OVERHEAD_S + SENTENCES_PER_SHARD * SECONDS_PER_ITEM ))
CAP=$(to_secs "$MAX_TIME")
TSEC=$(( EST < CAP ? EST : CAP ))
TIME_STR=$(printf '%d:%02d:%02d' $((TSEC/3600)) $(((TSEC%3600)/60)) $((TSEC%60)))
if [ "$EST" -gt "$CAP" ]; then
    echo "WARNING: estimated shard time ${EST}s exceeds MAX_TIME ${MAX_TIME}; a shard may time out and"
    echo "         resume on a later submit. Lower SENTENCES_PER_SHARD (now $SENTENCES_PER_SHARD) to fit."
fi

INPUT_STEM="$(basename "$INPUT_ABS")"; INPUT_STEM="${INPUT_STEM%.*}"
JOB_NAME="nc_${INPUT_STEM}_${CHANNEL}_${REJUV//+/}"

JOB_SCRIPT="$LOGS_DIR/submit.sbatch"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$LOGS_DIR/shard_%a_%A.out
#SBATCH --error=$LOGS_DIR/shard_%a_%A.err
#SBATCH --partition=$PARTITIONS
#SBATCH --gres=$GRES
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME_STR
#SBATCH --array=$ARRAY_SPEC
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail

# --- environment (edit slurm/submit_nc_batch.sh section 1 if this is wrong for your cluster) ---
$MODULE_PURGE 2>/dev/null || true
${MODULE_LOADS:-:}
if [ -n "$CONDA_BASE" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    source "\$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

export NC_LM="$NC_LM"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # grow GPU memory on demand (kinder to the node, lower host RSS)

echo "host=\$(hostname) task=\$SLURM_ARRAY_TASK_ID job=\$SLURM_JOB_ID gpu=\${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L 2>/dev/null || true

python "$RUNNER" \\
    --shard-index "\$SLURM_ARRAY_TASK_ID" \\
    --input "$INPUT_ABS" --results-root "$RR_ABS" --shard-size "$SENTENCES_PER_SHARD" \\
    --est-seconds-per-item "$SECONDS_PER_ITEM" \\
    $CFG $EXTRA
EOF

echo
echo "Config dir : $OUTPUT_DIR"
echo "Logs       : $LOGS_DIR/shard_<task>_<jobid>.{out,err}"
echo "Shards     : ${NUM_SHARDS} total, $(wc -w <<<"${SHARDS//,/ }" | tr -d ' ') with work -> array=$ARRAY_SPEC"
echo "Remaining  : $REMAINING items"
echo "Per task   : --time=$TIME_STR --mem=$MEM --gres=$GRES --cpus-per-task=$CPUS"
echo "Sbatch     : $JOB_SCRIPT"

if [ "$DRYRUN" = 1 ]; then
    echo
    echo "DRYRUN=1 -> not submitting. Generated sbatch script:"
    echo "-------------------------------------------------------------------"
    cat "$JOB_SCRIPT"
    exit 0
fi

sbatch "$JOB_SCRIPT"
echo "Submitted. Monitor: squeue -u \$USER   |   tail -f $LOGS_DIR/shard_*_*.out"
