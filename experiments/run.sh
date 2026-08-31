#!/bin/bash
# The experiment harness driver (harness plan §5-§6). One place for the whole sequence:
#
#   fetch-tabor -> build -> smoke-local -> probe -> submit -> status -> pull -> collect
#
# Local subcommands (build/smoke-local/probe/collect) run in the ncgenjax conda env on this
# machine. Cluster subcommands (submit/status/pull) drive the MIT ORCD SLURM harness over the
# multiplexed ssh alias `orcd` and REQUIRE the user to have opened the master connection first
# (`ssh -fN orcd` -- password + Duo; this script only ever does `ssh -O check`). run.sh NEVER
# pushes: submit refuses when the cluster is not on the local commit and says what to do.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$EXP_DIR")"
REMOTE_REPO="/orcd/data/rplevy/001/om2/thclark/noisy_channel_model"
SSH_HOST="orcd"
CONDA_ENV="ncgenjax"
PY() { conda run -n "$CONDA_ENV" python "$@"; }

TABOR_URL="https://osf.io/download/y4872/"
TABOR_SHA="f4030aadc67662f22c4d14a8ee5ee6a6b1d089a038bdb8c321079059bd01175e"

usage() {
    sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'
Subcommands:
  fetch-tabor              download data/tabor2004/items.csv (sha256-verified; skipped if present)
  build [args...]          run experiments/build_stimuli.py (stdlib, <1 s)
  smoke-local              <=10-min local end-to-end: smoke set under configs/smoke.env, ONE
                           gibbs+bd P=16 item (exercises the indel stats), then collect
  probe                    local cost probe: planning/bd_mem_probe.py over stimuli/probe.input.jsonl
                           at the main operating point (P=64, off AND gibbs+bd). Record the RESULT
                           lines in experiments/RUNLOG.md BEFORE any submit; size MEM ~ 2x Mac peak
                           RSS and SECONDS_PER_ITEM ~ 3x Mac runtime.
  submit <dataset> <cfg>   cluster submit of stimuli/<dataset>.input.jsonl under configs/<cfg>.env
                           (DRYRUN preflight; commit parity enforced; RUNLOG entry appended).
                           Pass-through env: MEM SECONDS_PER_ITEM SENTENCES_PER_SHARD CPUS
                           MAX_PARALLEL DRYRUN OVERWRITE SKIP_ERRORS NC_LM.
  status [<dataset> <cfg>] squeue summary; with args, the resume-aware remaining count
  pull <dataset>           rsync that dataset's results (all configs; *.viz.json excluded) into
                           local results_nc/, then run collect on it
  collect [dataset...]     experiments/collect.py -> experiments/outputs/
EOF
}

# ---- helpers ---------------------------------------------------------------------------------

need_master() {
    if ! ssh -O check "$SSH_HOST" 2>/dev/null; then
        echo "No ssh master for '$SSH_HOST'. Run:   ssh -fN $SSH_HOST   (password + Duo), then retry." >&2
        exit 1
    fi
}

#: The model knobs a config file may set (everything submit_nc_batch.sh reads except execution
#: sizing). Used both to compose the remote env line and to check the off/bd arm parity.
CFG_VARS=(CHANNEL REJUV PARTICLES N_SEEDS REJUV_LOOKBACK BAND MAX_DIST SEED LM_TEMP INS_RATE TOP
          WDEL WINS ALIGN_SLOPE ACTION_ALPHA UNIFORM_INS NO_DEDUP BD_P_STAY BD_MODE BD_ATTEMPTS
          NO_BD_FUNCWORDS WRITE_VIZ SORT_BY_LENGTH)

resolve_config() {                       # print VAR=VAL lines for configs/<name>.env, sorted
    local name="$1"
    local f="$EXP_DIR/configs/$name.env"
    [ -f "$f" ] || { echo "no such config: $f" >&2; exit 1; }
    ( set +u; . "$f"
      for v in "${CFG_VARS[@]}"; do
          eval "val=\${$v:-}"
          [ -n "$val" ] && echo "$v=$val"
      done ) | sort
}

check_arm_parity() {                     # the two main arms may differ in REJUV and NOTHING else
    local d
    d="$(diff <(resolve_config main_off | grep -v '^REJUV=') \
              <(resolve_config main_bd  | grep -v '^REJUV=') || true)"
    if [ -n "$d" ]; then
        echo "REFUSING: configs/main_off.env and configs/main_bd.env differ in more than REJUV" >&2
        echo "(user decision: the off-vs-rejuv comparison keeps everything else constant):" >&2
        echo "$d" >&2
        exit 1
    fi
}

# ---- subcommands -----------------------------------------------------------------------------

cmd_fetch_tabor() {
    local out="$REPO/data/tabor2004/items.csv"
    if [ -f "$out" ] && echo "$TABOR_SHA  $out" | shasum -a 256 -c - >/dev/null 2>&1; then
        echo "already fetched and verified: $out"
        return 0
    fi
    mkdir -p "$REPO/data/tabor2004"
    echo "fetching $TABOR_URL ..."
    curl -fsSL "$TABOR_URL" -o "$out.tmp"
    echo "$TABOR_SHA  $out.tmp" | shasum -a 256 -c -
    mv "$out.tmp" "$out"
    echo "fetched + verified: $out (see data/tabor2004/SOURCE.md for provenance)"
}

cmd_build() { python3 "$EXP_DIR/build_stimuli.py" "$@"; }

cmd_smoke_local() {
    local rr="$REPO/results_nc"
    echo "== smoke 1/3: the smoke set under configs/smoke.env (forward-only) =="
    ( . "$EXP_DIR/configs/smoke.env"
      PY "$REPO/slurm/run_nc_batch.py" --input "$EXP_DIR/stimuli/smoke.input.jsonl" \
          --results-root "$rr" --sort-by-length --shard-index 0 \
          --channel "$CHANNEL" --rejuv "$REJUV" --particles "$PARTICLES" \
          --n-seeds "$N_SEEDS" --top "$TOP" --no-viz )
    echo "== smoke 2/3: ONE gibbs+bd P=16 item (indel statistics) =="
    PY "$REPO/slurm/run_nc_batch.py" --input "$EXP_DIR/stimuli/smoke.input.jsonl" \
        --results-root "$rr" --sort-by-length --shard-size 1 --min-shard-size 1 --shard-index 0 \
        --channel align --rejuv gibbs+bd --particles 16 --n-seeds 1 --top 10 --no-viz
    echo "== smoke 3/3: collect =="
    PY "$EXP_DIR/collect.py" smoke --results-root "$rr"
    PY -c "
import glob, pandas as pd
for d in sorted(glob.glob('$EXP_DIR/outputs/*/smoke')):
    w = pd.read_csv(d + '/words.csv.gz')
    s = pd.read_csv(d + '/sentences.csv.gz', keep_default_na=False)
    print(f'{d.split(chr(47))[-2][:60]}: sentences ok={int((s.status==\"ok\").sum())}/{len(s)}'
          f'  words rows={len(w)} finite S_nc={int(w.surprisal_nc.notna().sum())}')
"
    echo "smoke-local done. Tables under experiments/outputs/; raw JSON under results_nc/smoke.input/."
}

cmd_probe() {
    local jsonl="$EXP_DIR/stimuli/probe.input.jsonl"
    echo "cost probe over $jsonl at the main operating point (P=64), one process per run"
    echo "(record the RESULT lines in experiments/RUNLOG.md; MEM ~ 2x peak RSS, SECONDS_PER_ITEM ~ 3x runtime;"
    echo " N_SEEDS=4 multiplies per-item time by 4)"
    local n
    n=$(grep -c . "$jsonl")
    for i in $(seq 0 $((n - 1))); do
        local text ctx
        text=$(python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).readlines()[int(sys.argv[2])])['text'])" "$jsonl" "$i")
        ctx=$(python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).readlines()[int(sys.argv[2])])['context'])" "$jsonl" "$i")
        for rejuv in off gibbs+bd; do
            echo "--- probe item $i rejuv=$rejuv ---"
            PY "$REPO/planning/bd_mem_probe.py" "$text" 64 "$rejuv" "$ctx"
        done
    done
}

cmd_submit() {
    local ds="${1:?usage: run.sh submit <dataset> <config>}"
    local cfg="${2:?usage: run.sh submit <dataset> <config>}"
    local input_rel="experiments/stimuli/$ds.input.jsonl"
    [ -f "$REPO/$input_rel" ] || { echo "no input list: $REPO/$input_rel (run.sh build first?)" >&2; exit 1; }
    case "$cfg" in main_off|main_bd) check_arm_parity ;; esac
    need_master

    # Commit parity: the results must be attributable to one commit on both sides. NEVER pushes.
    local lsha rsha
    lsha=$(git -C "$REPO" rev-parse --short HEAD)
    [ -z "$(git -C "$REPO" status --porcelain)" ] || echo "WARNING: local tree is dirty" >&2
    rsha=$(ssh "$SSH_HOST" "cd $REMOTE_REPO && git rev-parse --short HEAD")
    if [ "$lsha" != "$rsha" ]; then
        echo "REFUSING: cluster is at $rsha, local at $lsha. Sync first (asks you, not me):" >&2
        echo "    git push origin \$(git branch --show-current)      # outward-facing: your call" >&2
        echo "    ssh $SSH_HOST 'cd $REMOTE_REPO && git fetch origin && git pull --ff-only origin <branch>'" >&2
        exit 1
    fi

    local envline
    envline="INPUT=$input_rel $(resolve_config "$cfg" | tr '\n' ' ')"
    for v in MEM SECONDS_PER_ITEM SENTENCES_PER_SHARD CPUS MAX_PARALLEL OVERWRITE SKIP_ERRORS NC_LM; do
        eval "val=\${$v:-}"
        [ -n "$val" ] && envline="$envline $v=$val"
    done

    echo "== DRYRUN preflight ($ds x $cfg @ $lsha) =="
    local plan
    plan=$(ssh "$SSH_HOST" "cd $REMOTE_REPO && DRYRUN=1 $envline bash slurm/submit_nc_batch.sh")
    echo "$plan" | sed 's/^/  /'
    local remaining slug
    remaining=$(sed -n 's/^ *REMAINING_ITEMS=//p' <<<"$plan" | head -1)
    slug=$(sed -n 's/^Config dir : //p' <<<"$plan" | head -1 | xargs basename 2>/dev/null || true)
    if [ "${DRYRUN:-0}" = 1 ]; then
        echo "DRYRUN=1 -> not submitting."
        return 0
    fi
    if [ "${remaining:-0}" = 0 ]; then
        echo "nothing to submit ($ds x $cfg is complete)."
        return 0
    fi

    echo "== submitting =="
    local out
    out=$(ssh "$SSH_HOST" "cd $REMOTE_REPO && $envline bash slurm/submit_nc_batch.sh" | tail -20)
    echo "$out" | sed 's/^/  /'
    local jobid
    jobid=$(sed -n 's/^Submitted batch job //p' <<<"$out" | head -1)
    {   echo ""
        echo "### $(date -u +%FT%TZ) — $ds × $cfg"
        echo "- commit: \`$lsha\` (local == cluster)"
        echo "- slug: \`${slug:-?}\`  remaining before submit: ${remaining:-?}"
        echo "- env: \`$envline\`"
        echo "- job id: ${jobid:-<not parsed — see above>}"
        echo "- outcome: (append when finished)"
    } >> "$EXP_DIR/RUNLOG.md"
    echo "RUNLOG entry appended (experiments/RUNLOG.md). Monitor: run.sh status $ds $cfg"
}

cmd_status() {
    need_master
    ssh "$SSH_HOST" "squeue -u thclark" | sed 's/^/  /'
    if [ $# -ge 2 ]; then
        local ds="$1" cfg="$2"
        local envline
        envline="INPUT=experiments/stimuli/$ds.input.jsonl $(resolve_config "$cfg" | tr '\n' ' ')"
        ssh "$SSH_HOST" "cd $REMOTE_REPO && DRYRUN=1 $envline bash slurm/submit_nc_batch.sh" \
            | grep -E "REMAINING_ITEMS|Nothing to do|Config dir" | sed 's/^/  /'
    fi
}

cmd_pull() {
    local ds="${1:?usage: run.sh pull <dataset>}"
    need_master
    mkdir -p "$REPO/results_nc/$ds.input"
    rsync -a --exclude '*.viz.json' \
        "$SSH_HOST:$REMOTE_REPO/results_nc/$ds.input/" "$REPO/results_nc/$ds.input/"
    echo "pulled results_nc/$ds.input/ (viz excluded). Collecting..."
    PY "$EXP_DIR/collect.py" "$ds" --results-root "$REPO/results_nc"
}

cmd_collect() { PY "$EXP_DIR/collect.py" "$@"; }

# ---- dispatch --------------------------------------------------------------------------------
case "${1:-}" in
    fetch-tabor) shift; cmd_fetch_tabor "$@" ;;
    build)       shift; cmd_build "$@" ;;
    smoke-local) shift; cmd_smoke_local "$@" ;;
    probe)       shift; cmd_probe "$@" ;;
    submit)      shift; cmd_submit "$@" ;;
    status)      shift; cmd_status "$@" ;;
    pull)        shift; cmd_pull "$@" ;;
    collect)     shift; cmd_collect "$@" ;;
    -h|--help|help|"") usage ;;
    *) echo "unknown subcommand: $1" >&2; usage >&2; exit 2 ;;
esac
