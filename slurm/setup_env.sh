#!/bin/bash
# Create / populate the noisy-channel conda env on EITHER platform:
#   * arm64 macOS  -> jax on CPU/Metal, torch CPU/MPS   (for local debugging)
#   * x86_64 Linux -> jax on CUDA 12, torch CPU          (the GPU cluster nodes)
#
# Version pins below are taken verbatim from the working local env so the cluster reproduces it.
# The ONE dep that differs by platform is jax; torch is CPU-only on both (it only loads weights).
#
# Run it once per machine (idempotent):  bash slurm/setup_env.sh
#
# Overridable via env:
#   ENV_NAME (ncgenjax) | GENJAX_REPO | GENJAX_COMMIT | GENJAX_SRC (checkout dir) | CONDA_BASE
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load private, gitignored cluster settings if present (shares GENJAX_SRC / CONDA_BASE / env name).
[ -f "$SLURM_DIR/cluster.env" ] && source "$SLURM_DIR/cluster.env"

ENV_NAME="${ENV_NAME:-${CONDA_ENV:-ncgenjax}}"
GENJAX_REPO="${GENJAX_REPO:-https://github.com/genjax-community/genjax.git}"
GENJAX_COMMIT="${GENJAX_COMMIT:-0fa721649317aff03ffab93af2b32633238083cf}"   # == v0.10.3-19-g0fa72164
GENJAX_SRC="${GENJAX_SRC:-$HOME/genjax}"   # existing checkout is reused; otherwise cloned here

OS="$(uname -s)"; ARCH="$(uname -m)"
echo "Platform: $OS/$ARCH   env: $ENV_NAME"

# ---- locate the conda base and load its shell hook -------------------------------------------
# Resolve the base from `conda info --base` (just the path, on every conda version). Do NOT use
# `mamba info --base`: mamba 2.x (recent miniforge) prints a full info banner there, not a path.
if [ -z "${CONDA_BASE:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    elif [ -n "${CONDA_EXE:-}" ]; then
        CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
    else
        echo "ERROR: conda not found on PATH. 'module load' your miniforge (or install it) first," >&2
        echo "       or set CONDA_BASE to the dir that contains etc/profile.d/conda.sh." >&2
        exit 1
    fi
fi
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || {
    echo "ERROR: no conda.sh under CONDA_BASE='$CONDA_BASE'. Set CONDA_BASE to your real conda base." >&2
    exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
PKG="$(command -v mamba >/dev/null 2>&1 && echo mamba || echo conda)"   # prefer mamba for env creation

# ---- create the env if missing ----------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env '$ENV_NAME' exists; reusing."
else
    echo "creating env '$ENV_NAME'..."
    "$PKG" env create -n "$ENV_NAME" -f "$SLURM_DIR/environment.yml"
fi
conda activate "$ENV_NAME"
python -V

PIP="python -m pip"
$PIP install --upgrade pip

# ---- 1. jax (the platform-specific dependency) ------------------------------------------------
if [ "$OS" = "Darwin" ]; then
    echo "installing jax (CPU, arm64 macOS)..."
    $PIP install "jax==0.5.2" "jaxlib==0.5.1"
else
    echo "installing jax[cuda12] (x86_64 Linux GPU)..."
    # Bundled-CUDA wheels: jax ships its own CUDA libs, so you do NOT need a `module load cuda` for
    # jax -- only a compatible NVIDIA driver on the node. If you prefer the cluster's CUDA modules,
    # swap this for jax[cuda12_local] and load cuda/cudnn in the sbatch.
    $PIP install "jax[cuda12]==0.5.2"
fi

# ---- 2. torch (CPU-only on both platforms) ----------------------------------------------------
if [ "$OS" = "Darwin" ]; then
    $PIP install "torch==2.12.0"
else
    echo "installing torch CPU wheel (linux)..."
    $PIP install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cpu
fi

# ---- 3. the portable pinned deps --------------------------------------------------------------
$PIP install \
    "numpy==1.26.4" "scipy==1.17.1" "wordfreq==3.1.1" \
    "transformers==4.49.0" "tokenizers==0.21.4" "safetensors==0.8.0" "huggingface-hub==0.36.2" \
    "penzai==0.2.5"

# ---- 4. genjax (editable, pinned commit; drags in tfp/jaxtyping/beartype/treescope) -----------
if [ -d "$GENJAX_SRC/.git" ]; then
    echo "reusing genjax checkout at $GENJAX_SRC"
else
    echo "cloning genjax into $GENJAX_SRC ..."
    echo "  (if $GENJAX_REPO is private, clone it yourself / copy your local checkout and set GENJAX_SRC)"
    git clone "$GENJAX_REPO" "$GENJAX_SRC"
fi
git -C "$GENJAX_SRC" fetch --all --tags --quiet || true
git -C "$GENJAX_SRC" checkout "$GENJAX_COMMIT"
$PIP install -e "$GENJAX_SRC"

# ---- verify -----------------------------------------------------------------------------------
echo
echo "=== verification ==="
python - <<'PY'
import jax, jaxlib, penzai, transformers, torch, wordfreq, genjax
print("jax        ", jax.__version__, "| backend:", jax.default_backend())
print("jaxlib     ", jaxlib.__version__)
print("penzai     ", penzai.__version__)
print("transformers", transformers.__version__)
print("torch      ", torch.__version__, "| cuda build:", torch.version.cuda)
print("genjax     ", genjax.__version__)
PY
echo
echo "Done. Sanity-check the model itself with:"
echo "  NC_LM=EleutherAI/pythia-70m PYTHONPATH=$(dirname "$SLURM_DIR")/src python -m genjax_port.pythia_word_caprop --selftest"
echo "On a GPU node 'backend: gpu' should print above; on macOS it will say 'cpu' (or 'metal')."
