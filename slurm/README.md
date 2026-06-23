# SLURM batch harness for the noisy-channel model

Run the pair-HMM noisy-channel model over a file of sentences on a SLURM GPU cluster: many sentences
in parallel, fine-grained config control, organized outputs, and resume-on-preemption.

```
slurm/
  environment.yml      # base conda env (python + pip + git)
  setup_env.sh         # installs all deps, cross-platform (arm64 mac + x86 CUDA). Run once per machine.
  run_nc_batch.py      # the per-shard worker (load model once, loop sentences, resume, atomic writes)
  submit_nc_batch.sh   # build the config-encoded output dir + submit a throttled, requeue-able array
  cluster.env.example  # template for your cluster settings -> copy to cluster.env (gitignored)
  sentences.example.txt
  README.md            # you are here
```

## 0. Keep your cluster settings private

The scripts here are generic; your **org-specific values** (partition names, modules, paths) live in
`slurm/cluster.env`, which is **gitignored and never committed** — so the harness can sit in the public
repo while your cluster config stays out of it. Both `submit_nc_batch.sh` and `setup_env.sh` source it
automatically if present.

```bash
cp slurm/cluster.env.example slurm/cluster.env
$EDITOR slurm/cluster.env        # fill in PARTITIONS, MODULE_LOADS, CONDA_ENV, ...
```

Precedence is **command-line env > `cluster.env` > built-in defaults**, so you can still override any
value per-run (`PARTITIONS=foo bash slurm/submit_nc_batch.sh`). ⚠️ Never `git add slurm/cluster.env`.
On the cluster, create its own `cluster.env` once (it's untracked, so a `git pull` won't touch it).
```

The model itself still runs exactly as before — `python -m genjax_port.pythia_word_caprop --sentence ...`.
This harness wraps it: it processes one sentence per call internally but **loads the model once per
shard** so the ~minutes of Pythia-load + JIT-compile is amortized over many sentences.

---

## 1. One-time: build the environment (both machines)

Nothing is preinstalled on the cluster, and the local `ncgenjax` env is **arm64** so it won't exist on
the x86 nodes. `setup_env.sh` reproduces the working local env on either platform — the only dep that
differs is **jax** (CPU/Metal on mac, CUDA 12 on the cluster); torch is CPU-only on both (it just loads
weights; jax/penzai do the compute).

```bash
# on the cluster login node (after installing miniforge and `module load`-ing it if needed):
bash slurm/setup_env.sh
# locally on the mac, to reproduce/debug (point GENJAX_SRC at your existing genjax checkout):
GENJAX_SRC=/path/to/your/genjax bash slurm/setup_env.sh
```

It pins every version to what works locally and installs `genjax` editable at commit `0fa72164`
(`v0.10.3-19-g…`). Knobs: `ENV_NAME`, `GENJAX_REPO`, `GENJAX_COMMIT`, `GENJAX_SRC`, `CONDA_BASE`.

> **genjax is the one fragile dep.** It's an editable install from `genjax-community/genjax`. If that
> repo is private on the cluster, clone it yourself (or `scp` your local checkout) and point
> `GENJAX_SRC` at it. The verification block at the end of `setup_env.sh` should print `backend: gpu`
> on a GPU node.

Sanity-check the model end-to-end after setup:
```bash
NC_LM=EleutherAI/pythia-70m PYTHONPATH=src python -m genjax_port.pythia_word_caprop --selftest
```

---

## 2. The input file

One observed (noisy) sentence per line. Blank lines and `#` comments are ignored. A sentence's index
is its position among the kept lines, so **append new sentences at the end** — inserting in the middle
shifts indices and recomputes the shifted tail (changed lines are auto-detected and recomputed; see
Resume). See `sentences.example.txt`.

---

## 3. Submit a batch

```bash
INPUT=data/my_sentences.txt \
CHANNEL=align REJUV=gibbs+bd PARTICLES=128 REJUV_LOOKBACK=6 \
SENTENCES_PER_SHARD=8 MAX_PARALLEL=20 MEM=12G \
bash slurm/submit_nc_batch.sh
```

This is resume-aware: it writes a manifest, figures out which shards still have work, and submits a job
array over only those — re-running the same command continues where it left off. Use `DRYRUN=1` to print
the generated `sbatch` script and the plan **without submitting** (no SLURM needed — great for checking a
config first).

### Cluster settings (in `slurm/cluster.env`)

The execution logic is cluster-agnostic; the environment block is what you must get right:

| var | default | meaning |
|---|---|---|
| `PARTITIONS` | *(required)* | comma-separated GPU partition(s); SLURM picks the first free. No default — set it. |
| `GRES` | `gpu:1` | pythia-70m is tiny — any single GPU. Pin a type only if required (`gpu:a100:1`) |
| `CONDA_ENV` | `ncgenjax` | the **cluster-side** env name from step 1 |
| `CONDA_BASE` | *(auto)* | set to e.g. `$HOME/miniforge3` if `conda info --base` isn't found in the job |
| `MODULE_LOADS` | *(empty)* | e.g. `module load miniforge`; semicolon-separate multiples |

If a job fails immediately, it's almost always this block — check `logs/shard_*_*.err`.

### Common knobs

**Model** (each distinct value → its own output directory): `NC_LM`, `CHANNEL` (`align`/`word_action`/
`char_copy`), `REJUV` (`off`/`gibbs`/`gibbs+bd`), `PARTICLES`, `BAND`, `MAX_DIST`, `REJUV_LOOKBACK`,
`SEED`, `LM_TEMP`, `INS_RATE`, and optional overrides `WDEL`, `ALIGN_SLOPE`, `ACTION_ALPHA`, `WINS`,
`UNIFORM_INS`, `BD_P_STAY`, `BD_MODE`, `BD_ATTEMPTS`, `NO_BD_FUNCWORDS`.

**Execution**: `RESULTS_ROOT` (default `results_nc`), `SENTENCES_PER_SHARD` (8, the **max** per shard),
`MIN_SENTENCES_PER_SHARD` (4), `SORT_BY_LENGTH` (1), `MAX_PARALLEL` (20), `MEM` (12G), `MAX_TIME`
(3:59:00), `WRITE_VIZ` (1), `OVERWRITE` (0), `SKIP_ERRORS` (0), `SECONDS_PER_ITEM` (240, only used to
auto-size `--time`).

**Placement**: `USE_GPU` (1). pythia-70m barely uses the GPU — a particle filter of a 70M model is a
long sequence of tiny ops, so GPU ≈ CPU per item. Set `USE_GPU=0` to drop `--gres`, target
`CPU_PARTITIONS` (falls back to `PARTITIONS`), and force `JAX_PLATFORMS=cpu`; CPU partitions often
queue faster and are lighter on priority, so a batch can finish *sooner* on CPU. Bump `CPUS` for CPU.

**Length-bucketed sharding** (`SORT_BY_LENGTH=1`): the compiled SMC kernel is keyed by sentence shape
(word count, mainly), and the in-process JIT cache reuses it only for same-shape sentences — so each
shard is filled with same-length sentences and pays the JAX trace/lower compile ~once instead of
per-distinct-length. `SENTENCES_PER_SHARD`/`MIN_SENTENCES_PER_SHARD` bound shard size (max is a hard
cap to protect `--time`; min is best-effort). This only changes shard *membership* — outputs stay keyed
by original line index, so resume is unaffected and you can change these knobs between runs.
(Note: the JAX *persistent on-disk* compilation cache does **not** help here — the cost is
tracing/lowering, not XLA backend compile — so the in-process reuse from bucketing is the lever.)

---

## 4. How the requirements are met

- **Minimal memory.** `MEM` is a flat host-RAM request (GPU memory is separate; `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  keeps jax from grabbing the whole GPU). **Measure and tighten it** — over-requesting hurts your priority:
  ```bash
  seff <jobid>        # look at "Memory Efficiency"; set MEM ≈ 1.2× the peak
  ```
  12G is a deliberately safe starting point so the first run doesn't OOM-kill; drop it once you've measured.
- **Preemptions / node failures.** `--requeue` is set; writes are atomic (`.tmp` → `os.replace`, viz
  written before the compact file), so a preempted task loses at most the in-flight sentence and
  redoes only that on requeue.
- **Recoverable per-sentence failures.** Each sentence is wrapped in try/except: a crash writes a
  `status:"error"` record (with traceback) and the shard **continues**. Error items are **retried** on
  the next submit (they don't count as done) — add `SKIP_ERRORS=1` to stop retrying a stubborn one.
- **No needless re-runs.** An item is "done" iff its compact JSON exists, its stored `observed` matches
  the current line, and `status=="ok"`. Re-submitting skips done work; editing a line recomputes only
  that line. `OVERWRITE=1` forces a full redo.
- **Parallel-jobs quota.** The array is throttled with `…%MAX_PARALLEL`, so no more than `MAX_PARALLEL`
  tasks run at once regardless of how many shards there are. (Running several configs at once submits
  several arrays — set `MAX_PARALLEL` so their sum stays under your quota.)
- **Logs.** `<config dir>/logs/shard_<task>_<jobid>.{out,err}` plus the generated `submit.sbatch`, kept
  next to the results they produced.
- **Multiple configs → separate dirs.** The output directory encodes the config (see below), so
  re-running with different knobs never collides.

---

## 5. Output layout

```
results_nc/
  my_sentences/                                              # <input stem>
    lm-pythia-70m__ch-align__rej-gibbsbd__P128__b2__d2__lb6__s0/   # <config slug>
      manifest.json            # full config + git sha + input file + shard layout
      results/
        item_00000.json        # observed, top-k inferred + probs, logZ, runtime, config, git sha, SLURM ids
        item_00000.viz.json    # directly viz-loadable trace (python -m genjax_port.viz item_00000.viz.json)
        item_00001.json
        ...
      logs/
        submit.sbatch
        shard_0_12345.out / .err
```

Core knobs are always in the slug; overridden optionals are appended (e.g. `__K-5.0__lt0.5`), so a
vanilla run stays short and any varied knob makes a distinct directory. Every `item_*.json` is also
fully self-describing (it embeds its `config`), so analysis never needs to parse the directory name:

```python
import glob, json, pandas as pd
rows = [json.load(open(f)) for f in glob.glob("results_nc/**/results/item_*.json", recursive=True)
        if ".viz." not in f]
df = pd.json_normalize(rows)          # columns: observed, map, logZ, runtime_s, config.channel, config.particles, ...
```

Set `WRITE_VIZ=0` to skip the heavy `*.viz.json` (the compact `item_*.json` is always written).

---

## 6. Debugging locally before you submit

The runner works on the mac too — run a single shard with no SLURM:

```bash
# one shard (sentences 0–1) of the example, locally, no GPU needed (jax falls back to CPU):
PYTHONPATH=src python slurm/run_nc_batch.py \
  --input slurm/sentences.example.txt --results-root /tmp/nc_local \
  --shard-size 2 --shard-index 0 --particles 32

# preview where a config writes / what's left to do (instant, no model load):
python slurm/run_nc_batch.py --input slurm/sentences.example.txt --plan --shard-size 8
```

---

## 7. Sweeps (several configs)

Re-run the submit script with different env vars — each lands in its own directory:

```bash
for K in -3.5 -4.5 -5.5; do
  INPUT=data/my_sentences.txt ALIGN_SLOPE=$K MAX_PARALLEL=10 bash slurm/submit_nc_batch.sh
done
for S in 0 1 2; do
  INPUT=data/my_sentences.txt SEED=$S MAX_PARALLEL=10 bash slurm/submit_nc_batch.sh
done
```

Keep `MAX_PARALLEL × (configs running at once)` under your cluster's per-user array/job quota.
