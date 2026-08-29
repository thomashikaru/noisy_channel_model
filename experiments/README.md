# `experiments/` — the noisy-channel experiment harness

Runs the genjax noisy-channel model over the published stimulus sets in `data/<study>/`, on the MIT ORCD
SLURM cluster, and collects the results into analysis-friendly tables.

**Status: under construction.** The design, the per-phase work and the decisions taken with the user live in
[`../planning/NOISY_CHANNEL_HARNESS_IMPLEMENTATION_PLAN.md`](../planning/NOISY_CHANNEL_HARNESS_IMPLEMENTATION_PLAN.md);
the goals it serves live in
[`../planning/NOISY_CHANNEL_EXPERIMENT_HARNESS.md`](../planning/NOISY_CHANNEL_EXPERIMENT_HARNESS.md).
This file becomes the full reproduction guide in Phase 6; until then, read the plan.

## Layout

| path | what it is | phase |
|---|---|---|
| `converters/` | one module per study: raw `data/<study>/…` -> the common stimulus schema | 1 |
| `build_stimuli.py` | runs the converters -> `stimuli/`, checks invariants, writes `MANIFEST.json` | 1 |
| `stimuli/` | the harmonized stimuli (**tracked** — this is the reproducibility anchor) | 1 |
| `configs/` | named model configs, as env files for `slurm/submit_nc_batch.sh` | 4 |
| `run.sh` | `fetch-tabor \| build \| smoke-local \| probe \| submit \| status \| pull \| collect` | 4–5 |
| `collect.py` | `results_nc/**/item_*.json` -> `outputs/<config>/<dataset>/*.csv.gz` | 3 |
| `outputs/` | tidy results (**untracked**; the raw per-item JSON stays in `results_nc/`) | 5 |
| `RUNLOG.md` | append-only record of every cluster launch | 5 |

The compute layer is the existing `slurm/run_nc_batch.py` + `slurm/submit_nc_batch.sh`, which this harness
wraps rather than replaces.

## The blind protocol

The order of operations is: run the model on the stimuli **blind to the human data**, and only compare to
human behavior once the model results are in. Concretely, nothing under `experiments/` may read a
human-data file. `converters/common.py` enforces this: every source file is opened through `open_source()`,
which refuses the paths listed in `HOLDOUT_PATHS` (currently `data/clark2026/exp_data_merged.csv` and
`data/clark2026/lists/`) and records a sha256 of everything it does open into `stimuli/MANIFEST.json`.

`data/` itself is gitignored, so the stimuli that ship with this repo are the harmonized copies under
`stimuli/`, plus the source hashes in the manifest.
