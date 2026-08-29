# Cluster run log

Append-only. One entry per launch (including re-runs of failed or out-of-memory items). Never edit or
delete an earlier entry; correct it by appending a new one that says what changed.

Each entry records:

- **date** (UTC), **commit** (the sha the cluster had checked out, and whether the tree was dirty)
- **dataset** and **config slug**, and the number of remaining items `DRYRUN=1` reported
- the full `sbatch` line / env, including `MEM`, `SECONDS_PER_ITEM`, `SENTENCES_PER_SHARD`, `CPUS`
- the SLURM **job id(s)**
- the **outcome**: items finished, failures, wall time, and anything that had to be re-run

Cost-probe numbers (`planning/bd_mem_probe.py` on `stimuli/probe.input.jsonl`) are recorded here *before*
the first submit of each config, since they are what `MEM` and `SECONDS_PER_ITEM` are sized from.

---

<!-- entries below, newest last -->
