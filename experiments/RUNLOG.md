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

### 2026-08-31T03:51:21Z — cost probe (local Mac, pre-submit sizing; harness plan §5)

- commit: `c94fcf1` (+ uncommitted Phase-4 files: configs/, run.sh, bd_mem_probe context arg)
- `planning/bd_mem_probe.py` over all 9 `stimuli/probe.input.jsonl` items, P=64, one process
  per run, arms off and gibbs+bd (bd_mode=gibbs, lb6, band2, dedup on, funcwords on).
- **off**: runtime 22.0–40.2 s/item, peak RSS 6.2–7.3 GB.
- **gibbs+bd**: runtime 134–440 s/item (worst = item 4, Wmax=21, Kc=31, 651 grid forwards),
  peak RSS 8.8–15.6 GB (worst = item 3, 15.55 GB). Context items (LCTX up to ~68) are NOT the
  worst case — long sentences are.
- **Sizing (MEM ≈ 2× Mac RSS, SECONDS_PER_ITEM ≈ 3× Mac runtime; per item×seed):**
  - `main_off`:  MEM=16G  SECONDS_PER_ITEM=120   SENTENCES_PER_SHARD=8  (EST ≈ 79 min/shard @ 4 seeds)
  - `main_bd`:   MEM=32G  SECONDS_PER_ITEM=1320  SENTENCES_PER_SHARD=2  (EST ≈ 3:11/shard @ 4 seeds — fits MAX_TIME 3:59)
- full RESULT lines:
  - RESULT sentence='The child with a learning disability benefited from the extra time.' P=64 ctx_words=0 Kc=-1 Wmax=-1 LCTX~1 grid_forwards=1 runtime_s=22.0 peak_rss_GB=6.31
  - RESULT sentence='The child with a learning disability benefited from the extra time.' P=64 ctx_words=0 Kc=24 Wmax=15 LCTX~33 grid_forwards=360 runtime_s=134.4 peak_rss_GB=8.79
  - RESULT sentence='The scuba instructor rented the equipment to the tourist.' P=64 ctx_words=21 Kc=-1 Wmax=-1 LCTX~1 grid_forwards=1 runtime_s=24.3 peak_rss_GB=6.31
  - RESULT sentence='The scuba instructor rented the equipment to the tourist.' P=64 ctx_words=21 Kc=23 Wmax=13 LCTX~52 grid_forwards=299 runtime_s=156.0 peak_rss_GB=9.83
  - RESULT sentence='The teacher taught that subtraction was the opposite of additions.' P=64 ctx_words=0 Kc=-1 Wmax=-1 LCTX~1 grid_forwards=1 runtime_s=23.9 peak_rss_GB=6.17
  - RESULT sentence='The teacher taught that subtraction was the opposite of additions.' P=64 ctx_words=0 Kc=24 Wmax=14 LCTX~45 grid_forwards=336 runtime_s=167.5 peak_rss_GB=9.43
  - RESULT sentence='The tunnel under the lake has cost a lot of money beyond enigeering challenges.' P=64 ctx_words=0 Kc=-1 Wmax=-1 LCTX~1 grid_forwards=1 runtime_s=30.6 peak_rss_GB=7.15
  - RESULT sentence='The tunnel under the lake has cost a lot of money beyond enigeering challenges.' P=64 ctx_words=0 Kc=28 Wmax=18 LCTX~57 grid_forwards=504 runtime_s=314.8 peak_rss_GB=15.55
  - RESULT sentence='The player who was paid the bonus remained essentially the same despite his sudden fame and wealth.' P=64 ctx_words=0 Kc=-1 Wmax=-1 LCTX~1 grid_forwards=1 runtime_s=40.2 peak_rss_GB=7.26
  - RESULT sentence='The player who was paid the bonus remained essentially the same despite his sudden fame and wealth.' P=64 ctx_words=0 Kc=31 Wmax=21 LCTX~66 grid_forwards=651 runtime_s=440.2 peak_rss_GB=10.71
  - RESULT sentence='She got a lovely tan while spending some time in the smoothie at the juice bar.' P=64 ctx_words=0 Kc=-1 Wmax=-1 LCTX~1 grid_forwards=1 runtime_s=27.5 peak_rss_GB=7.16
  - RESULT sentence='She got a lovely tan while spending some time in the smoothie at the juice bar.' P=64 ctx_words=0 Kc=28 Wmax=20 LCTX~43 grid_forwards=560 runtime_s=280.7 peak_rss_GB=11.83
  - RESULT sentence='At the dinner party, I met a man who was allowed the pleasure of eating sweets by his doctor.' P=64 ctx_words=0 Kc=-1 Wmax=-1 LCTX~1 grid_forwards=1 runtime_s=36.0 peak_rss_GB=6.88
  - RESULT sentence='At the dinner party, I met a man who was allowed the pleasure of eating sweets by his doctor.' P=64 ctx_words=0 Kc=31 Wmax=24 LCTX~51 grid_forwards=744 runtime_s=397.2 peak_rss_GB=9.86
  - RESULT sentence='How many animals of each kind did Moses take on the ark?' P=64 ctx_words=0 Kc=-1 Wmax=-1 LCTX~1 grid_forwards=1 runtime_s=27.4 peak_rss_GB=6.16
  - RESULT sentence='How many animals of each kind did Moses take on the ark?' P=64 ctx_words=0 Kc=26 Wmax=16 LCTX~51 grid_forwards=416 runtime_s=263.2 peak_rss_GB=14.88
  - RESULT sentence='The letter was written by the husband.' P=64 ctx_words=31 Kc=-1 Wmax=-1 LCTX~1 grid_forwards=1 runtime_s=24.2 peak_rss_GB=6.71
  - RESULT sentence='The letter was written by the husband.' P=64 ctx_words=31 Kc=22 Wmax=11 LCTX~68 grid_forwards=242 runtime_s=168.4 peak_rss_GB=9.12
