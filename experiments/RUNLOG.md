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

### 2026-08-31T15:06:45Z — smoke × main_bd
- commit: `80a4722` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-gibbsbd__P64__b2__d2__lb6__s0__nseed4`  remaining before submit: 8
- env: `INPUT=experiments/stimuli/smoke.input.jsonl BAND=2 CHANNEL=align MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=gibbs+bd REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=32G SECONDS_PER_ITEM=1320 SENTENCES_PER_SHARD=2`
- job id: 21653887
- outcome: **FAILED at startup**, all 4 shards (SLURM 21653887_0..3, <1 min each on node2802): `ModuleNotFoundError: No module named 'regex'` at `src/genjax_port/unigram.py: import wordfreq` (wordfreq is a `~/.local` user-site package; its dependency `regex` was installed NOWHERE — the ncgenjax env had been borrowing it from `~/.local`, whose contents changed 2026-07-29). No results written; dirs stayed empty. Fix + resubmit recorded in the entries below.

### 2026-08-31T15:06:47Z — smoke × main_off
- commit: `80a4722` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__nseed4`  remaining before submit: 8
- env: `INPUT=experiments/stimuli/smoke.input.jsonl BAND=2 CHANNEL=align MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=16G SECONDS_PER_ITEM=120 SENTENCES_PER_SHARD=8`
- job id: 21653888
- outcome: **FAILED at startup**, the 1 shard (SLURM 21653888_0, <1 min on node2802): `ModuleNotFoundError: No module named 'regex'` at `src/genjax_port/unigram.py: import wordfreq` (wordfreq is a `~/.local` user-site package; its dependency `regex` was installed NOWHERE — the ncgenjax env had been borrowing it from `~/.local`, whose contents changed 2026-07-29). No results written; dirs stayed empty. Fix + resubmit recorded in the entries below.

### 2026-08-31T15:10:30Z — cluster env fix (not a launch)

- Installed `regex==2026.8.31` into the cluster's `ncgenjax` env (`~/.mamba/envs/ncgenjax/lib/python3.12/site-packages/regex/`,
  plain `python -m pip install regex` with the env active; NOT `--user`). Nothing else changed.
- Root cause: `slurm/setup_env.sh` pip-installs the pinned deps, but pip treated packages already present in the
  user-site (`~/.local/lib/python3.12/site-packages`, populated 2025-12-04) as satisfied, so the env never got its
  own `numpy`, `wordfreq` (nor `regex`, `click`, `python-dateutil`, ... — see `pip check`). The env still loads
  `numpy 1.26.4` and `wordfreq 3.1.1` from `~/.local` (same versions as the pins and as the June runs). Something
  in `~/.local` changed 2026-07-29 (dir mtime; no entry is newer, so an entry was removed) — that is when `regex`
  disappeared. `regex` only affects wordfreq's tokenizer, not frequency values for ordinary words.
- Consequence for later: do NOT set `PYTHONNOUSERSITE=1` in the sbatch (the env depends on `~/.local` for numpy
  and wordfreq); a future rebuild of the env should set it (or `pip install --ignore-installed`) so the pins land
  in the env. Login-node import tests of the jax chain are not possible (64-process limit kills XLA's thread pool).

### 2026-08-31T15:11:20Z — smoke × main_bd
- commit: `80a4722` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-gibbsbd__P64__b2__d2__lb6__s0__nseed4`  remaining before submit: 8
- env: `INPUT=experiments/stimuli/smoke.input.jsonl BAND=2 CHANNEL=align MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=gibbs+bd REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=32G SECONDS_PER_ITEM=1320 SENTENCES_PER_SHARD=2`
- job id: 21654049
- outcome: (append when finished)

### 2026-08-31T15:11:21Z — smoke × main_off
- commit: `80a4722` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__nseed4`  remaining before submit: 8
- env: `INPUT=experiments/stimuli/smoke.input.jsonl BAND=2 CHANNEL=align MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=16G SECONDS_PER_ITEM=120 SENTENCES_PER_SHARD=8`
- job id: 21654051
- outcome: **COMPLETED** (SLURM 21654051_0, elapsed 24:03, MaxRSS 12.97 GB of 16G, exit 0; node2802). 8/8 items ok (4 seeds each, merged), pulled + collected: 430/430 finite unit surprisals, `sum(S)+S_end == −logZ` exact. FINDING: 5/8 merged records carry ~2 deleted leading units (`del_before` at unit 0 = 1.99/0.32/1.72/0/0/1.94/2.0/0) that are `\n` tokens (ids 187,187 + no-space `The` 510) hidden by the decoder's `.strip()` — the plain-literal parse is ABSENT from the cloud although the model's joint score prefers it by ~8 nats (LM −0.74, channel −7.9 for the newline path). Inference defect, under investigation (dedup A/B) BEFORE any Phase-5 fan-out. Off-arm MEM note: 12.97 GB on a 13-word item; datasets go to 19 words -> raise main_off MEM to 24G.

### 2026-08-31T17:05:00Z — smoke findings: Phase-5 fan-out ON HOLD (not a launch)

- The off-arm smoke surfaced an inference failure of the deployed operating point (align,
  α=(200,2,2), band=2, rejuv=off): the forward filter loses the plain-literal parse (weight
  exactly 0 at P=16/64/256, three keys) to a cloud carrying two deleted leading "\n" units that
  the decoder strips from every reported string — although the model's own joint score prefers
  the plain literal by ~8 nats (LM alone disfavors the junk by 0.74). logZ is ~10 nats below the
  band=1 run of the SAME channel (impossible for exact inference; band 2 ⊃ band 1). Mechanism
  (deletion-priced loitering in the intended-word-synchronous intermediate targets) and direct
  tests: `planning/LEADING_DELETION_FINDINGS.md` + `planning/leading_del_probe/`.
- The gibbs+bd arm repairs it on smoke item 0 (del_before(The) 2.0→0.0, logZ −62.4→−49.5,
  indel move n_chosen_del 3–7/seed). 5/8 smoke items are affected in the off arm.
- Phase 5 (both arms, all datasets) is NOT submitted; whether to run as-is, fix the
  intermediate target, change band/α_del, or run bd-only is the user's decision.

### 2026-08-31T18:20:00Z — decision + handoff (not a launch)

- User chose to FIX the resampling weights before Phase 5 (option 2 of
  `planning/LEADING_DELETION_FINDINGS.md`): a lookahead charge at resampling events, to be
  implemented in a fresh session per `planning/LOOKAHEAD_CHARGE_PLAN.md` (design, exact math,
  touch-points, gates, and the cluster state at handoff are all in that plan).
- At handoff, smoke × main_bd job 21654049 shards 1 and 3 were still running; the next session
  records their outcome on the 15:11:20Z entry above.
