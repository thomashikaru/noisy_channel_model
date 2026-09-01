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
- outcome: shards 0–2 **COMPLETED** (elapsed 44:10 / 58:58 / 30:20, MaxRSS 12.67 / 11.65 / 10.71 GB,
  exit 0); shard 3 still RUNNING at 2:06:10 elapsed as of 2026-08-31T17:19Z (on pace for its
  ~3:11 estimate under the 3:59 cap). Pulled at 17:15Z: 7/8 merged items on disk (item 7 pending
  in shard 3). bd-arm reference numbers for the lookahead A/B: item 0 logZ −49.48 p_lit 0.57,
  item 1 −46.22 / 0.94, item 2 −89.72 / 0.00 (del_before[0] 0.83 — bd only PARTIALLY repairs the
  Medics item), item 5 −59.55 / 0.51, item 6 −75.64 / 0.94; unit-0 del_before 0.0 everywhere but
  item 2.

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

### 2026-08-31T17:25:00Z — lookahead charge implemented + local gates (not a launch)

- `planning/LOOKAHEAD_CHARGE_PLAN.md` EXECUTED (P1–P3): `pairhmm_smc.run(lookahead_lp=)` APF
  twist at the resample site, `pythia_word_caprop.run(lookahead=/lookahead_lp=)`, worker
  `--lookahead` (slug suffix `la`), `LOOKAHEAD` env in submit_nc_batch.sh + run.sh CFG_VARS.
  Default OFF everywhere; `configs/main.env` NOT changed (user's call after these gates).
  One design correction (done-particle psi=0 guard, caprop-only) — see the plan's
  "Execution results" section for all gate numbers.
- Gates: off-path bit-identity PASS (toy 9-case + Pythia probe item byte-identical); toy
  exactness PASS (4 new gates; suite 117 passed); probe flip PASS (del_before(The) 2.00→0.00 on
  3/3 keys, logZ −63.6→≈−52.5; the candle posterior is genuinely BIMODAL — the to-repair ties
  the literal, LM gain +4.62 vs deletion price −4.62); §3.1 identity on disk EXACT (8/8
  lookahead-ON records, local P16 run).
- Smoke A/B at the main operating point (local, off arm P=64 ×4 seeds, `la` slug vs the pulled
  cluster off arm): leading-deletion artifact ELIMINATED on 4/5 affected items (unit-0
  del_before →0.00; logZ +9.0/+8.0/+1.1/+5.3 on items 0/1/5/6; item 1's MAP repaired). Item 2
  (Medics) NOT fixed — a DIFFERENT failure: with no context, the literal 'Medics' costs ~14 nats
  at step 1, so the one-step caprop proposal never instantiates the literal at P=64 (verified
  the joint prefers plain by ~17–18 nats over both junk MAPs); a resampling twist cannot
  resurrect an un-instantiated hypothesis, and gibbs+bd only partially repairs it (del_before
  0.83, p_lit 0.0). NEW COST at P=64: heavier-tailed seeds — on items 5/6 the la clouds
  mode-collapsed onto joint-inferior parses in all 4 seeds (item 5 seed spread 2.4→7.0;
  bd reference finds 10–13 more nats there and keeps the literal). Battery A/B = gate 7,
  user's call.

### 2026-08-31T18:55:00Z — smoke × main_bd COMPLETION (appends the 15:11:20Z entry; source: this session's sacct watch)

- shard 3 **COMPLETED**: elapsed 2:15:57, MaxRSS **14.35 GB**, exit 0 → all 4 shards done. bd-arm
  memory peak on the smoke = 14.35 GB (the 32G budget is generous; 24G would also have held here).
- item 7 (the 22-word context prime): merged 4/4 seeds → 'The mother gave the candle to the
  daughter.' — p_literal 0.15, logZ −24.90, seed spread 0.6. The context drives a confident
  "to" restoration.
- item 4: logZ −97.55, p_literal 0.0, runtime 3673 s for 4 seeds (~918 s per item×seed — the
  SECONDS_PER_ITEM=1320 sizing holds with margin).
- Smoke set now 8/8 merged on BOTH arms; bd words table 430/430 finite surprisal_nc after pull.

### 2026-08-31T18:13:46Z — battery A/B for the lookahead charge (gate 7), off arm × {la off, la on}
- commit: `784b476` (local == cluster; pushed + ff-pulled this session)
- input: `planning/calibration_battery_v0.txt` (87 items), P=64, N_SEEDS=2, band 2, lb 6, align, rejuv=off
- sizing: MEM=24G SECONDS_PER_ITEM=120 SENTENCES_PER_SHARD=8 (13 shards/arm, --time 47:00)
- baseline slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__nseed2` — job id 21666107
- lookahead slug: `...__s0__la__nseed2` (LOOKAHEAD=1) — job id 21666110
- purpose: LOOKAHEAD_CHARGE_PLAN gate 7 — "did anything else move"; decides LOOKAHEAD in main.env.
- outcome: **both COMPLETED** (all 26 shards exit 0; per-shard 3:03–9:29, MaxRSS ≤ 9.0 GB — 24G
  generous; wall ≈ 25 min from submit to last shard). 87/87 merged on both arms; pulled; diff =
  `planning/la_vs_off_diff.py` → `planning/calibration_la_vs_off.csv`. HEADLINE: the
  leading-deletion artifact was on **46/87** baseline items (far beyond the smoke's 5/8) and the
  lookahead run clears it on 39 → **7 remain** (Medics-class proposal-support cases + partial
  clears). logZ (la − off): mean +4.05, median +1.49, up 51 / down 14 / flat 22.
  Matches-expected 42→44/87 exact, 47→49 case-insensitive; edited-rate 32→28 (junk MAPs like
  '# They…', '- The clerk…', 'For the very tall man left.', spurious caps → cleaned). Cost:
  MAP changed on 35 items = 11 newly-correct vs 9 newly-wrong, several of the wrong ones the
  P=64 heavy-tail collapse ('The chef seasoned the author.', 'The Bakerite the children the
  cake.'); 2-seed logZ spread 2.72→2.99 (la>off on 43/87). Stochastic eval at HALF the
  experiment's seed count (2 vs main.env's 4) — the merge is stronger in the real config.

### 2026-08-31T18:46:06Z — smoke × main_off
- commit: `e70b8d1` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 8
- env: `INPUT=experiments/stimuli/smoke.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=24G SECONDS_PER_ITEM=120 SENTENCES_PER_SHARD=8`
- job id: 21668423
- outcome: **COMPLETED** (21668423_0, 19:02, MaxRSS 12.58 GB of 24G, exit 0). 8/8 ok; matches
  the local gate-6 A/B to two decimals (x86 vs arm agreement); del_before[0] = 0 everywhere
  except item 2 (Medics, 1.99 — the known proposal-support case).

### 2026-08-31T18:46:22Z — smoke × main_bd
- commit: `e70b8d1` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-gibbsbd__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 8
- env: `INPUT=experiments/stimuli/smoke.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=gibbs+bd REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=32G SECONDS_PER_ITEM=1320 SENTENCES_PER_SHARD=2`
- job id: 21668496
- outcome: **COMPLETED** (21668496_0..3: 27:24/1:24:54/23:11/1:44:43, MaxRSS ≤ 14.1 GB of 32G,
  exit 0 — every shard FASTER than its pre-fix counterpart, worst 2:16→1:45). 8/8 ok, all
  surprisals finite, sum(S)+S_end == −logZ exact on merged records. del_before[0] = 0 on all
  items incl. Medics (0.83 → 0.01 — the fix completes what rejuvenation only partially
  repaired). logZ shifts −4..+4 vs the pre-fix bd smoke (seed noise + changed sampler);
  p_literal stable. off and bd configurations now AGREE on evidence for item 0 (−53.4 vs −53.7,
  previously −62.4 vs −49.5).

### 2026-08-31T20:38:14Z — moses × main_off
- commit: `f856895` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 1
- env: `INPUT=experiments/stimuli/moses.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=24G SECONDS_PER_ITEM=120`
- job id: 21676493
- outcome: (append when finished)

### 2026-08-31T20:38:16Z — tabor2004 × main_off
- commit: `f856895` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 128
- env: `INPUT=experiments/stimuli/tabor2004.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=24G SECONDS_PER_ITEM=120`
- job id: 21676496
- outcome: (append when finished)

### 2026-08-31T20:38:18Z — huang2024 × main_off
- commit: `f856895` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 144
- env: `INPUT=experiments/stimuli/huang2024.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=24G SECONDS_PER_ITEM=120`
- job id: 21676499
- outcome: (append when finished)

### 2026-08-31T20:38:19Z — gibson2013 × main_off
- commit: `f856895` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 240
- env: `INPUT=experiments/stimuli/gibson2013.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=24G SECONDS_PER_ITEM=120`
- job id: 21676500
- outcome: (append when finished)

### 2026-08-31T20:38:21Z — clark2026 × main_off
- commit: `f856895` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 360
- env: `INPUT=experiments/stimuli/clark2026.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=24G SECONDS_PER_ITEM=120`
- job id: 21676502
- outcome: (append when finished)

### 2026-08-31T20:38:22Z — qian2023 × main_off
- commit: `f856895` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 472
- env: `INPUT=experiments/stimuli/qian2023.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=24G SECONDS_PER_ITEM=120`
- job id: 21676505
- outcome: (append when finished)

### 2026-08-31T20:38:24Z — ryskin2021 × main_off
- commit: `f856895` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 504
- env: `INPUT=experiments/stimuli/ryskin2021.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=24G SECONDS_PER_ITEM=120`
- job id: 21676506
- outcome: (append when finished)

### 2026-08-31T21:09:46Z — chen2023 × main_off
- commit: `2b59960` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 480
- env: `INPUT=experiments/stimuli/chen2023.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=24G SECONDS_PER_ITEM=120`
- job id: 21678760
- outcome: (append when finished)

### 2026-08-31T21:09:49Z — moses × main_bd
- commit: `2b59960` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-gibbsbd__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 1
- env: `INPUT=experiments/stimuli/moses.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=gibbs+bd REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=32G SECONDS_PER_ITEM=1320 SENTENCES_PER_SHARD=2`
- job id: 21678765
- outcome: **COMPLETED** (21678765_0, 1:15:23, exit 0). 1/1 merged. The single-item bd/off
  runtime ratio here is 75.4 min vs 3.8 min = 20x; the 8-item smoke A/B gives 240.2 min vs
  19.0 min = 12.6x. Use 12.6x-20x when sizing the remaining bd datasets.

### 2026-09-01T14:18:02Z — huang2024 × main_off
- commit: `dc84b5a` (local == cluster)
- slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`  remaining before submit: 1
- env: `INPUT=experiments/stimuli/huang2024.input.jsonl BAND=2 CHANNEL=align LOOKAHEAD=1 MAX_DIST=2 N_SEEDS=4 PARTICLES=64 REJUV=off REJUV_LOOKBACK=6 SEED=0 SORT_BY_LENGTH=1 TOP=20 WRITE_VIZ=0  MEM=24G SECONDS_PER_ITEM=1800 SENTENCES_PER_SHARD=1`
- job id: 21746380
- outcome: **COMPLETED** (21746380_143, 1:37, exit 0) -> huang2024 144/144. Repairs the ONE
  Phase-5 casualty: 21676499_20 hit TIMEOUT at 1:19:24, exactly its auto-sized --time of
  900 + 8*4*120 = 4740 s, after finishing 7 of 8 items plus 3 of 4 seeds on sentence_id 136
  ("After the contestant lost, the money became unavailable ...", 17 words, the longest in
  huang2024). At 1:37 for the re-run the item is NOT intrinsically slow -- the shard simply ran
  out of budget carrying 8 long items x 4 seeds. Lesson for sizing: SECONDS_PER_ITEM=120 is too
  thin for datasets whose p90 length is 15+ words; huang2024 and tabor2004 are the two.

### 2026-09-01T14:30:00Z — PHASE-5 `main_off` COMPLETE (8/8 datasets) + verification

- commit: `dc84b5a`; config `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`
- **2337/2337 items ok, 0 error, 0 missing** across moses/tabor2004/huang2024/gibson2013/
  clark2026/qian2023/ryskin2021/chen2023. All pulled into `results_nc/` and collected into
  `experiments/outputs/`; `status.md` regenerated over all 8 datasets in ONE collect call
  (a per-dataset collect rewrites status.md with only that dataset -- always pass them together).
- cost actually spent: 322 array tasks, 114.5 CPU-hours (chen 17.1 / clark 18.3 / gibson 7.9 /
  huang 11.0 / moses 0.1 / qian 25.6 / ryskin 26.4 / tabor 8.1).
- verification (plan section 8, gates 1-5):
  - 154,695 word rows, **zero non-finite** surprisal_nc or surprisal_lm.
  - p_copy + p_sub + p_ins == 1 on every row (max deviation < 1e-4).
  - identity sum(S_k) + S_end == -logZ holds on all 2337 merged records, **worst 2.84e-14**.
  - gibson2013 grammaticality contrast is in the right direction: ungrammatical items edited
    51% of the time vs grammatical 25%.
- MAP behaviour of the off arm across the 7 real datasets (2335 items): 52% keep the literal
  MAP, 48% edited (sub 606 / multi 286 / ins 226 / del 8); **731 items (31%) have p_literal
  exactly 0**. Per dataset the edited rate runs ryskin2021 72% / clark2026 56% / huang2024 52% /
  gibson2013 41% / tabor2004 42% / qian2023 38% / chen2023 32%. Median 4-seed logZ spread 7.11.
- **spot-checks (plan gate 5) -- the off arm misses 2 of 3.** Comparing the smoke set off vs bd,
  both at LOOKAHEAD=1:
  - candle/daughter -> "to": PASSES on both arms, but only on the context-primed item
    (p_map 0.603 off, 0.809 bd). The uncontexted copy stays literal on both.
  - inflection -> infection: bd gives the clean 'Medics cleaned and bandaged the wound to
    prevent an infection.' (p_map 0.994, edit=sub); **off corrupts the start** to 'This article
    medic cleaned and bandaged ...' -- the known step-1 proposal-support failure.
  - licked -> kicked: **FAILS ON BOTH ARMS.** off emits junk ('The boy looked licked from the
    big round ball into the net.'); bd stays literal. This gate is not met by either
    configuration and is a real open item, not a sizing problem.
  - more broadly on the smoke set the off arm leaves 4 of 8 items at p_literal 0 with junk
    multi-edit MAPs, where bd holds 5 of 8 literal. The rejuvenation arm is visibly the
    better-behaved one.
- **user decision (2026-09-01): HOLD OFF on the `main_bd` arm** for the remaining 7 datasets;
  decide after inspecting the off results. When it does run, the agreed sizing is
  SENTENCES_PER_SHARD=2 everywhere (--time 3:11:00, MEM=32G), dataset by dataset -- 1164 tasks
  total exceeds the account's MaxSubmit=500, so it CANNOT go in as one batch. Estimated cost
  ~1440 CPU-hours at 12.6x, up to ~2300 at 20x.
- housekeeping: two stray editor backup files sit untracked on the cluster checkout,
  `slurm/#cluster.env#` and `slurm/cluster.env~`. Harmless, but they are the only thing making
  the cluster tree dirty.


### 2026-09-01T17:43:49Z — battery A/B for the lookahead-IN-PROPOSAL fix (OFF_ARM_INFERENCE_FIX.md §6 decision 1), off arm × {la, la+lap}
- commit: `4e61c50` (local == cluster; pushed + ff-pulled this session). The fix = `lookahead_proposal`
  (`--lookahead-proposal` / `LA_PROPOSAL=1`, slug part `lap`), default OFF; 4 new exact gates, suite 121 passed.
- input: `planning/calibration_battery_v0.txt` (87 items), P=64, **N_SEEDS=4** (the real main.env count this
  time), band 2, lb 6, align, rejuv=off, LOOKAHEAD=1 on BOTH arms — identical except LA_PROPOSAL.
- sizing: MEM=24G SENTENCES_PER_SHARD=8 (13 shards/arm, auto --time 2:23:00), MAX_PARALLEL=20.
- baseline slug: `lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4` — job id 21752588
- fix slug:      `...__s0__la__lap__nseed4` (LA_PROPOSAL=1) — job id 21752596
- purpose: the pre-committed criteria (a) unit-0 del_before>0.5 artifact items cleared, (b) genuine repairs
  (expected==edit) retained, (c) edit rate not worse, (d) logZ up on the same model. Report:
  `planning/lap_vs_la_diff.py` → `planning/calibration_lap_vs_la.csv`. Decides LA_PROPOSAL in main.env and
  whether Phase-5 main_off is re-run (§6 decision 4).
- local smoke before submit: Medics p(target) 0.000/0.213/0.000 → 1.000/0.984/0.968 (keys 0-2), logZ −91.7 → −79.7;
  candle item identical between arms.
- run: all 26 shards were RUNNING within 1 min of submission; the queue was EMPTY ~27 min after submit (both
  arms finished). Results NOT yet pulled: the ORCD login nodes (orcd-login, 001–004) went unreachable
  (connection refused / handshake timeout) right after — cluster-side outage, no public notice; the SSH master
  dropped with it. Records are on the shared FS (atomic writes). Pull + diff pending the login nodes' return.

### 2026-09-01T17:58:10Z — battery A/B OUTCOME (la vs la+lap), jobs 21752588 / 21752596
- **both COMPLETED**: 26/26 shards COMPLETED, per-shard 6:15–26:28, MaxRSS 6.1–9.9 GB (24G generous; wall ≈ 27 min
  submit → queue empty). 87/87 merged + 0 errors on both arms; pulled to `results_nc/calibrationbatteryv0/`;
  diff `planning/lap_vs_la_diff.py` → `planning/calibration_lap_vs_la.csv`. (Pull was delayed ~1.5 h by an
  unannounced ORCD login-node outage — all of orcd-login/001/002/003 refused or timed out, then flapped back.)
- **HEADLINE — all four pre-committed criteria pass (OFF_ARM_INFERENCE_FIX.md §6 decision 1):**
  (a) unit-0 del_before>0.5 artifact items la **4 → 0**; any-unit del_before>0.5 (the §3.4 signature) 13 → 9.
  (b) genuine repairs (expected==edit, n=43) **14/43 → 14/43** — retained, none lost.
  (c) edited MAPs **28 → 26**/87 — not worse.
  (d) logZ (lap − la): **mean +1.04**, median +0.14, up 31 / down 14 / ~flat 42; the biggest movers are +7 to
  +9.8 nats and every one is an artifact clear (SUBW-01a, DELFROM-01b, CTRL-04, DELFOR-01a, SUBW-04a).
- matches-expected **48 → 54**/87 exact (+6), 54 → 58 case-insensitive (+4). MAP changed on 17 items:
  **8 newly-correct vs 2 newly-wrong** (lost: SUBW-02a 'medic'→'media' substitution; LADDER-send-2 spurious
  'Clerk' capital + dropped 'to'). Gains are the junk-MAP class from the 08-31 entry cleaned up
  ('The Bakerite the children the cake.' → 'The baker iced…', 'The tailor seed…' → 'sewed', 'The chef seasoned
  the author.' → 'the soup.').
- **4-seed logZ spread 6.70 → 1.34** (lap spread larger on only 13/87): the seeds now agree — the P=64
  heavy-tail collapse flagged on 08-31 is largely gone. Same model, same particle count, same cost/item.
- open with the user (§6 decisions 2 and 4): LA_PROPOSAL=1 in main.env? re-run Phase-5 main_off (~115 CPU-h)?
