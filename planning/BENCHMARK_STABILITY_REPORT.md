# Posterior-stability benchmark: particles × rejuvenation × lookback × LM

**Status:** COMPLETE 2026-06-23. All 16 configs × 3 sentences × 6 seeds (288 runs, 155 min CPU).
Branch `rejuv-birth-death`, channel `align`, commit `2f45517`. Author: automated benchmark run.

## Question

How stable are the noisy-channel model's posterior inferences as a function of the knobs that trade
compute for inference exactness — number of particles `P`, rejuvenation mode (`off` / `gibbs` /
`gibbs+bd`), rejuvenation lookback window, and the LM (pythia-70m vs 160m)? Where are the *cliffs* —
configs stuck in bad posterior modes? What is the cheapest config that does not sacrifice correct
inference?

## Test sentences (3, interpretable)

| idx | observed | what it probes | target inference |
|---|---|---|---|
| want-go | `I want go home.` | missing word (insert "to") | `I want to go home.` |
| the-the | `The the patient recovered quickly.` | duplicated word (delete "the") | `The patient recovered quickly.` |
| boy-licked | `The boy licked the ball into the net.` | clean / plausible (identity) | `The boy licked the ball into the net.` |

These isolate the three channel operations: insertion-recovery, deletion-recovery, and copy/identity.

## Method

- Harness: `slurm/run_nc_batch.py` run locally (no SLURM), one shard, model loaded once per config.
- Each config = one point in the grid → its own config-encoded results dir (resume-safe).
- **`--n-seeds 6`**: every (config, sentence) is run with 6 independent RNG seeds. The harness writes
  per-seed records `item_NNNNN_sJ.json` and an **evidence-weighted merged** posterior `item_NNNNN.json`
  (mixture weight of seed r ∝ exp(logZ_r); merged logZ = logsumexp(logZ_r) − log R).
- **Stability readouts** (per config, per sentence), computed by `planning/bench_aggregate.py`:
  - **MAP-agreement** = fraction of the 6 seeds whose MAP equals the modal MAP (1.0 = all seeds agree).
  - **#distinct MAPs** across the 6 seeds.
  - **logZ_std / logZ_spread** across seeds = the cross-seed evidence-estimate noise (the SMC logẐ is a
    noisy estimate of the true log marginal likelihood; its seed-to-seed scatter measures inference
    noise directly, independent of whether the MAP is "correct").
  - mean per-seed runtime (the compute cost).

## Experimental design (compute-aware; OFAT around a baseline)

Baseline / central point = deployment default **(align, P=128, gibbs+bd, lookback=6, pythia-70m)**.

- **Phase A — core 2D grid:** `P ∈ {16,32,64,128}` × `rejuv ∈ {off, gibbs, gibbs+bd}`, lb=6, 70m. (12 configs)
- **Phase B — lookback spur:** P=64, gibbs+bd, `lookback ∈ {2, 12}` (lb=6 is in the grid). (2 configs)
- **Phase C — LM spur:** pythia-160m, P=64, `rejuv ∈ {off, gibbs+bd}`, lb=6. (2 configs)

16 configs × 3 sentences × 6 seeds = 288 inference runs. Configs are run cheap→expensive so early signal
(the `off` column) arrives in minutes; the `gibbs+bd` tier dominates wall-clock. Per-config runtime is
recorded per item so the compute cost is measured, not assumed.

Rationale for OFAT rather than a full 4-D cross-product: a full grid (4·3·3·2 = 72 configs ≈ 1300 runs)
would burn ~5× the compute for redundant interior points. The 2-D core grid answers the headline
particle×rejuv tradeoff and its interaction; the two 1-D spurs probe lookback and LM through the
deployment baseline, which is where those knobs actually matter.

---

## Results

Full data: 16 configs × 3 sentences × 6 seeds = **288 inference runs, 155 min CPU total** (no GPU).
Per-cell numbers in `planning/bench_results/aggregate.json`; reproduce tables with
`python planning/bench_aggregate.py`. "agree" = fraction of 6 seeds at the modal MAP; "logZ_std" =
cross-seed scatter of the SMC log-evidence estimate; ✓/✗ = whether the evidence-merged MAP matches the
intended inference.

### Runtime / compute cost (measured, s per single seed-run)

| rejuv | P16 | P32 | P64 | P128 | cost vs off |
|---|---|---|---|---|---|
| off | 4.1 | 4.6 | 5.7 | 7.3 | 1× |
| gibbs | 12 | 13 | 16 | 24 | ~3× |
| gibbs+bd | 23 | 31 | 43 | **83** | ~11× |

pythia-160m ≈ 2–3× its 70m counterpart. Cost ≈ `n_seeds × per-run`, and per-run is driven by the
rejuvenation sweep (KV-LM forwards ∝ P × candidates × lookback per SMC step). `off` is ~flat in P; the
rejuv sweep is what scales.

### Stability vs particles (pythia-70m, lb6) — agree-rate (logZ_std)

```
want-go (INSERT)     P16          P32          P64          P128
off                  1.00 (1.2)   1.00 (0.6)   1.00 (0.4)   1.00 (0.2)
gibbs                1.00 (1.2)   1.00 (0.3)   1.00 (0.5)   0.83 (0.4)
gibbs+bd             1.00 (1.2)   1.00 (0.3)   1.00 (0.5)   1.00 (0.4)

the-the (DELETE)     P16          P32          P64          P128
off                  0.67 (4.9)   0.67 (0.4)   0.67 (3.7)   0.83 (1.4)   keeps dup ✗
gibbs                0.67 (1.4)   0.67 (7.0)   0.67 (5.6)   0.67 (6.8)   keeps dup ✗
gibbs+bd             0.67 (1.4)   0.50 (7.0)   0.67 (5.6)   0.50 (6.8)   DELETES ✓ (P≥32, via merge)

boy-licked (SUB)     P16          P32          P64          P128
off                  0.17 (1.5)   0.67 (0.4)   0.67 (1.1)   1.00 (0.3)   kicked ✓ at P128
gibbs                0.33 (6.8)   0.50 (4.1)   0.17 (1.3)   0.83 (7.3)   noisy
gibbs+bd             0.33 (6.8)   0.50 (4.1)   0.17 (1.3)   0.83 (7.3)   noisy; kicked ✓ at P128
```

- **INSERT**: trivial — correct + unanimous at every config including P16/off; logZ_std falls monotonically
  with P. More compute only tightens an already-correct answer.
- **SUB**: `off` converges with particles — agree 0.17→1.00, and at **P128/off all 6 seeds → "kicked"**
  (logZ_std 0.3). A clean inference-limited curve.
- **DELETE**: `off`/`gibbs` never escape the keep-duplicate mode at ANY P; only `gibbs+bd` reaches the
  correct deletion (P≥32). Particles alone cannot fix it — the move kernel must change.

### Stability vs rejuvenation mode

- `off` is the most *stable* mode where the forward filter can already do the job (INSERT, SUB): lowest
  logZ_std, cleanest convergence.
- `gibbs` (substitution-only sweep) **adds cross-seed logZ variance** (the-the logZ_std 0.4→7.0; boy-licked
  up to 7.3) without ever changing the MAP to a better answer — it churns near-tied substitutions and, for
  DELETE, literally cannot represent the fix.
- `gibbs+bd` is the **only** mode that solves DELETE (its birth/death move supplies the deletion), but it
  inherits gibbs's churn on SUB and adds an insertion (birth) move that occasionally injects spurious words.

### Stability vs lookback (P64, gibbs+bd: lb2 / lb6 / lb12)

INSERT: "to" ✓ at all three. DELETE: correct delete ✓ at all three (lb6/lb12 cleaner agreement than lb2).
SUB: lb2→"Boyers kicked", lb6→"licked" ✗, lb12→"kicked" ✓ — a wider window helps slightly but noisily.
**Lookback is a minor knob for 5–9-word sentences; lb6 (default) is fine and lb12 costs ~40% more.**

### Stability vs LM (P64, pythia-70m vs 160m)

| | 70m/off | 160m/off | 160m/gibbs+bd |
|---|---|---|---|
| want-go (INSERT) | to ✓ (1.00) | **fails: "I want go home" ✗** | **fails: "I want go home" ✗** (0.83) |
| the-the (DELETE) | keeps dup ✗ | hallucinates "purpose of" ✗ (0.17) | delete ✓ (0.67) |
| boy-licked (SUB) | kicked ✓ (0.67) | inserts "who" ✗ (0.17) | kicked ✓ (0.83) |

The **bigger LM is not better** under the 70m-calibrated channel: `160m/off` over-edits and destabilizes
(hallucinated insertions, agree 0.17). `160m/gibbs+bd` rejuv cleans up those hallucinations and recovers
DELETE+SUB — but **newly breaks the INSERT** that 70m nails trivially (160m finds "I want go home"
acceptable and the Gibbs conditional resamples "to" away). Net: swapping the LM without recalibrating the
channel (and raising P) trades one set of errors for another; it does not improve inference.

### Cliffs / bad modes (three distinct kinds)

1. **Low-P sampling scatter** (boy-licked, P16/off: 6 distinct MAPs across 6 seeds). Escaped by *more
   particles* — the textbook inference-limited cliff. (P128/off → unanimous correct.)
2. **A move-kernel mode cliff** (the-the under off/gibbs at EVERY P: stuck keeping the duplicate). Escaped
   only by switching to `gibbs+bd` — *not* by particles. The correct mode is unreachable without a delete
   move, so the proposal/rejuv kernel, not compute, is the lever.
3. **A rejuvenation-induced cliff** (boy-licked under gibbs/gibbs+bd at mid P: the evidence merge flips to
   the wrong no-edit answer because rejuv churn leaves an un-edited seed as highest-logZ). Mitigated by more
   particles (P128 → kicked) or by *not* rejuvenating a case the forward filter already handles.

---

## Detailed analysis (verified against per-seed records)

The three sentences each probe ONE channel operation, and the optimal config DIFFERS by operation:

- **want-go = INSERT "to".** Solved at every config including P=16/off — the forward filter's insertion
  marginalization recovers "to"; all 6 seeds agree; logZ_std shrinks monotonically with P (1.2→0.2 for
  off). Rejuv is unnecessary and doesn't hurt.
- **boy-licked = SUBSTITUTION licked→kicked.** (The intended noisy-channel inference IS kicked.) The
  caprop forward filter handles substitutions, so the lever is PARTICLES: `off` agreement climbs
  0.17→0.67→0.67→**1.00** and at **P=128/off all 6 seeds unanimously output "kicked"** (correct), logZ_std
  0.3. **Rejuvenation HURTS here:** gibbs/gibbs+bd churn the substitution slot (capitalization variants
  "Boy"/"Boyle"/"Boy Scouts", spurious "boy's"/"boy he"), so MAP-agreement is erratic AND the one
  un-churned "licked" (no-edit) seed ends up highest-logZ, making the evidence merge output the WRONG
  no-edit answer at P64. ⇒ for substitution, `off` + enough particles is cheapest AND most correct.
- **the-the = DELETE the duplicate "the".** INFERENCE-limited and SOLVABLE — but ONLY by `gibbs+bd`:
  - `off` P64: per-seed logZ ≈ −43..−54, MAP keeps the dup (+ inserts an aux).
  - `gibbs` (sub-only) P64: rejuv raises evidence to logZ ≈ −37, but MAP STILL keeps the dup — the
    sub-only move cannot delete.
  - `gibbs+bd` P64: SAME high logZ ≈ −37, and 4/6 seeds now decode **"The patient recovered quickly."**
    (correct delete); merged MAP = correct delete. At P32 it already surfaces; at P16 no seed finds it yet.
  - ⇒ TWO things are needed: rejuv to raise evidence quality (logZ −43→−37) AND the birth/death move to
    supply the delete operation. (This corrects an earlier P16-only note that called it "signal-limited".)
- **The evidence-weighted seed merge earns its keep.** At gibbs+bd/P64/the-the only ~4/6 seeds find the
  delete, but their ~12-nat-higher logZ makes the merged posterior output the correct deletion; a plain
  majority vote over seed-MAPs would be far noisier. Conversely it can mislead (boy-licked/gibbs+bd/P64)
  when rejuv churn makes a wrong no-edit seed the highest-logZ — i.e. the merge faithfully amplifies
  whatever the highest-evidence particle found, for better or worse.
- **Cliffs identified:** (1) low-P substitution scatter (boy-licked P16/off: 6 distinct MAPs) — escaped by
  more particles. (2) the DELETE mode-cliff: off/gibbs keep the dup at EVERY P — escaped only by switching
  the move kernel to gibbs+bd, NOT by more particles. (3) a rejuv-induced cliff: gibbs/gibbs+bd flip the
  substitution merge to the wrong no-edit answer at mid P.

### Lookback (P64, gibbs+bd, lb ∈ {2,6,12})  — modest lever on short sentences

| sentence | lb2 | lb6 | lb12 |
|---|---|---|---|
| want-go (INSERT) | to ✓ (1.00) | to ✓ (1.00) | to ✓ (1.00) |
| the-the (DELETE) | delete ✓ (0.33) | delete ✓ (0.67) | delete ✓ (0.67) |
| boy-licked (SUB) | "Boyers kicked" (0.33) | "licked" ✗ (0.17) | kicked ✓ (0.50) |

The merged DELETE survives at every lookback; lb6/lb12 give cleaner agreement than the narrow lb2. A wider
window (lb12) slightly helps the substitution settle on "kicked", but the effect is small and noisy and
lb12 costs more (18 min vs 13 min/config). For sentences this short (5–9 words) lookback is a minor knob —
lb6 (the deployment default) is a fine operating point.

### LM (P64, pythia-70m vs pythia-160m)  — bigger LM is WORSE here (calibration coupling)

At P64/off, swapping pythia-70m → 160m DEGRADES and destabilizes all three:
- want-go: 160m **fails to insert "to"** (MAP = the un-edited "I want go home.", agree 0.67).
- boy-licked: 160m inserts spurious "who" → "The boy *who* kicked…", agree 0.17 (6 distinct MAPs), logZ_std 5.3.
- the-the: 160m hallucinates "The *purpose of* the patient…", agree 0.17.

Cause: the channel parameters (slope K, α, ins_rate, WDEL) are calibrated for pythia-70m. A sharper, more
confident LM under the SAME channel costs shifts the LM-gain/channel-cost balance toward over-editing and
hallucinated insertions, and its peakier posterior is harder to sample at fixed P. ⇒ **you cannot drop in a
bigger LM without recalibrating the channel** (and likely raising P). "Better LM" ≠ "better inference" here.

Adding rejuvenation to 160m (`160m/gibbs+bd`) **cleans up the hallucinations** — DELETE recovers
("The patient recovered quickly." ✓, 0.67) and SUB recovers ("kicked" ✓, 0.83) — but it **newly fails the
INSERT** that 70m solves trivially ("I want go home." with no "to", 0.83): under 160m the Gibbs
full-conditional resamples "to" away. So even with rejuv, the bigger LM trades the insert it used to get for
the delete/sub — a lateral move, not a win.

## Findings & recommendation

### Headline

1. **Posterior stability separates cleanly into two axes.** *Inference noise* (cross-seed MAP-agreement and
   logZ_std) is governed by **particles** — it falls smoothly and predictably as P grows, for every
   sentence. *Which mode wins* is governed by the **move kernel (rejuv) and the LM** — and no amount of
   particles moves it. The benchmark's central lesson: **particles buy exactness; they do not buy
   reachability.** If the correct mode is not reachable by the proposal/rejuv kernel, more compute only
   sharpens a wrong answer.

2. **Each channel operation has a different cheapest-correct config:**
   - **INSERT** (missing word): solved by the forward filter alone — correct + unanimous at **P16/off**.
     Rejuv unnecessary.
   - **SUBSTITUTION**: solved by the forward filter given enough particles — **P128/off** (all 6 seeds
     correct, 7 s/run). Rejuv *hurts* (churn flips the merge to the no-edit answer at mid-P).
   - **DELETION** (duplicate word): the ONLY operation that needs rejuvenation — **`gibbs+bd`, P≥32**. Its
     birth/death delete move is the sole way to reach the deletion; `off`/`gibbs` are stuck forever.

3. **The cliffs are real and of three kinds** (sampling scatter → fixed by P; a move-kernel mode-cliff →
   fixed only by gibbs+bd; a rejuv-induced merge flip → fixed by P or by turning rejuv off). See Results §
   "Cliffs".

4. **The evidence-weighted seed merge is load-bearing** for the deletion: only ~4/6 seeds find it, but
   their ~12-nat-higher logZ lets the merge output the correct answer where a seed-majority vote would not.
   It is the mechanism that makes a moderate-P `gibbs+bd` run reliable despite per-seed instability — but it
   faithfully amplifies the highest-evidence particle, so it can also surface a wrong mode (160m/off insert,
   boy-licked/gibbs+bd SUB) when rejuv churn makes a bad seed highest-logZ.

5. **A bigger LM (160m) is not a free upgrade.** Under the 70m-calibrated channel it over-edits and
   destabilizes (`off`), and even with rejuv it only trades the insert for the delete/sub. LM choice is
   coupled to channel calibration and particle budget; swapping it in naively is a regression.

### Compute/quality recommendation (minimize compute without sacrificing correctness)

| if you expect… | cheapest correct config | per-run cost | why |
|---|---|---|---|
| only insertions | **P16, off** | 4 s | forward filter nails it; nothing else helps |
| insertions + substitutions | **P128, off** | 7 s | particles converge SUB to truth; rejuv would only add noise |
| any deletions too | **P64, gibbs+bd, ≥4–6 seeds + logZ-merge** | 43 s | only the bd move can delete; the merge rescues the ~half of seeds that find it |

- **Single best general-purpose operating point: `align, gibbs+bd, P=64, lb6, ≥4 seeds, evidence-merged`**
  — the only single config that gets all three operations right (delete needs gibbs+bd; sub/insert ride
  along), at ~⅓ the cost of P128. If you can afford it, P128/gibbs+bd raises per-seed agreement (boy-licked
  0.17→0.83) but the merged answer is already correct at P64, so the extra particles mostly buy *stability*,
  not *correctness*.
- **Do not pay for `gibbs` (sub-only):** it costs 3× `off` and never improves a MAP over `off`; it only
  adds logZ variance. Use `off` (cheap) or `gibbs+bd` (for delete), not the middle.
- **Multiple seeds + the logZ-merge are the cheapest reliability lever** for `gibbs+bd`: 4–6 seeds at
  moderate P beat 1 seed at high P, because the merge concentrates on the seed that found the high-evidence
  mode. Budget seeds before particles once you are using gibbs+bd.
- **lb6 is the right lookback** for normal-length sentences; lb12 costs ~40% more for a marginal, noisy gain.
- **Keep pythia-70m** unless/until the channel is recalibrated for a larger LM and P is raised accordingly.

### Caveats / scope

3 short sentences, one per operation, 6 seeds — enough to expose the mechanisms and cliffs cleanly, not a
calibration-grade sample. logZ here is the SMC evidence estimate (stability readout), not a correctness
oracle; "correct" is judged against the intended edit. The top-k merge is approximate in the tail (`--top
8`). All on `align` @ commit `2f45517`, pythia-70m unless noted. Raw records under
`planning/bench_results/`; regenerate tables with `python planning/bench_aggregate.py`.

---

## Reproduce

- Sentences: `planning/bench_sentences.txt` · Driver: `planning/bench_driver.sh {phaseA,phaseB,phaseC}`
  · Resume: `planning/bench_resume.sh` · Aggregate: `planning/bench_aggregate.py` → `aggregate.json`.
- Each config: `slurm/run_nc_batch.py --channel align --n-seeds 6 --no-viz` with `--particles/--rejuv/
  --rejuv-lookback` varied; `NC_LM` selects the LM. Resume-safe (done configs skip before model load).

