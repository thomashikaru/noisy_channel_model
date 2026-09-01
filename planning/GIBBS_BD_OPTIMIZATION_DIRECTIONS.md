# Remaining optimization directions for the `gibbs+bd` indel move

*Written 2026-07-06 after the suffix-tail KV rewrite was built, measured ~3.6× slower on CPU, and reverted.
This is a forward-looking map for a fresh session: what is worth trying, what is not, and in what order.*

## Read these first (so you don't repeat work)

- **The KV suffix-tail rewrite is a DEAD END on CPU.** Built exact + gated, measured ~3.6× SLOWER
  (indel exec 28.6→103.5 s, full run 87.5→260.9 s at P=64). Reverted. Do not retry on CPU.
  Evidence: `planning/bd_kv_probe.py`, `planning/bd_kv_surgical.py`; correction banner in
  `planning/GIBBS_BD_SLOWDOWN_REPORT.md`; docstring note in `pairhmm_rejuv.make_gibbs_indel_sweep`.
- **The move is EXEC-bound, not compile-bound** (surgical split: COMPILE 3.8 s vs EXEC 28.6 s @P64). So
  per-run recompile (the report's "Cause 3") is negligible — do NOT bother lru_caching the indel sweep.
- **Already banked (do not re-do):** single-forward `seq_token_logprobs` teacher forcing (cut bd cost
  ~348→60 s, `REJUV_BIRTH_DEATH_PLAN.md` §12) and dedup over the degenerate post-resample cloud.
- **Deployment is CPU-only** (pythia-70m; cluster `USE_GPU=0`). This is the key constraint — several
  "wins" that would work on GPU do not on CPU.

## The cost being optimized

Per unique particle, the indel move scores `{no-op} ∪ {Wmax·Kc insertions} ∪ {Wmax deletions}` candidate
sentences, each a full forward over `LCTX ≈ seed_len + Wmax` tokens, producing a 50304-wide vocab
log-softmax per position. Cost ∝ `Wmax·Kc·LCTX` forward-token-work (deduped over the ~2–6% unique
particles). `Kc` = unique observed surfaces + ~15 funcwords. Insertions dominate (deletions have no
candidate dimension).

---

## Directions, ranked by honesty about promise

### A. Exact (posterior-preserving) — essentially exhausted

- **Skip provably-masked candidates.** ~20–30% of the grid is invalid (gap > n_words; pad deletions) and
  is currently *scored then masked to −inf*. Not computing them is exact but fiddly under JAX static
  shapes; buys ~1.2×. Low ROI.
- Everything else exact is tapped. You must run the full-vocab log-softmax per scored position (the
  normalizer is required for a true log-prob), and prefix-sharing across candidates only pays off via a
  KV cache, which loses on CPU. **Do not spend more effort on exact CPU speedups.**

### B. Approximations — real, but validate against the exact result on a battery subset

- **Surprisal-gated insertion gaps (most principled untried lever).** A dropped word shows as a locally
  improbable bigram transition; a doubled word as a repeat. Score insertions only at gaps flagged by those
  cues instead of all `Wmax` gaps → cuts the dominant grid dimension. Tied to the actual signal the move
  exploits. Risk: misses a restoration at a non-spiking gap. **Prototype this one first if speed is needed.**
- **Cloud / sentence gating.** Fire the indel sweep only on particles/items that look suspicious (contain a
  repeat, or a low-probability transition); clean sentences — the common case — skip it entirely.
- **Smaller candidate pool** (fewer funcwords / cap `Kc`). Direct multiplier on the insertion grid; risks
  missing a restoration target.
- **Particle count is already at P=128** for the battery (align-opt/calibration runbook), so the
  "P=256→128 ≈2×" is already banked. Below ~P=32 breaks DELETE-dup reachability (posterior-stability
  finding) — little headroom left.

### C. Systems — underused and exact (same computation)

- **More CPU cores per job.** jax-on-CPU scales with cores; the `/orcd-cluster` skill notes `CPUS=8–16`
  closes much of the per-job gap. Free, exact, likely underused.
- **Cluster parallel fan-out.** Per-item latency is irreducible exactly; throughput comes from running many
  items at once (the intended cluster use).
- **GPU would change the KV calculus.** The per-call overhead that killed KV on CPU is largely hidden on
  GPU. Moot unless you move off CPU — but *if* you ever do, re-measure the KV suffix-tail path before
  assuming it still loses.

### D. The deeper question (highest ceiling, highest risk)

The forward filter **already marginalizes word insertions/deletions** (see the model-edit-capability
finding) — the expensive post-hoc indel *rejuvenation* sweep exists because the filter's proposal doesn't
*reach* those readings well, not because it can't represent them. The highest-ceiling direction is a
**smarter channel-aware proposal inside the filter** that reaches insert/delete readings during filtering,
reducing reliance on the sweep. This is a core-inference redesign: high effort, high uncertainty, and it
needs the same measure-first discipline. Open it only if throughput becomes a real blocker.

---

## Recommended order (if throughput becomes a real constraint)

Given the calibration battery run is already complete, there may be no urgent problem. If one arises:

1. **Systems first (free, exact):** more CPU cores per job + cluster fan-out.
2. **Then a principled approximation:** surprisal-gated insertion gaps or cloud gating — validate against
   the exact result on a battery subset before trusting it.
3. **Do NOT** reach for another algorithmic rewrite of the LM scoring; the measurement says that vein is
   tapped. **Do NOT** retry KV on CPU.
4. Consider **(D)** only as a deliberate research effort, not a quick optimization.

## Meta-lesson (so this doesn't recur)

A flagged "perf win" is a HYPOTHESIS to MEASURE, not automatically a debt to pay. The KV rewrite was
comment-flagged as "the Phase-2 perf win" and re-raised as the slowdown report's #1 fix, but
`REJUV_BIRTH_DEATH_PLAN.md` §12 had already warned it was "aimed at the wrong cost." A forward-count / FLOP
cost model does not capture framework per-call overhead. Measure on a representative case before building.

## 2026-09-01 addendum — USER DIRECTIVE: targeted allocation BEFORE main_bd

Explore next session, before any `main_bd` run (user decision 2026-09-01). Extends the
"surprisal-gated insertion gaps" item above with two refinements:

1. **Trigger on RELATIVE surprisal:** gate rejuvenation on a unit's contextual LM surprisal being
   high **relative to its unigram surprisal** (the word-specific baseline — a rare word is allowed
   to be contextually surprising; the same decomposition the frequency-aware insertion cost uses).
   Both signals are already computed per item (`lm_word_surprisals` runs for LOOKAHEAD anyway;
   `unigram_surprisal` exists), so the gate is one subtraction at run time.
2. **Target the proposal:** propose rejuvenations at reanalysis-likely locations (the spike
   profile / alignment strain), not across the whole sentence — cuts the dominant O(Wmax×Kc)
   indel grid, and matches bd's actual remaining job now that the off arm is clean post
   lookahead_proposal: localized structural repairs (chen2023 passive↔active, tabor2004/huang2024
   "who was") at identifiable disambiguation points.

Correctness note: gating computed from the OBSERVED sentence is a constant of the target, so a
position-selection mixture built from it keeps every Gibbs component invariant. Gating on the
particle's current parse makes the kernel state-dependent — needs SMCP3/MH, avoid unless
deliberate. Validation rule unchanged: A/B any gated variant against the exact full sweep on a
battery subset first, and measure before building (the KV meta-lesson above).
