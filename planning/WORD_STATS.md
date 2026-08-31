# The per-word model outputs, in plain English

What the numbers in the `words` block (and `words.csv.gz`) mean, how they are computed, and what
to watch for when reading them. Written for analysis time; the code lives in
`src/genjax_port/word_stats.py`, the hooks in `pairhmm_smc.run(word_stats=, diag=)`, and the
design record in `NOISY_CHANNEL_HARNESS_IMPLEMENTATION_PLAN.md` §3.

**One rule above all: these are observers.** Turning them on changes nothing about inference —
no extra randomness, no change to the certified math. That is a tested guarantee
(`tests/test_word_stats.py`), not a hope: a run with the hooks on is bit-identical to a run with
them off.

## The outputs at a glance

| column | one-line meaning |
|---|---|
| `surprisal_nc` | how surprising this observed word was to the NOISY-CHANNEL reader (nats) |
| `surprisal_lm` | how surprising it was to a plain LM reading the sentence literally (nats) |
| `p_copy` / `p_sub` / `p_ins` | posterior probability this word was read verbatim / corrected / dismissed as spurious (sums to 1) |
| `p_err` | 1 − p_copy: the probability the model thinks something is wrong with this word |
| `del_before`, `del_after_last` | expected number of intended words the reader restored in the gap before this word / after the last one |
| `p_err_positional` | a cheap position-based approximation of p_err (see caveats) |
| `rejuv_*`, `indel_*` | diagnostics of the rejuvenation moves, not model quantities (see below) |

## 1. `surprisal_nc` — surprisal under the noisy-channel reader

The quantity people usually call "surprisal" is −log P(word | words so far). For the
noisy-channel model the natural analogue is: how much probability mass did the model assign to
the observed sentence continuing with this word, **given that the writer may make errors**? A
typo'd word is far less surprising to this reader than to a plain LM, because "they meant the
obvious word and slipped" is a cheap explanation.

Formally: let Q_k be the total probability of all generative stories (intended sentence +
channel errors) that produce exactly the first k observed words. Then

    surprisal_nc of word k  =  −log( Q_k / Q_{k−1} )

and the per-word values add up to the whole-sentence score exactly:
`sum(surprisal_nc) + surprisal_end_nc = −logZ`. That identity holds to machine precision on
every record, so it doubles as a per-item integrity check.

**How it is estimated.** From the ONE inference run you already paid for — no separate
prefix-by-prefix runs. At each step of the particle filter, the same dynamic-programming row
update that extends every particle also (as a second output of the identical computation) says
how much of the new mass consumed observed word k at that moment — either "the intended word
aligned to it" or "it was a spurious insertion". Those per-step contributions, weighted by the
particle weights before any resampling touches them, accumulate into Q_k. On the toy model this
estimator was checked against exact enumeration and matches at the same tolerance as the logZ
gate.

**Caveats you must keep in mind when analyzing it:**

- The channel's form score is unnormalized (it sums to ~1.05–1.4 per word rather than 1), so
  every `surprisal_nc` carries a small, near-constant offset (~0.05–0.3 nats). This is the same
  convention `logZ` has always used. Any analysis with an intercept absorbs it; just do not
  read the absolute values as calibrated probabilities.
- Only defined at `lm_temp = 1` (the run refuses otherwise; the worker then skips the block).
- A `null` in `prefix_logq` or `surprisal_nc` means "unreachable under the band", not
  "infinitely surprising". Never impute a large number for it.

## 2. `surprisal_lm` — the plain-LM baseline, built to be comparable

The obvious comparison is "what would a plain LM say about this word?" — but that is only fair
if both readers score the *same tokens*. `surprisal_lm` therefore scores exactly the token
spans a verbatim-copy particle pays for (same tokenization, same leading-space handling, same
context prime), in one forward pass. So per word you can subtract: on the smoke run, the typo
'teh' costs 16.6 nats literally but 8.4 under the channel — that gap IS the noisy-channel
effect the harness exists to measure. `surprisal_end_lm` is the LM's end-of-sentence cost.

## 3. `p_copy` / `p_sub` / `p_ins` and the deletion columns — what the reader thinks happened

After inference finishes, every surviving particle carries an intended sentence, and the
alignment between it and the observed sentence can be worked out exactly (a small
forward–backward pass per particle, done in double precision on the host). For each observed
word this yields the posterior probability that it was:

- **read verbatim** (`p_copy` — case-insensitive: "she" → "She" counts as verbatim),
- **corrected to a different intended word** (`p_sub`), or
- **dismissed as a spurious insertion** (`p_ins`).

These three sum to 1 per word (asserted, per particle). `del_before` is the expected number of
missing intended words the reader restored in the gap just before this word ("The mother gave
the candle [to] the daughter" puts deletion mass at that gap); `del_after_last` is the trailing
gap. Averages are over particles, weighted by the final (terminal-corrected) particle weights.

One technical point, recorded so nobody "fixes" it into a bug later: the filter's band is
applied at the end of each DP row, after the insertion sweep — so the backward pass must run on
the unmasked lattice with the band only on row-to-row transitions. Done naively (mask
everywhere), the three probabilities stop summing to 1. The implementation mirrors the kernel
and was checked against brute-force enumeration of every alignment path.

**`p_err_positional`** is the cheap approximation the θ-refresh already used internally: "is the
word at position i a copy of observed word i?". It is exact when nothing shifted the alignment
and wrong when an insertion/deletion did. It is kept precisely so you can see where the two
disagree; prefer the exact columns for analysis.

## 4. The rejuvenation diagnostics — about the sampler, not the language

`rejuv_*` and `indel_*` describe what the rejuvenation moves did, which is useful for
diagnosing inference (was a correction found by the forward pass or rescued by a move?), not
for psycholinguistic claims:

- `rejuv_n_events` / `rejuv_change_rate` / `rejuv_stay_prob`: how often the substitution sweep
  revisited this slot, how often it actually changed the word there, and the average
  full-conditional probability of keeping it. (The move's formal "acceptance rate" is always 1
  — these are the informative numbers instead.)
- `indel_p_ins_gap_before` / `indel_p_del` (+ chosen-edit counts): the Gibbs indel move's own
  probabilities of inserting at the gap before this slot / deleting this slot, and what it
  actually sampled. `p_noop` near 1 on a clean sentence is the move's over-edit guard working.

Caveat: these are indexed by intended-word **slot**, matched to observed units by position — so
under a parse that inserted or deleted words, read them positionally, like `p_err_positional`.

## 5. Multi-seed records

With `N_SEEDS > 1`, each seed writes its own words block, and the merged record follows each
quantity's own algebra: prefix masses are averaged as probabilities (then surprisals
recomputed), posterior expectations are evidence-weighted (a collapsed seed with low logZ gets
little say, exactly like the hypothesis merge), rejuvenation counts are pooled and rates
recomputed, and `surprisal_lm` is deterministic so it is taken once. `words.csv.gz` keeps both:
one row per word per seed, plus the `merged` row.

## 6. If something looks wrong

- `sum(surprisal_nc) + surprisal_end_nc ≠ −logZ` on a record → a real bug; report it.
- `words.status == "error"` → the per-word pass failed for that item; the sentence-level result
  is still valid, and the traceback is in the record.
- Nulls in the surprisal columns → band-unreachable, by design (see §1).
- p_* not summing to 1 → cannot happen silently; the code asserts it per particle.

## Pointers

- Code: `src/genjax_port/word_stats.py` (estimators), `slurm/run_nc_batch.py` (`words` block
  assembly + multi-seed merge), `experiments/collect.py` (tables).
- Design record with the derivations: `planning/NOISY_CHANNEL_HARNESS_IMPLEMENTATION_PLAN.md` §3.
- Gates: `src/genjax_port/tests/test_word_stats.py`, `experiments/tests/test_phase3_worker_collect.py`.
