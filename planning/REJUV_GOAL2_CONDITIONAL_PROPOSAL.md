# Goal 2 — Suffix-aware (near-Gibbs) substitution proposal

> **✅ DONE (2026-06-16, session 2). Uncommitted on `genjax-port-unified`.** `manual_subflip_move`
> (`src/genjax_port/rejuv_bridge.py`) now proposes from the EXACT suffix-aware local conditional
> `q(x) ∝ P_LM(x|prefix)·channel(x)·P_LM(suffix|prefix,x)` via ONE `[P*K,M]` batched forward over K
> candidate buffers (memory-efficient gather; suffix summed over `[posc,i_len)`). This equals the MH
> target restricted to the candidate set ⇒ the move is Gibbs, **measured accept ≈ 1.0000** (kept the MH
> coin as a live correctness assertion, per option (a) of step 2). The motivating under-proposal is
> fixed: corrective candidates justified only by downstream evidence now carry real proposal mass.
>
> **Test:** the old `test_manual_subflip_detailed_balance` only revisited the LAST word (no suffix), so
> it never exercised the new term — replaced by `test_manual_subflip_full_conditional` (revisits a
> MIDDLE word, checks vs a brute-force suffix-aware posterior + asserts accept≈1). Integration test
> `test_aligned_conditional_composes_with_forward_deletions` still PASSES.
>
> **Perf:** the exact version costs ~K/2× the old move's forwards. Two cuts landed: `MAX_SUB_CANDIDATES`
> (config) capped K 129→33 (~4×); K-bucketing trims per-window. The structural fix — a prefix-KV-cache
> that removes the K-fold shared-prefix recompute — is **de-risked (both spikes passed) but NOT yet
> built**; it is the PRIMARY next task. See the `rejuv-prefix-kv-cache-spike` memory + the migration
> memory's NEXT section. The cheap-fallback (1–2 token lookahead) was NOT needed (our suffix is already
> short — ≤ lookback — because rejuv only revisits the last-`lookback` window).


## Problem
Reanalysis exists to let **downstream** evidence flip an **earlier** word ("Because of all the
threats, my dog has become overweight" → *treats*). But the substitution-flip proposal samples a
candidate from the **local** distribution only — the LM at the flip position plus the channel
(`q_logits = lm_at[...] + cand_l`, `rejuv_bridge.manual_subflip_move` ~L457). At the moment you
flip "threats", its left context does not favor "treats", so "treats" has low proposal probability
and is **rarely proposed**. The MH acceptance ratio *does* re-score the whole suffix (`chain_new`
vs `chain_old`), so the correction would likely be accepted *if offered* — but the sampler almost
never offers it. Result: many forwards, little useful movement, and the one reanalysis you wanted
is the one that doesn't happen.

Note the **deletion** proposal already fixes this for its case via one-step lookahead
(`smc_substitution.deletion_gap` ~L109, the `look_o` reweight). The substitution flip has no
lookahead. This goal gives it one — ideally the full-conditional version.

## Key idea (answers "no bidirectional LM?")
A causal LM cannot *generate* leftward, but it can **score** known continuations. The suffix tokens
already exist in the buffer (observed/intended). So for each candidate token `x` at the flip
position `k`, place `x` at `k` and ask the causal LM how likely the already-seen suffix is given
`x`. The LM only ever scores observed tokens — no bidirectionality needed.

Because substitution candidates are a **small** edit-neighbor set (`cand_x [K]`, K ~ 1–10;
`noise_word.word_sub_candidates`), you can afford the **exact** version.

### Target design: full-conditional ("Gibbs") proposal
Make the proposal the exact local posterior over the candidate set, including all observed-so-far
downstream evidence:

  `q(x) ∝ P_LM(x | prefix) · channel(x) · P_LM(suffix | prefix, x)`

- Build `K` copies of the buffer, each with candidate `k` placed at position `k`.
- **One** batched forward over `[P·K, M]` gives every position's logits for every candidate.
- For each candidate sum the suffix log-probs over `[k, i_len)` (reuse the `chain_from_pos`
  masking logic already in `manual_subflip_move`), add `lm_at` (position-k prior) + `cand_l`
  (channel).
- That sum **is** the full conditional. Sample `x_new` from it.

Consequences:
- The corrective candidate ("treats") now carries real probability once "overweight" is in the
  suffix → it actually gets proposed.
- The proposal equals the MH target restricted to the candidate set, so the move becomes a
  **Gibbs update: acceptance ≈ 1**. You can drop the accept/reject coin for this move (no
  rejection waste). Keep the weight computation available for Goal 3 / asymmetric moves.

### Cheap fallback: 1–2 token lookahead
If the full-conditional forward proves too expensive (large K × long suffix), mirror
`deletion_gap` exactly: score only the next 1–2 tokens under each candidate (one extra `[P·K, M]`
forward at `k+1`). **Honest limitation:** this captures *local* disambiguation (agreement, short
collocations) but **misses long-range** cases like "overweight". Prefer the full-conditional unless
profiling forces the fallback.

## Read these first
- `src/genjax_port/rejuv_bridge.py`: `manual_subflip_move` (~L431) — the current local proposal
  (`q_logits` ~L457), the suffix re-score (`chain_from_pos` ~L445), the weight (~L471), accept
  (~L472). This is what you modify.
- `src/genjax_port/smc_substitution.py`: `deletion_gap` (~L83) — the existing lookahead pattern to
  imitate (`bufs` broadcast over candidates, `look_o`, `logZ`).
- `src/genjax_port/noise_word.py`: `word_sub_candidates`, `word_sub_loglik` — the candidate set and
  channel log-likelihoods (already assembled into `cand_l` by the caller).
- `src/genjax_port/tests/test_rejuv_bridge.py`: detailed-balance test (must adapt — see below).
- `docs/model.tex` Thm 2 / eq smcp3 — the move's correctness math.

## Implementation steps
1. In `manual_subflip_move`, replace the local `q_logits` with the full-conditional construction:
   broadcast the buffer to `[P·K, M]` with each candidate placed at `posc`, one `_raw_logits`
   forward, gather per-candidate suffix sums over `[posc, i_len)` + `lm_at` + `cand_l`. This
   **replaces** one of the two existing forwards' purpose (you now forward the K-candidate batch
   instead of the single proposed buffer). Mind the batch shape: `[P·K, M]` — confirm it fits the
   bucket/compile story and the dedup work from Goal 1 (do Goal 1 first; they touch the same call).
2. Since the proposal is now exact-conditional, either (a) keep MH and observe acceptance ≈ 1
   (cheap correctness check that you built it right), or (b) convert to a Gibbs move and drop the
   coin. Recommend (a) first to validate, then (b).
3. Apply the same change wherever the sub-flip proposal lives if there are two copies (the `@gen`
   `Rejuvenate` path uses `rejuvenation.cand_prop`; the production path uses
   `manual_subflip_move`). Production is `manual_subflip_move` — do that first; note the `@gen`
   oracle (`rejuvenation.py`) for parity if you want the cross-check.

## Validation
- **Acceptance rate:** with the full-conditional proposal under MH, the accept rate should jump to
  ≈ 1 (it's Gibbs). If it doesn't, the suffix-scoring is wrong (likely an off-by-one in the
  `posc-1` logits-vs-token indexing, or the mask bounds).
- **The motivating case:** run the "threats/treats" example (and similar late-disambiguation
  sentences) through `run_smc_conditional_rejuv_aligned`; the posterior should now place real mass
  on the corrected reading where the old local proposal did not. Add such a case to
  `tests/test_noisy_channel.py` if not present.
- **Detailed balance:** if you keep MH, the existing detailed-balance test must still pass; if you
  switch to Gibbs, replace it with a Gibbs-invariance check (the move leaves the conditional target
  invariant). Do **not** silently delete the balance test.
- **Eval:** `eval_rejuv.py` mean must improve or hold; this goal is the main lever on *quality*.

## Done when
The corrective candidate is proposed with non-trivial probability on late-disambiguation cases,
acceptance is ≈ 1 (or the move is Gibbs), `eval_rejuv.py` improves/holds, and the balance/invariance
test passes. This goal should be the one that fixes "doesn't consistently lead to better
noisy-channel inferences."
