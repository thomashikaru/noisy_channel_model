# Implementation plan — KV suffix-tail scorer for the `gibbs+bd` indel move

> ## ❌ OUTCOME (2026-07-06): BUILT, MEASURED ~3.6× SLOWER, REVERTED. This plan is a DEAD END — do not execute it.
>
> The plan below was implemented exactly (gated: the toy gate confirmed the KV logits == the full-forward
> `_indel_logits` to <1e-3). Then measured on pythia-70m/CPU (`planning/bd_kv_probe.py` + `bd_kv_surgical.py`):
> indel exec **28.6 s → 103.5 s**, full gibbs+bd run **87.5 s → 260.9 s** — **~3.6× SLOWER**. Reverted.
>
> **Why it failed** (the "expected win" math below is wrong): (1) a KV candidate feed over a FULL-length exact
> tail costs ~a full forward — the indel move CANNOT window the tail (it scores insertions at gap 0, whose
> suffix is the whole sentence), so Stage 2's windowing (the only real speedup) is unavailable; and (2) penzai's
> KV-caching transformer has large PER-CALL overhead on CPU (variable bind/unbind, per-candidate
> `unfreeze_as_copy` of the prefix KV, cache setup, prefill over a ~2× padded buffer) that a forward-count cost
> model doesn't capture. The plain `seq_token_logprobs` whole-sentence forward (already the code) is the
> efficient path. **This was ALSO foreseen** in `REJUV_BIRTH_DEATH_PLAN.md` §12 ("the planned suffix-tail KV
> sharing was aimed at the wrong cost … lower marginal value"); the slowdown report re-raised it as Cause 1
> anyway, and the measurement settled it. See the correction banner in `GIBBS_BD_SLOWDOWN_REPORT.md`.

Target: **Cause 1** of `planning/GIBBS_BD_SLOWDOWN_REPORT.md` — the indel move (`bd_mode="gibbs"`)
re-scores the *whole sentence* once per (gap, candidate) with `seq_token_logprobs`, instead of scoring
only the candidate-dependent suffix the way the substitution sweep already does. This plan makes the indel
move reuse the substitution sweep's tail-scoring machinery (`model.tail_logprobs`, the forced-KV scorer)
plus a one-forward prefix-sum, so the prefix LM is computed once and shared across candidates.

**Expected win (estimate, to be measured):** the dominant cost is the full-vocab (50304-wide) log-softmax
per scored position. Today that is `Wmax·Kc·LCTX` softmaxed positions per particle (≈ 11·21·13 ≈ 3000 for
an 8-word sentence). Stage 1 (exact, full tail) cuts the redundant prefix re-scoring; Stage 2 (windowed
tail) bounds the suffix. Together ≈ **2–3× on the indel move's LM work** — the single biggest lever on the
~150 s/item the indel move adds, but realistically a few-fold, not 10×. Deeper wins need Cause 2 (chunked
batch) and Cause 3 (per-item recompile), which compose with this and are tracked separately.

This plan changes **only** the gibbs indel move's LM scoring. The substitution sweep, the forward filter,
the `mh`/`smcp3` birth-death modes, and the toy/test path are untouched (the toy keeps the existing
`score`-based `_indel_logits` as the correctness reference).

---

## 1. The identity this exploits

For the current parse with word token-spans `w_0..w_{n-1}` (seed = `[EOS, "."]`, `sl=2` tokens, `T_max=1`
so word index = token index), an autoregressive LM factorizes any *edited* parse around the edit position.
Let `S[g] = Σ_{j<g} log P(token_j | seed, token_{<j})` be the cumulative LM logprob of the first `g` word
tokens of the **no-op** parse — a prefix-sum of one forward.

- **Insert candidate `c` at gap `g`** (parse = `w_0..w_{g-1}, c, w_g..w_{n-1}`):
  `LM = S[g] + tail_lp([c, w_g..w_{n-1}, EOS?] | seed, w_0..w_{g-1})`.
- **Delete word `i`** (parse = `w_0..w_{i-1}, w_{i+1}..w_{n-1}`):
  `LM = S[i] + tail_lp([w_{i+1}..w_{n-1}, EOS?] | seed, w_0..w_{i-1})`.
- **No-op:** `LM = S[n] + log P(EOS | seed, all)` — read directly from the same forward.

The prefix term `S[·]` is **identical** to the edited parse's prefix LM because the prefix tokens are
unchanged by an edit at/after their position and the LM is autoregressive — so `S[·]` is exact, not an
approximation. Only `tail_lp` is candidate-dependent, and it is exactly what `model.tail_logprobs`
(the KV scorer that prefills the prefix once and shares it across the `K` candidates at a gap) computes.

So the per-candidate LM cost drops from "score all `n` positions" to "look up a prefix-sum + score the
tail," and the `Kc` candidates at one gap share a single prefix prefill.

This is the same decomposition the substitution sweep uses (`make_sweep` docstring,
[pairhmm_rejuv.py:490-502](../src/genjax_port/pairhmm_rejuv.py#L490-L502)); the indel path simply never
adopted it ([pairhmm_rejuv.py:948](../src/genjax_port/pairhmm_rejuv.py#L948): *"no suffix-tail KV
cancellation yet; that is the Phase-2 perf win"*).

---

## 2. What exists to reuse (do not re-invent)

| piece | where | reuse as |
|---|---|---|
| KV tail scorer, prefill shared across `K` cands | `lm_penzai.batch_tail_logprobs` (use_kv=True), injected as `model.tail_logprobs` ([pythia_word_caprop.py:315](../src/genjax_port/pythia_word_caprop.py#L315)) | the tail LM, unchanged |
| substitution tail-input builder (prefix boundary, `_pack`, suffix slice, EOS append) | `_tail_inputs` ([pairhmm_rejuv.py:324-354](../src/genjax_port/pairhmm_rejuv.py#L324-L354)) | template for the insert/delete variants |
| per-token logprobs in one forward | `lm_penzai.seq_token_logprobs`; already consumed by `_lm_logprior` ([pairhmm_rejuv.py:170-199](../src/genjax_port/pairhmm_rejuv.py#L170-L199)) | the no-op prefix-sums |
| per-word buffer surgery | `_insert_word` / `_delete_word` ([pairhmm_rejuv.py:661-695](../src/genjax_port/pairhmm_rejuv.py#L661-L695)) | build the edited parse to pack |
| windowed tail budget convention | `mt_tokens = (rejuv_lookback+1)*T_max + 1` ([pairhmm_smc.py:638](../src/genjax_port/pairhmm_smc.py#L638)) | Stage 2 window |
| dedup-over-unique-particles wrapper | `_dedup_indel_logits` ([pairhmm_rejuv.py:1124-1158](../src/genjax_port/pairhmm_rejuv.py#L1124-L1158)) | wraps the new logits fn unchanged |
| channel marginal per candidate | `channel_carry` inside `_make_bd_score_fn` ([pairhmm_rejuv.py:968-985](../src/genjax_port/pairhmm_rejuv.py#L968-L985)) | factor out; keep as-is |

---

## 3. Stage 1 — exact prefix-sum + full-tail KV scoring (the committed deliverable)

Bit-identical to today's logits (within fp tolerance); just computed via tail + prefix-sum instead of a
full per-candidate forward. New code lives in `pairhmm_rejuv.py`; nothing else changes behavior.

### 3.1 New helper: no-op prefix-sums
`_noop_lm_prefix(ctx, bufs, total, done) -> (S [N, n_out+1], noop_lm [N])`
- Generalize `_lm_logprior` to also return per-token logprobs (it already computes them): when
  `model.seq_token_logprobs` is present, `tok_lp = seq_token_logprobs(bufs)`; else the existing per-position
  loop. `S = concat([0], cumsum(tok_lp[:, sl:sl+n_out]))` (word-token positions only, matching
  `_lm_logprior`'s mask). `noop_lm = S[total] + done * logP(EOS | all)` (the EOS slot is `tok_lp` at index
  `sl+total`, exactly as `_lm_logprior` reads it).
- One forward per particle (deduped). This single call also covers the no-op column of the move set.

### 3.2 New helper: edited tail-inputs
`_post_edit_tail_inputs(word_tok2, word_len2, boundary_tok, done, sl, Wmax, T, mt, eos_id, seed_ids)`
- Generalize `_tail_inputs`: given the **already-edited** per-word buffer `(word_tok2, word_len2)` and a
  per-particle prefix token boundary `boundary_tok [N]`, build `(ctx_bufs [N,LCTX], ctx_lens [N]=sl+boundary,
  tail [N,K,mt], tail_len [N,K])` — `_pack` the edited buffer, slice the tail tokens from `boundary_tok`,
  append EOS for done particles. `_tail_inputs` is the `K=1`-per-row, substitution-splice special case;
  this is the same body with the splice already applied.
- **Insertions:** for each gap `g`, splice every candidate via `_insert_word(word_tok, …, gap=g, x=cand_c)`
  → tail = `[c, w_g..]`; `boundary_tok = cumsum(word_len)[g-1]` (tokens of words `< g`). Produce
  `tail [N, Wmax, Kc, mt]` → reshape rows `B=N·Wmax`, `K=Kc` for `tail_logprobs`.
- **Deletions:** for each `i`, `_delete_word(word_tok, …, i)` → tail = `[w_{i+1}..]`; `boundary_tok =
  cumsum(word_len)[i-1]`; `K=1`. Rows `B=N·Wmax`, `K=1`.

### 3.3 New logits builder
`_indel_logits_tail(word_tok, word_len, word_surf, n_words, done, ctx, theta_costs, cand_*, tail_fn, max_tail)`
mirrors `_indel_logits`'s output contract `[N, 1 + Wmax·Kc + Wmax]` (column 0 = no-op, then gap-major
insertions, then deletions), but:
1. `S, noop_lm = _noop_lm_prefix(...)`.
2. Insertion LM: `ins_tail_lp = tail_fn(ins ctx, ctx_len, ins tail, tail_len).reshape(N, Wmax, Kc)`;
   `ins_lm = S[:, gaps, None] + ins_tail_lp`.
3. Deletion LM: `del_tail_lp = tail_fn(del ctx, …).reshape(N, Wmax)`; `del_lm = S[:, dels] + del_tail_lp`.
4. **Channel** per candidate from a factored-out `_indel_channel_logits` (the `channel_carry` half of the
   current `_make_bd_score_fn`, computed over the edited parses exactly as `_indel_logits` does today).
5. `logits = lm_temp * LM + channel`, with the **same** validity masks (`g <= n_words < Wmax` for
   insertions, `i < n_words` for deletions) and the same `NaN -> -inf` guard
   ([pairhmm_rejuv.py:1064-1077](../src/genjax_port/pairhmm_rejuv.py#L1064-L1077)).

`_indel_apply` (the categorical sample + splice) is **unchanged** — it consumes `logits` only.

### 3.4 Wire into the production sweep only
- `make_gibbs_indel_sweep`: build `_logits` to call `_indel_logits_tail` with `tail_fn =
  ctx.model.tail_logprobs or partial(_tail_chain_uncached, ctx.model.lm_fn)` (same selection as `make_sweep`,
  [pairhmm_rejuv.py:511](../src/genjax_port/pairhmm_rejuv.py#L511)) and `max_tail = Wmax*T_max + 2`
  (full tail, exact). Keep the `_dedup_indel_logits` wrapper around it. Add `max_tail` to the signature
  (default = full).
- `gibbs_indel_move` and the toy tests keep calling the existing **`score`-based** `_indel_logits` — that
  stays as the reference path. Only the filter-shaped sweep switches.

No change to `pairhmm_smc.run` is required for Stage 1 (full tail = default). The win is automatic for
`rejuv="gibbs+bd"`.

---

## 4. Stage 2 — windowed tail (approximate, the larger constant-factor win)

Bound the rescored suffix to a few words after the edit, exactly as the in-filter substitution sweep does
(`max_tail = (lookback+1)*T_max + 1`). Beyond the window, reuse the no-op's suffix logprobs:
`LM ≈ S[g] + tail_lp(window | prefix) + (S[n] - S[g + win])` for insertions (the far suffix scored without
`c` in context — the standard single-word-context-decay approximation the substitution sweep already
relies on). This makes each candidate `O(window)` instead of `O(suffix)`.

- Plumb a knob `bd_lookback` (default = full/exact) through `pairhmm_smc.run` →
  `make_gibbs_indel_sweep(max_tail=…)` and through the `pythia_word_caprop.run` / `calibration_word_action_smc`
  env (`NC_BD_LOOKBACK`), mirroring `rejuv_lookback`.
- Document it as an approximation (like `rejuv_lookback`); keep exact (full tail) as the default until the
  battery confirms the windowed restorations match.

---

## 5. Correctness & certification (the bar before this ships)

The codebase certifies indel-move changes by exact equivalence; mirror that.

1. **Decomposition identity (toy, exact).** New test `test_indel_logits_tail_matches_score`: on the toy
   model (no `seq_token_logprobs`/`tail_logprobs` → uncached fallbacks), assert `_indel_logits_tail(...,
   max_tail=full)` equals the existing `_indel_logits(..., score)` within ~1e-4 over a small cloud. This
   proves `S + tail_lp == _lm_logprior` and the channel factoring is faithful.
2. **KV == seq decomposition (Pythia, exact).** Small Pythia example: `_indel_logits_tail` with the KV
   `tail_logprobs` equals the `score`-based `_indel_logits` within fp tolerance (full tail). Reuses the KV
   scorer's own certification (`planning/kv_cache_spikes/tail_scorer_verify.py`).
3. **Unchanged toy gates still pass:** `test_gibbs_indel_conditional_and_zero_weight`,
   `test_gibbs_indel_dedup_equivalence` ([tests/test_rejuv_birth_death.py:417,469](../src/genjax_port/tests/test_rejuv_birth_death.py#L417-L469))
   — the toy path is untouched, so these are regression guards.
4. **Dedup still exact:** `_dedup_indel_logits` wraps the new logits fn; the existing dedup-equivalence test
   covers it (the logits fn is still a pure function of particle state).
5. **End-to-end behavior:** on the indel battery subset (DELFROM-01a, DELTO-02a, INS-02b, DEL-the-01a) under
   pythia-70m, P=128/256, assert E/L/junk match the `GIBBS_INDEL_RESULTS.md` numbers within seed noise
   (full tail = bit-identical posterior; windowed = within tolerance). Logs go next to the existing
   `planning/bd_gibbs_*.log`.

The 31-test suite (`python -m pytest src/genjax_port/tests/`) must stay green.

---

## 6. Measurement (prove the win, size Stages)

- Wrap the indel sweep in the existing `rejuv_stats` accounting and add an indel-grid LM-forward / softmax-row
  counter so before/after is quantified, not assumed.
- Time `NC_REJUV=gibbs+bd` on 3 representative battery items (short/median/long M) before and after Stage 1,
  then Stage 2 at `bd_lookback=3`. Report wall-clock and the softmax-row counter.
- **Cost-probe first** (per repo `CLAUDE.md`): a single median-length item is enough to confirm the speedup
  sign and rough magnitude before any cluster sweep.

---

## 7. Scope, risks, non-goals

- **Scope:** `bd_mode="gibbs"` only (the deployed indel move). `mh`/`smcp3` modes call `score_fn` in
  `_del_logq`/`_ins_logq` and would benefit from the same tail trick, but they are out of scope here (note
  it as a follow-on).
- **Risk — channel still per-candidate.** This plan leaves the `channel_carry` per candidate (Cause 5,
  small). If profiling shows it now dominates after the LM drops, factor the channel onto the same
  shared-prefix structure next; not needed for the headline win.
- **Risk — prefill still per (particle, gap).** Stage 1/2 re-prefill each gap's prefix (`B=N·Wmax` rows).
  With the short seed (`prime="."`) this is cheap; a Stage 3 "prefill the no-op once per particle, rewind
  the KV `cache_end_index` per gap" (penzai already exposes `cache_end_index`,
  [lm_penzai.py:214](../src/genjax_port/lm_penzai.py#L214)) would remove it, but the tail feeds, not the
  prefills, dominate once the prime is short — so Stage 3 is a low-priority follow-on, not part of this plan.
- **Non-goal:** changing the move set, the candidate pool, `bd_attempts`, or particle count — those are
  Causes 2/4/6 with their own tradeoffs.
- **Compose with Cause 3.** When the per-item recompile fix lands (lru_cache + traced-arg factory for the
  indel sweep), `_indel_logits_tail` should be built inside that same memoized factory so it compiles once
  per structure. Keep the new helpers free of baked-in per-run constants (thread `cand_*`, `emit_full`,
  `tail_fn` as args) so they are ready for it.

## 8. Order of work

1. `_noop_lm_prefix` + factor channel out of `_make_bd_score_fn` into `_indel_channel_logits`; toy test 1.
2. `_post_edit_tail_inputs` (insert + delete); `_indel_logits_tail` (full tail); toy test 1 green.
3. Switch `make_gibbs_indel_sweep` to `_indel_logits_tail` (full tail); Pythia equivalence test 2; rerun
   the 31-suite + battery subset (test 5); measure (§6).
4. Stage 2: `bd_lookback`/`max_tail` knob + `NC_BD_LOOKBACK`; window-vs-full agreement check; battery subset.
5. Leave a one-line pointer in `GIBBS_INDEL_RESULTS.md` "Perf / caveats" recording that the Phase-2 KV
   suffix-tail win is now implemented, with the measured speedup.
