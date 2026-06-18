# Plan: the pair-HMM noisy-channel SMC on the real Pythia LM (channel-aware proposal, GenJAX traces, multi-token words, KV-extend cache)

**Audience:** a fresh agent picking this up cold in a new session.
**Status:** all the pieces are validated in isolation (three PoCs + two KV-cache spikes). This plan
**assembles them into one system** running on the real Pythia LM, with **GenJAX `@gen` traces from
day one** — not a hand-rolled filter that reimplements importance weights.
**Env:** the `ncgenjax` arm64 conda env (all default Pythons are x86/Rosetta; jax won't run on them).
Run modules as `python -m genjax_port.<mod>` from `src/`, or set `PYTHONPATH=src`. The LM is selected
by `NC_LM` (default `EleutherAI/pythia-70m` — keep it; it's ~6× cheaper than 410m and adequate).

---

## 0. Big picture (read this first)

The original Gen.jl noisy-channel model corrects a noisy sentence (typos, dropped/spurious words)
by reasoning about what the writer *intended*. The GenJAX port accreted complexity because it
**sampled the edit-alignment** between intended and observed strings with trans-dimensional moves —
exactly the latent that dynamic programming marginalizes exactly and in fixed shape.

The reframing (validated) is a **nested pair-HMM**:

```
            intended sentence (sampled, left-to-right, from the LM)   ← the ONLY sampled latent
                    │
                    ▼  for each intended-word step, one fixed-shape forward-DP row update:
  word-level pair-HMM:  align word↔observed word | word MISSING (deleted) | observed word SPURIOUS (inserted)
                    │        the word↔word "emission" cost IS …
                    ▼
  char-level pair-HMM:  copy/substitute/insert/delete CHARACTERS of the surface forms  → channel logpdf
```

Per particle we carry only fixed-shape state: the LM context buffer + the word-level forward vector
`alpha[k]` = log P(intended prefix, k observed words consumed). The alignment is **summed out by the
DP**, never sampled — so the whole "fighting JAX" problem (ragged shapes, K-bucketing, reversible
jump) dissolves and everything is `vmap`/`jit`-clean over particles.

Four further design choices complete the system:
1. **Channel-aware (fully-adapted) proposal.** Don't propose the next intended word from the LM
   prior (bootstrap, myopic). Propose from a small candidate set `C` scored by `LM × channel
   evidence`; the incremental importance weight is the candidate-set normalizer `logsumexp_C`,
   which has **near-zero variance** — the reason small particle counts work with an expensive LM.
2. **GenJAX `@gen` traces for every sampled choice.** The intended-sentence generative story and
   SMC weights live in an `@gen` step kernel called via `kernel.importance` — not a bare
   `jax.random.categorical` + manual `log_w` accumulation. The pair-HMM `alpha` vector is
   **Rao–Blackwellized carry state** (outside the trace); channel evidence enters via `factor` /
   `exact_density`. This is what makes rejuvenation, constraints, and weight auditing work later
   without another rewrite.
3. **Multi-token words.** A candidate intended "word" is a sequence of BPE tokens; its LM score is
   the chain-rule sum over those tokens given the shared context, and committing it appends those
   tokens. The char-level channel works on *surface forms*, so it is BPE-agnostic — this is why the
   M:N (multi-token-word) problem dissolves here.
4. **KV-extend cache.** The forward filter is strictly left-to-right, so the LM context grows
   **monotonically** — the append/extend KV-cache pattern, not the fork/rescore churn of the old
   port. The *same* prefix-KV mechanism also scores the `K` candidate word-tails over the shared
   current context at each proposal step. One mechanism, two uses.

**Key unification to exploit:** "score K candidate token-tails over a shared prefix" is *exactly*
what the rejuvenation prefix-KV-cache spike already validated. The channel-aware proposal's
multi-token candidate scoring is the same operation — reuse that scorer.

---

## 0.1 GenJAX integration (read this second — not optional)

The migration goal is to **use GenJAX's RV tracking where it makes sense**, not to reimplement
importance sampling by hand. The pair-HMM reframing makes the split explicit:

| Component | In `@gen` trace? | GenJAX mechanism |
|-----------|------------------|------------------|
| Next action (word / INSERT / EOS) | **Yes** | `genjax.categorical(scores) @ "action"` |
| LM token emissions (multi-token word) | **Yes** | `lm_token` / `make_word_model` copy-branch `"t0".."t{n-1}"` |
| Char-level channel likelihood | **No** (marginalized) | `channel_logpdf` as precomputed `EMIT` columns, or `exact_density` |
| Word-level alignment `alpha` | **No** (RB state) | Deterministic carry updated each step |
| Locally-optimal caprop weight | **Yes** (deterministic) | `factor(incr) @ "ev"` with `incr = logsumexp_C(scores)` |

### Reference implementations (copy these patterns)

**Char channel + one-shot posterior** — `poc_pairhmm_channel.py`:
- `edit_channel = exact_density(...)` wrapping the char DP
- Pure `@gen` model; built-in `Target` + `ImportanceK` reproduces the exact posterior

**Bootstrap word-SMC with `@gen` kernel** — `poc_word_indel.py`:
```python
@genjax.gen
def kernel(state):
    ctx_buf, ctx_len, log_alpha, done = state
    s = genjax.categorical(lm_logits(ctx_buf, ctx_len)) @ "s"
    ...
    _ = factor(incr) @ "ev"          # RB weight; alignment marginalized
    return (ctx_buf2, ctx_len2, new_alpha, done2)

tr, w = kernel.importance(key, constraint, (state,))
```

**Caprop proposal math** — `poc_word_indel_caprop.py` (`step_caprop`, candidate assembly, EOS
terminal accounting). Note: this file currently uses `jax.random.categorical` for speed of iteration
on the toy LM; **graduating to Pythia must restore the `@gen` + `importance` pattern** from
`poc_word_indel.py`, with caprop scores fed into `genjax.categorical` and `factor(logsumexp_C)`.

**Multi-token word emission template** — `genjax_model.py`:
- `make_word_model(n, n_sub)` — Switch over COPY (n tokens) / SUB (1 token) branches
- `factor` — deterministic log-weight injection
- `make_lm_scan_model` — autoregressive intended-sentence generator as a `@gen` Scan

### SMC outer loop (hybrid by design)

The outer loop stays hand-rolled for things GenJAX does not own:
- ESS-triggered resampling over particles
- Carrying `(ctx_buf, ctx_len, log_alpha, done)` between steps
- Batched Pythia forwards / KV cache / SymSpell candidate tables (inputs to the kernel)

Each step: `vmap(kernel.importance)` over particles → accumulate `w` from GenJAX, not re-derived
`log_w += incr` by hand. Optionally explore `genjax.inference.smc` over a `Scan` of the step
kernel once the step kernel is stable.

### What this is NOT

- Putting the DP alignment **inside** `@gen` as sampled latents (that was the old port's mistake).
- Deferring `@gen` until "after correctness" — correctness and GenJAX structure ship together in
  M-A. A hand-rolled loop that passes the toy tests but has no trace addresses is a dead end for
  rejuvenation and for the migration's stated goal (`planning/FRUSTRATIONS.md`).

---

## 0.2 Current state & stabilization (June 2026)

A recent implementation session merged M-A/M-B features prematurely into `pythia_word_caprop.py`
(multi-token tails, `batch_tail_logprobs`, INSERT gates, broken KV path). Results regressed:

| Case | Pre-churn (M-A only) | After churn |
|------|----------------------|-------------|
| Missing (`i want go home`) | **6/6** | failed |
| Substitution (`teh cat sat on teh mat`) | **5/6** | worse |
| Spurious (`the cat sat sat on the mat`) | **0/6** | **0/6** |
| Runtime @ P=256 | ~8–10s | ~60–100s+ |

**Phase 0 (stabilize)** before any new features:
1. Strip `pythia_word_caprop.py` back to **M-A only**: single-token words, one `next_token_logprobs`
   per step, no `batch_tail_logprobs`, no boolean INSERT gates.
2. Rebuild the step as an **`@gen` kernel + `kernel.importance`** (caprop proposal inside the
   kernel, `factor` for weights).
3. Do **not** call the broken KV path from the filter (`NC_USE_KV=0`; spikes in
   `planning/kv_cache_spikes/` remain valid — the new `lm_penzai` integration is wrong).
4. Gate: restore missing **6/6**, substitution **≥5/6**, runtime **≤15s @ P=256**, and every step
   goes through `kernel.importance` with a trace address `"action"`.

---

## 1. What is already validated (and where the code is)

PoCs (toy, char-level, toy bigram LM — all `python -m genjax_port.<mod>`):
- **`poc_pairhmm_channel.py`** — char-level edit channel as a forward DP exposed as a GenJAX
  `exact_density`. `channel_logpdf(observed_ids, intended_ids, n_x)`, `encode`, `L=12`, `PAD=0`.
  Pure-`@gen` model; built-in `Target`+`ImportanceK` reproduces the exact posterior. **Solid.**
- **`poc_word_smc.py`** — sentence-level bootstrap particle filter, one step per observed word; toy
  bigram `ToyLM.logits(ctx_buf, ctx_len) -> [V]` mimicking the heavy LM interface. **Solid.**
- **`poc_word_indel.py`** — full-word insert/delete via the nested pair-HMM. `_word_row_update`
  (the word-level row update; **reuse for `dZ`**), the `EMIT[k,w]` table, the RB-SMC loop with
  **`@gen` kernel + `factor`**, the `-inf - -inf = NaN` guard. **Spurious works; missing word is
  the contested case on the toy bigram.**
- **`poc_word_indel_caprop.py`** — the **channel-aware proposal** math and candidate assembly.
  Candidate set `C` = frontier-window emission candidates ∪ top-J LM ∪ INSERT ∪ EOS;
  `score(w)=log p(w|ctx)+dZ(w)`; propose `w~softmax_C`, weight `logsumexp_C`. LM is injectable
  (`lm_fn`); `proposal="caprop"|"bootstrap"` for A/B. **Read `step_caprop` and EOS accounting —
  port the math into an `@gen` kernel for Pythia.**

**Finding that reshapes the goal (do not re-litigate):** the missing-word case (`the cat on the
mat` → `the cat sat on the mat`) is **mostly a MODEL limit on the toy bigram, not inference
myopia**. Under a *bigram* the true posterior favours the verbatim 5-word parse. Restoration becomes
the true MAP only with an LM that **knows the sentence** — i.e. **the real Pythia LM**. What the
channel-aware proposal demonstrably buys is the fully-adapted filter's **low-variance weights** (logZ
std 2–7× smaller than bootstrap at equal P).

KV-cache spikes (validated 2026-06-16, `planning/kv_cache_spikes/`, run with `NC_LM=…python …`):
- **`kv_spike.py`** — penzai `KVCachingTransformerLM.from_uncached(...)` driven + forked via the
  functional `pz.unbind_variables(…, freeze=True)` / `pz.bind_variables(…, unfreeze_as_copy=True)`
  pattern; cached incremental logits match the full forward to ~4.5e-3 (float gap, not a bug);
  forking the prefix to score candidates adds **zero** extra error. **Gotcha: `pad_id=-1`, not 0**
  (0 is our `EOS_ID`/BOS seed; feed exactly `[0, i_len)`).
- **`kv_vmap_spike.py`** — the make-or-break: the stateful cache runs **under `jax.vmap`** with a
  **per-particle split point** via the **REWIND** trick (build single-sequence with `batch_axes={}`,
  vmap over particles so `cache_end_index` becomes per-particle; set `cache_end_index.value=posc`,
  feed a fixed-length tail). No K-way cache copy — rewind reuses the one prefix.

---

## 2. Target architecture (the unified forward loop)

State per particle (all fixed shape, `vmap` axis = particles `P`):
- `ctx_tokens [max_tokens]` int32 — committed intended-sentence BPE tokens, position 0 = `EOS_ID`
  seed (lm_penzai convention); `ctx_len` filled positions.
- `log_alpha [M+1]` float32 — word-level forward vector (M = #observed words).
- `done` bool — emitted EOS.
- (with the cache, M-C+) the prefix KV `vars` pytree for `ctx_tokens[:ctx_len]`.

### `@gen` step kernel (one SMC step)

Inside the kernel (mirrors `poc_word_indel_caprop.step_caprop` math):

1. **Assemble candidate set `C`** from scanned inputs (precomputed emission tables + LM logits).
   Emission candidates = edit-neighbours via `noise_word.word_sub_candidates`; fluency/deletion =
   top-J LM; INSERT slot; EOS. Dedup, pad to fixed `(K,)`.
2. **Score:** `dZ(w)` via `_word_row_update`; `lm_score(w)` from `next_token_logprobs` (M-A) or
   chain-rule tail score (M-B+). `scores = lm + dZ`, mask invalid to −inf.
3. **Propose:** `action ~ genjax.categorical(scores) @ "action"` where action indexes
   `{cand_0..cand_{K-1}, INSERT, EOS}`.
4. **Weight:** `_ = factor(incr) @ "ev"` with `incr = logsumexp(scores)` (locally-optimal;
   independent of which action was drawn). NaN-guard on `incr`.
5. **Advance:** update `log_alpha`, append chosen token(s) to `ctx`, set `done` on EOS. Return new
   carry `(ctx, ctx_len, log_alpha, done)`.

Outer loop: `vmap(kernel.importance)(keys, states)` → `log_w += w`; ESS resample when
`ESS < P/2`; terminal correction per EOS accounting in `poc_word_indel_caprop.run`.

### Terminal correction (subtle — get this right)

Folding the EOS terminal read into the EOS candidate's score makes the per-step `logsumexp_C`
telescope to the exact log-marginal, so EOS-in-`C` **replaces** the one-shot terminal correction
for particles that emitted EOS. Apply an end correction **only** to particles still live at the
step budget. (`poc_word_indel_caprop.run` already does exactly this; copy its accounting.)

Loop length `M + slack`; decode = most-frequent intended-sentence trajectory across the cloud.

---

## 3. Milestones (do them in order; correctness + GenJAX before speed)

### Phase 0 — Stabilize M-A on Pythia with `@gen` kernel

**Goal:** working uncached single-token filter that uses GenJAX traces.

Build / restore `pythia_word_caprop.py` from `poc_word_indel_caprop.py` + `poc_word_indel.py`
`@gen` pattern, swapping:
- `lm_fn` → `lm_penzai.next_token_logprobs(token_bufs, i_lens)` → `[P, vocab]`
- toy vocab → real tokenizer + char channel on surface forms (§4)
- emission candidates → `noise_word.word_sub_candidates` (single-token only: `max_word_tokens=1`)
- `segment_words(obs_ids)` instead of `split()` for observation units
- **`@gen` step kernel** with `genjax.categorical @ "action"` + `factor @ "ev"`; outer loop calls
  `kernel.importance`, not `jax.random.categorical` + manual weights

**Explicit non-goals for Phase 0:**
- No multi-token tails / `batch_tail_logprobs`
- No boolean INSERT gates (`allow_insert` heuristics)
- No KV cache in the hot path
- No rejuvenation stub

**Gate (all must pass):**
- Missing: `i want go home` → `i want to go home` — **6/6 @ P=256**
- Substitution: `teh cat sat on teh mat` — **≥5/6 @ P=256**
- Runtime: **≤15s @ P=256** on pythia-70m
- GenJAX: each step returns a trace with `"action"` address; weights from `importance`, not hand-sum
- caprop logZ std < bootstrap at equal P (reconfirm low-variance win)

### Phase 1 — Insertion (spurious observed words)

Add INSERT as an explicit action in the **same `@gen` kernel** (no heuristic gates). Score via
`_wins_only_row`; propose via the same `categorical` over `{words…, INSERT, EOS}`.

Work order: fix in `poc_word_indel_caprop.py` (toy, fast), then port to Pythia.

**Gate:**
- Toy: `the cat cat sat` → `the cat sat` MAP **≥ 0.6** @ modest P
- Pythia: `the cat sat sat on the mat` → `the cat sat on the mat` — **≥4/6 @ P=256**
- If toy gate fails after proper INSERT (no gates): write a short decision note — intended-word
  loop vs observed-word loop (see `smc_substitution.py` INSERT branch for reference semantics)

### Phase 2 — Multi-token intended words (M-B)

A candidate word is a BPE token sequence:
- **Candidate generation:** `tokenizer.encode` padded tails; edit-neighbours from lexicon surfaces
- **LM scoring:** chain-rule sum over word tokens — structure from `make_word_model` copy-branch
- **Inside `@gen`:** for a chosen word candidate, emit tokens as `lm_token @ "t{i}"` (or constrain
  via pre-scored tail + single `factor` if scoring is externalized for batching — but keep trace
  addresses for rejuvenation)
- **Channel unchanged** (surface forms). Commit = append all of word's tokens.

**Gate:** `threats` → `treats` (or similar M:N correction) recovers at modest P.

### Phase 3 — KV-extend cache (M-C)

Replace O(T²) full forwards with the cached path from spikes:
- **Commit/extend:** append-only after choosing action
- **Candidate scoring:** rewind scorer for K tails over shared prefix (`kv_vmap_spike.py` recipe)
- Factor into `lm_penzai` (`prefill`, `extend_logprobs` / `batch_tail_logprobs` with parity tests)

**Gate:** cached == uncached within **~5e-3**; Phase 0–2 outputs unchanged; measured speedup;
do not wire KV until parity proven (spikes are the source of truth, not the current broken integration).

### Phase 4 — Rejuvenation (M-D, optional)

Suffix-aware sub-flip enabled by trace addresses from Phase 0:
- `manual_subflip_move` / `Rejuvenate` edits `"action"` or `"t{i}"` at past addresses
- Same prefix-KV rewind scorer as Phase 3
- See `planning/REJUV_GOAL2_CONDITIONAL_PROPOSAL.md`, `REJUV_GOAL3_SMCP3_REWEIGHT.md`

---

## 4. Fixed regression suite

Run via `python -m genjax_port.pythia_word_caprop --selftest` (or dedicated tests in
`tests/test_pythia_word_caprop.py`). Document expected behaviour:

| Tag | Observed | Expected intended | Edit type |
|-----|----------|-------------------|-----------|
| DEL | `i want go home` | `i want to go home` | missing word |
| SUB | `teh cat sat on teh mat` | `the cat sat on the mat` | substitution |
| INS | `the cat sat sat on the mat` | `the cat sat on the mat` | spurious word |
| KEEP | `the cat sat on the mat` | (unchanged) | no edit |
| MULTI | `…threats…` | `…treats…` | multi-token sub (Phase 2) |

Report as **k/6** MAP hits @ P=256 unless noted otherwise.

---

## 5. Interfaces & conventions to honour

- **LM buffer convention** (`lm_penzai`): `token_bufs [P, max_tokens]`, position 0 = `EOS_ID` (=0,
  GPT-NeoX `<|endoftext|>`) seed; `i_lens [P]` filled positions; next-token logits read at
  `i_len-1`; padded positions hold `EOS_ID`. **Call `lm_penzai.load_model()` eagerly at startup**
  (else the penzai model builds inside a jit trace → `UnexpectedTracerError`).
- **Cache (M-C+):** `pad_id=-1` (NOT 0); feed exactly `[0, i_len)`; functional unbind/bind; build
  single-sequence + vmap over particles for a per-particle split point.
- **Channel:** bounded char alphabet on lowercase surface forms; `channel_logpdf` unchanged in
  structure. Knobs ↔ `config.SUB_PARAM`, `P_DELETE_PRIOR`; `WDEL/WINS` = word-level delete/insert
  log-priors.
- **Candidates:** `noise_word.word_sub_candidates`, `segment_words(obs_ids)`; frontier-localized,
  bounded K.
- **GenJAX:** `factor` from `genjax_model.py`; `constraint = ChoiceMap.d({"ev": 0.0})` when using
  `kernel.importance` with deterministic factors.

---

## 6. Code pointers (entry points)

- **`src/genjax_port/poc_word_indel.py`** — **`@gen` kernel + `factor` + `importance` pattern.
  Start here for GenJAX structure.
- **`src/genjax_port/poc_word_indel_caprop.py`** — caprop math (`step_caprop`, candidate assembly,
  EOS/terminal accounting, INSERT slot). Port math into `@gen` kernel.
- `src/genjax_port/poc_pairhmm_channel.py` — `channel_logpdf`, `encode` (char channel).
- `src/genjax_port/pythia_word_caprop.py` — **production target** (restore Phase 0 baseline here).
- `src/genjax_port/lm_penzai.py` — `load_model`, `next_token_logprobs`, `EOS_ID`. KV scorer added
  in M-C only after spike parity.
- `src/genjax_port/genjax_model.py` — `make_word_model`, `make_lm_scan_model`, `factor`, `obs_dist`.
- `src/genjax_port/noise_word.py` — `word_sub_candidates`, `segment_words`.
- `src/genjax_port/smc_substitution.py` — reference for per-obs-word INSERT/COPY/SUB semantics.
- `planning/kv_cache_spikes/` — validated KV recipes (do not break these).
- `planning/CHANNEL_AWARE_PROPOSAL_PLAN.md` — caprop predecessor plan.
- `planning/FRUSTRATIONS.md` — why `@gen` matters for this migration.

---

## 7. Definition of done

- **Phase 0:** Pythia M-A filter with **`@gen` step kernel + `importance`**, uncached, single-token,
  channel-aware. Passes DEL + SUB gates; ≤15s @ P=256; caprop logZ variance < bootstrap.
- **Phase 1:** INSERT without heuristic gates; passes INS gate on toy + Pythia.
- **Phase 2:** multi-token word recovered (MULTI gate); LM scored by chain-rule; trace addresses
  for emitted tokens.
- **Phase 3:** cached == uncached within ~5e-3; outputs unchanged; measured speedup ≈
  `T / max_word_tokens`, roughly K-independent.
- **Throughout:** LM behind `lm_penzai`, open-vocab candidates, bounded-K frontier-localized sets,
  fixed-shape `vmap`-clean particle state, **no hand-rolled importance weights** in the hot path.

---

## 8. Risks & gotchas (carry forward — these already bit us)

- `-inf - -inf = NaN` in `dZ`/weights for dead particles silently poisons categorical sampling.
  Guard every weight with `jnp.where(isnan, -inf)`.
- Cache `pad_id=-1` (0 is the BOS seed). Eager `load_model()` before any jit.
- Don't double-apply the terminal correction (EOS-in-C already applies it for particles that chose
  EOS; end correction is for live-only particles at budget).
- **Don't ship a hand-rolled filter** that passes correctness tests but lacks `@gen` traces — it
  blocks rejuvenation and repeats `FRUSTRATIONS.md`.
- **Don't merge phases** (multi-token + KV + INSERT gates) before Phase 0 gate passes.
- Tiny P + expensive LM: keep `proposal="caprop"`; bootstrap needs far more particles.
- INSERT boolean gates (`allow_insert` on `argmax(alpha)`) are whack-a-mole — forbidden in Phase 1+.
