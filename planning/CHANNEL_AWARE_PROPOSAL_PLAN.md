# Plan: channel-aware proposal for the pair-HMM noisy-channel SMC

**Audience:** a fresh agent picking this up cold in a new session.
**Status:** the reframing (marginalize the alignment with a DP; pure-`@gen` model; built-in SMC)
is validated through three proof-of-concept scripts. This plan fixes the one remaining gap.
**Env:** the `ncgenjax` arm64 conda env (all default Pythons are x86/Rosetta and jax won't run).
Run scripts as e.g. `python -m genjax_port.poc_word_indel` from `src/`.

---

## 1. Where we are (read this first)

The original Gen.jl noisy-channel model was being ported to GenJAX. The port accreted complexity
(trans-dimensional rejuvenation, K-bucketing, KV-cache thrash, hand-rolled MH weights) because it
**sampled the alignment** between intended and observed strings — exactly the latent that dynamic
programming marginalizes exactly and in fixed shape. The reframing: don't sample the alignment,
**marginalize it with a forward DP (pair-HMM)**, sample only the intended sentence left-to-right
from the LM, and let GenJAX's built-in SMC do the rest.

Three PoCs validate this (all self-contained, char-level, toy LM, run in seconds):

- **`src/genjax_port/poc_pairhmm_channel.py`** — char-level edit channel as a forward DP, exposed
  as a GenJAX `exact_density` (`edit_channel`, `channel_logpdf`). Pure-`@gen` model; built-in
  `Target` + `smc.ImportanceK` reproduces the exact enumerated posterior. **Solid.**
- **`src/genjax_port/poc_word_smc.py`** — sentence-level bootstrap particle filter, one step per
  observed word, with a toy bigram LM (`ToyLM.logits(ctx_buf, ctx_len) -> [V]`) that mimics the
  heavy model's interface. Per-particle state is fixed-shape and `vmap`-clean. **Solid.** Recovers
  truth; resampling demonstrably load-bearing (min ESS ~0.12/particle vs ~0.002 without).
- **`src/genjax_port/poc_word_indel.py`** — full-word insertions/deletions via a **nested**
  pair-HMM (the same row-update lifted to the sentence: align word = substitute, missing word =
  delete, spurious word = insert; the word-level "emission" cost IS the char-level DP score). Each
  particle carries the word-level forward vector `alpha[k]` (= log P(prefix, k observed words
  consumed)) as fixed-shape state. RB-SMC marginalizes the word alignment. **Partial — this plan
  finishes it.**

### The gap (precise)

In `poc_word_indel.py`:
- ✅ indel-disabled reduces to correct 1:1 length (regression check vs PoC #2).
- ✅ **spurious** word works: `the cat cat sat` → `the cat sat` (p≈0.32, decisive).
- ❌ **missing** word does not resolve: `the cat on the mat` ↛ `the cat sat on the mat`.

The missing-word failure is **inference, not model.** By hand, the true posterior favors the
correct 6-word answer over the junk parses by ~3 nats (one deletion at ~−2.3 beats the competing
all-insertion parses, and the LM strongly prefers the real sentence). The SMC just doesn't find it.

**Why:** the proposal is bootstrap — it samples the next intended word from the LM prior **blind to
the observed string.** To infer a deletion, a particle must (a) happen to sample the soon-to-be-
deleted word `sat` from the LM, then (b) have the DP delete it, paying the deletion cost *now* while
the payoff (clean alignment of `on the mat` afterward) only arrives at later steps. Resampling prunes
that particle before the payoff. ESS-triggered resampling (`_ess(log_w) < 0.5*P`) helped the spurious
case but not this. Brute-force particle counts is treating the symptom; do not go there.

**Gotcha already fixed (keep it):** `-inf - -inf = NaN` in the forward-mass increment `dZ` for dead
particles silently poisons `jax.random.categorical` (it samples NaN-weighted junk). Guarded with
`jnp.where(jnp.isnan(...), -jnp.inf, ...)` in both the kernel and the terminal weight.

---

## 2. The fix: a locally-optimal (fully-adapted) channel-aware proposal

Replace "propose the next intended word from the LM prior" with "propose it from a small candidate
set, scored by **LM × channel evidence**, using the locally-optimal proposal weight." This is the
fully-adapted particle filter — the standard cure for exactly this kind of myopia, and it slashes
the particle count needed.

### The math (why it works)

At each intended-word step a particle holds context (LM state) and the alignment vector `alpha`.
For a candidate next intended word `w`, the **incremental forward-mass change is cheap to compute**:

    dZ(w) = logsumexp(row_update(alpha, EMIT[:, w])) - logsumexp(alpha)     # one length-M row update

(`row_update` and `EMIT` already exist in `poc_word_indel.py` as `_word_row_update` and the
per-sentence `EMIT[k, w]` table.) For the EOS candidate, its "dZ" is the **terminal correction**
`alpha[M] - logsumexp(alpha)` (the evidence of stopping now with full consumption).

Build a candidate set `C` (fixed size K, padded), then:

    proposal_logit(w) = LM_logit(w | context) + dZ(w)        # for each w in C, plus EOS
    sample   w* ~ Categorical(softmax(proposal_logit))
    weight  += logsumexp_{w in C}(proposal_logit(w))         # <-- independent of which w* was drawn

That weight is the locally-optimal proposal's normalizer. Because it marginalizes the candidate
choice, the **incremental importance weight has near-zero variance** — particles stop getting
randomly pruned for a locally-unlucky draw, which is precisely the myopia that kills the deletion
hypothesis. The correct-but-locally-costly `sat`-deletion survives because (i) `sat` is in `C` (it's
a top-LM word given `cat`), and (ii) the low-variance weight doesn't punish the particle that drew
it. Note: truncating to `C` introduces bias only for intended words that are simultaneously
LM-implausible AND channel-incompatible with the frontier — negligible with a sane candidate set.

### The candidate set `C` (this is where "channel-aware" lives)

Per particle, the union of:
1. **Emission candidates** — intended words edit-close to the observed words at the **alignment
   frontier** (the `k` where `alpha` has mass). These are the channel-compatible words. Reuse
   `noise_word.word_sub_candidates(word_str, max_dist=2)` (edit-distance neighbors via a SymSpell
   index) — it already returns exactly this.
2. **Deletion / fluency candidates** — the **top-J LM words** given context (`jax.lax.top_k` on the
   LM logits). These cover missing words: a fluent word that bridges the context but doesn't match
   any observed word, which the DP will mark as deleted.
3. **EOS.**

Dedup and pad to fixed K. Keep `C` **frontier-localized** (candidates for observed positions in a
window around `argmax(alpha)`), so K stays bounded regardless of sentence length — important for
graduating to long real sentences. For the toy PoC a whole-sentence union is fine to start.

---

## 3. Implementation steps (do these in order, on the toy LM first)

Work in `src/genjax_port/poc_word_indel.py` (copy to `poc_word_indel_caprop.py` if you want to keep
the bootstrap baseline for comparison). Keep the toy bigram LM for fast iteration.

1. **Candidate table per sentence.** Precompute, for each observed word position `k`, a fixed-width
   list of emission-candidate intended-word ids (edit-neighbors). In the toy, the vocab is tiny so
   you can compute edit distance against `VOCAB` directly; structure it so swapping in
   `noise_word.word_sub_candidates` later is a drop-in. Shape: `(M, K_emit)` padded.

2. **Per-particle candidate assembly.** Given a particle's `alpha`, pick the frontier window, gather
   that window's emission candidates, union with top-J LM ids (`top_k(lm_logits, J)`) and EOS, dedup
   to a fixed `(K,)` id buffer with a validity mask.

3. **Score candidates.** For the `(K,)` candidates compute `LM_logit(w)` (gather from the LM logits;
   for the toy that's `lm_logits[w]`) and `dZ(w)` (vmap `_word_row_update` over the K candidates;
   each is O(M)). EOS slot uses the terminal `alpha[M]-Z`. Mask invalid slots to `-inf`.

4. **Propose + weight.** `w* ~ Categorical(LM+dZ over C)`; `incr = logsumexp(LM+dZ over C)`. Replace
   the kernel's `genjax.categorical(lm_logits) @ "s"` + bootstrap `dZ` factor with this. You can keep
   the GenJAX-native flavor by sampling `w*` index via `genjax.categorical` over the candidate logits
   and injecting `incr` through the existing `factor` (the deterministic-log-weight `exact_density`).

5. **Advance.** Update `alpha` with the chosen `w*` (existing `_word_row_update`); append `w*`'s token
   to the context. EOS sets `done`. Everything else (ESS-triggered resampling, one-shot terminal
   `alpha[M]` correction, NaN guards, decode) stays.

6. **Validate.** The **missing-word** case must now resolve at a *modest* particle count (target:
   `the cat on the mat` → `the cat sat on the mat` as MAP at P≈2–4k, where bootstrap failed at
   16k+). Re-confirm the spurious case and the indel-disabled regression check still pass. As a
   stronger check, the SMC's log-marginal-likelihood estimate should be stable across seeds and
   roughly invariant to P (a fully-adapted filter has low-variance weights).

---

## 4. Keep it general — graduating to Pythia later

Do not bake in toy-LM assumptions. The design above is LM-interface-agnostic; the three things that
change when `ToyLM.logits` becomes a Pythia/penzai forward:

- **Multi-token words.** A candidate intended word is a sequence of BPE tokens, so
  `LM_logit(w | context)` becomes the **chain-rule sum** over the word's tokens, and appending `w`
  to the context appends its BPE tokens. This is the monotonic, left-to-right **KV-cache extend**
  pattern (append + extend), not the fork/rescore churn of the old port — see the
  `rejuv-prefix-kv-cache-spike` memory and `lm_penzai.py`. The existing word-span machinery in
  `genjax_model.py` (`make_word_model`, the copy branch emitting `n` tokens) is the template.
- **Char-level channel is already BPE-agnostic.** `channel_logpdf` (PoC #1) aligns *characters* of
  the observed and candidate *surface forms*, so multi-token-ness never touches the channel. This is
  the whole reason the M:N problem dissolves here.
- **Open-vocab candidate generation.** No fixed dictionary. Emission candidates come from a lexicon's
  edit-neighborhood (`noise_word.word_sub_candidates` + `_symspell_index`); deletion/fluency
  candidates from the LM's `top_k`. Keep `K` bounded and the candidate assembly frontier-localized.

Bounded-K candidate scoring is also what keeps the Pythia LM cost in check: O(K) token-gathers per
step over a shared prefix, not a full-vocab sweep.

---

## 5. Code pointers

- `src/genjax_port/poc_pairhmm_channel.py` — `channel_logpdf`, `edit_channel`, `encode`, `L`, `PAD`.
  The char-level DP (the per-word "emission" cost). Reuse as-is.
- `src/genjax_port/poc_word_smc.py` — `ToyLM` (Pythia-interface bigram), `VOCAB`/`VOCAB_IDS`/
  `VOCAB_LEN`, `CORPUS`, `LCTX`, the bootstrap word-SMC loop. The LM and harness to extend.
- `src/genjax_port/poc_word_indel.py` — `_word_row_update` (the nested row update; reuse for `dZ`),
  the `EMIT[k,w]` table construction, the `kernel`/`run`/`decode` RB-SMC loop. **Edit here.**
- `src/genjax_port/noise_word.py` — `word_sub_candidates`, `_symspell_index`, `_damerau_levenshtein`,
  `segment_words`. The real edit-neighborhood candidate generator for the Pythia graduation.
- `src/genjax_port/config.py` — `MAX_SUB_CANDIDATES`, `LOOKAHEAD_K`, `P_DELETE_PRIOR/PROPOSAL`,
  `MAX_DELETIONS`. Tuned knobs from the old port; reuse the priors, retire the trans-dim ones.
- `src/genjax_port/genjax_model.py` — `factor` (deterministic-log-weight `exact_density`),
  `make_word_model`/copy-branch (multi-token word emission template for Pythia).
- `src/genjax_port/proposal.py` — existing `propose(key, log_ev)`; the data-driven proposal this
  generalizes. Worth reading before writing the candidate scorer.

## 5b. Revising earlier words from later evidence (rejuvenation, reframed)

Later context should be able to flip an earlier word (e.g. "All the **threats** made my dog
overweight" → "treats", obvious only at "overweight"). The forward filter already does this
**natively**: when the disambiguating words arrive, the predictive LM up-weights the particles
carrying "treats", and resampling flips the posterior — *provided those particles survived*. The
real risk is **path degeneracy**: aggressive resampling collapses early-word diversity, and
forward-only SMC cannot resurrect a hypothesis no particle holds.

Fixes, in order of cost — none trans-dimensional:
1. **Keep diversity (cheapest):** ESS-triggered (gentle) resampling + bounded **lookahead** in the
   proposal (peek L observed words ahead before committing the current one). Handles any dependency
   within the lookahead window; the channel-aware proposal above already widens early diversity.
2. **Word-level rejuvenation:** re-sample an earlier word-slot's *identity*, look up its precomputed
   tokenization into the padded buffer, re-score the affected LM suffix. Token-count change is
   absorbed by the SAME padding as the forward pass — fixed-shape, vmappable, **not** reversible-jump.
   This is the already-prototyped suffix-aware sub-flip (Gibbs, accept≈1); tractable now because we
   flip a WORD with the alignment MARGINALIZED, not tokens with the alignment sampled.
3. **Particle smoothing (heavy):** backward-simulation/FFBS — only if 1–2 fall short.

Cost that doesn't vanish: a mid-sequence flip invalidates the KV cache from that point (≈ re-run LM
from there). So **surprisal-gate** it (reuse the unigram-relative gate) — rejuvenate only words the
downstream context flags as suspicious, not every slot. Not yet spiked; do it AFTER the proposal.

## 6. Definition of done

- Missing-word case resolves to the correct longer intended sentence as MAP at P≈2–4k on the toy LM.
- Spurious-word and indel-disabled checks still pass.
- SMC log-marginal-likelihood estimate stable across seeds / roughly P-invariant (low-variance weights).
- No toy-LM assumptions hard-coded that would block the Pythia swap (multi-token LM scoring, open-vocab
  candidate generation, bounded-K frontier-localized candidate sets).

## 7. Result (2026-06-17) — implemented in `src/genjax_port/poc_word_indel_caprop.py`

The fully-adapted proposal is implemented exactly as specified (frontier-windowed emission
candidates ∪ top-J LM ∪ EOS, deduped/padded; `score(w)=log p(w|ctx)+dZ(w)`; propose
`w~softmax_C`, weight `logsumexp_C`). One math refinement: folding the EOS terminal read
`alpha[M]-Z` in as the EOS candidate's `dZ` makes the per-step `logsumexp_C`'s telescope to the
exact log-marginal, so the EOS candidate **replaces** the one-shot terminal correction (don't also
add it at the end; only un-stopped particles need an end correction). LM is injectable (`lm_fn`),
candidate gen structured to swap in `noise_word.word_sub_candidates` — Pythia path unblocked.

**§6 bullet 1 was based on a false premise (corrected here).** Brute-forcing the exact joint shows
that under the flat toy **bigram** the TRUE posterior favours the *verbatim* 5-word parse
('the cat on the mat', −11.7) over the 'correct' restoration ('the cat sat on the mat', −14.3) by
~2.6 nats: a deletion multiplies in an extra LM factor (<1) AND pays `wdel`, and a bigram cannot
see that 'cat on' is odd. So a correct sampler *must* return the verbatim parse — caprop does, and
the earlier bootstrap 'failure' was largely seed variance (both proposals roughly track the true,
verbatim-favouring posterior). Restoration is the true MAP only under an LM that knows the sentence
(demo uses a peaked bigram as a Pythia stand-in); there **both** proposals recover it — once the
deletion word is itself LM-likely there is no myopia to cure.

**What caprop demonstrably delivers (§6 bullet 3):** the fully-adapted filter's low-variance
weights. logZ std over 8 seeds at P=2000 is 2–7× smaller than the bootstrap's (spurious/flat 0.14
vs 0.28; missing/peaked 0.014 vs 0.098); means agree up to a small (~0.18 nat) candidate-truncation
bias. That variance edge — not a missing-word "fix" — is the real reason to carry this to Pythia,
where a wasted particle (a full transformer forward) is far more expensive. §6 bullets 2 and 4 hold.
