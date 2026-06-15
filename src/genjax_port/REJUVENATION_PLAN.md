# Rejuvenation (MCMC reanalysis) — tentative plan

> Status: **draft for discussion**, no code yet. Goal is to add MCMC rejuvenation moves to the
> unified word-scan filter so the model can *revise earlier interpretations as later context
> arrives* (incremental reanalysis) and *re-diversify* a particle set that resampling has
> collapsed.

---

## 0. Architecture decision (pivotal — resolve before building)

Rejuvenation forces a decision we deferred when hand-rolling the forward filter: **migrate the
model to genjax `@gen` generative functions, or keep the hand-rolled JAX SMC and bolt on a manual
trace?** This gates everything below.

- **The hand-roll's cost is now due.** We discarded `@gen`'s automatic random-choice tracking and
  the functional scan interface. §5 (manual alignment trace) and §6/§8 (hand-rolled MH-ratio +
  reversible-jump math) are re-implementing exactly what `@gen` traces + `Rejuvenate` give for free.
- **genjax CAN express this model natively (verified):** `Scan`/`Vmap` combinators;
  `exact_density(sampler, logpdf)` to wrap the penzai LM as a custom categorical distribution
  (logits from penzai); `Importance`/`ImportanceK`/`ChangeTarget` SMC; native `Rejuvenate`
  (SMCP3). The in-graph LM is **not** a blocker (it becomes a custom distribution).
- **Tradeoff:** native → traces + `Rejuvenate` for free, functional scan, library-correct
  inference, much less custom code; **BUT** rewrite effort + performance unknowns: does the penzai
  forward stay *one batched call* under `vmap`+`Scan`? the prefix-dedup ~1.5–2× probably won't
  survive vmap's per-particle model; variable-length alignment via `Scan`+mask is fiddly.
- **Resolution = a de-risking spike** (see `/tmp/genjax_spike*.py`): wrap penzai as
  `exact_density`, a minimal substitution-only `@gen` `Scan` model, run genjax `Importance` SMC on
  one sentence; check (1) posterior matches the hand-rolled filter and (2) the penzai forward
  batches efficiently. Decide native-vs-handroll from the result.

**If NATIVE:** §5 and most of §6/§8 evaporate (addresses + `Rejuvenate` replace them).
**If HAND-ROLL:** proceed with §5 onward as written.

### Spike result (2026-06-14, `/tmp/genjax_spike.py`) — native path looks VIABLE

Retired the core integration risks (Pythia-70m):
- **penzai-as-`exact_density`-distribution inside a `@gen` `Scan` model: works** — `simulate`
  generated a valid sentence; **the trace auto-records per-step `tok` choices** (the §5 manual
  alignment trace is FREE with `@gen`).
- **`importance()` weight is correct**: log P(sentence) = −32.522, matches a direct LM chain-rule
  score exactly → genjax's inference math composes with the penzai LM.
- **`vmap` batches the penzai forward**: P=1 run 0.094s vs P=64 run 0.567s (~6×, not ~64×) — the
  feared per-particle blow-up did NOT happen; one batched forward per step.
- `Rejuvenate` imports; wiring an edit on a scan address is the next probe.

**Spike 2 (`/tmp/genjax_spike2.py`) — noise channel + rejuvenation move both work:**
- **Noisy channel expresses correctly:** a `@gen` model `x ~ LM; o ~ table-channel(x)` (candidates
  as args) — fully-constrained `importance` weight = manual joint log-density exactly (−11.431).
- **Native `Rejuvenate` performs reanalysis:** starting a trace at the LITERAL reading
  (`x0 = " too"`), a `StaticRequest({"x0": Rejuvenate(cand_proposal, arg_map)})` MH move **flipped
  it to the higher-posterior `" to"`** and stayed. This is the core reanalysis move working
  end-to-end on our model type (LM prior + edit-candidate channel), via the library.

**Still unproven (migration *engineering*, not blockers):** the **data-driven proposal for full
SMC** (`Importance(target, q)` needs a `SampleDistribution`/GenSP `q` over scan addresses —
supported, but more API surface); **scan-composition of `Rejuvenate`** (vectorized edit over scan
steps — the HMC scan test shows it's feasible); **variable-length deletions** via `Scan`+`Mask`;
and the prefix-**dedup** (~1.5–2×) likely lost under vmap (acceptable).

**Recommendation: go genjax-native.** Across both spikes, every fundamental risk cleared —
penzai-as-distribution, auto-traces, correct `importance`, batched vmap, noisy channel, and a
working `Rejuvenate` reanalysis flip. The native path gives the §5 trace and the §6/§8 MH math for
free. What remains is migration engineering (custom proposal, scan+mask for deletes), not
feasibility. **Next: plan the migration of the unified model to `@gen`**, starting from
substitution (proven) and layering deletes (mask) + the GenSP proposal.

---

## 1. Motivation

The forward filter commits, word by word, to an interpretation of each observed word, and
resampling repeatedly kills the losing alternatives. So an early word that looked fine in
isolation can never be reinterpreted once later context reveals it was a typo / omission. Example
classes:

- **Incremental disambiguation:** a word that's only mildly surprising at position *t* becomes
  clearly a typo given words *t+1…t+k* (e.g. a garden-path-like correction).
- **Degeneracy escape:** after many resamples, particles share a long identical prefix; the true
  reconstruction may have been pruned. Rejuvenation reopens earlier choices.

Important framing (consistent with prior session findings): rejuvenation **targets the same
posterior** — it doesn't manufacture new posterior mass, it (a) lets the chain *reach*
high-posterior reanalyses the greedy forward pass missed, and (b) reduces variance / re-diversifies.

---

## 2. How Gen.jl does it (the reference — `src/gen_inference.jl`)

Rejuvenation there is **involutive Metropolis–Hastings** moves applied to each particle's trace,
interleaved into the particle filter loop (`particle_filter_with_rejuv`, lines 346–489):

- **Trigger (conditional reanalysis, lines 407–417):** after each observation it computes the
  word's `surprisal = -log_mean_weight` vs its `unigram_surp`, maps the gap through a sigmoid to
  `cond_rejuv_p`, and rejuvenates a particle only with that probability — *high surprisal ⇒ high
  rejuvenation probability*. This is the "this word was unexpected, reconsider recent history" signal.
- **Scope (lines 419–422):** revisit timesteps in a `lookback` window `t-lookback … t`, in
  FORWARD / BACKWARD / shuffled order.
- **Moves (per revisited timestep `tt`):**
  - **Substitution flips** (`rejuv_proposal_{form,sem,morph}_sub` + `involution_sub`, 286–344):
    propose a new intended word at the position aligned to `tt` from the top-k substitution
    candidates (reweighted by LM), and flip the action `normal ↔ {form,sem,morph}_sub`. The
    involution just swaps word + action; fixed length.
  - **Add/Delete** (`rejuv_proposal_add_delete` + `involution_add_delete`, 193–284): a
    reversible-jump move that **inserts or deletes an intended word** at the aligned position
    (coin flip `:add`), regenerates the intended-sentence suffix from the LM and the noisy-sent
    alignment, and the involution maps `add ↔ !add` for reversibility. This is the
    variable-length move.
  - **Parameter moves (478–486):** plain `Gen.mh(select(:action_prior))` and
    `select(:form_sub_param,:sem_sub_param))` — resample global params.
- Each move returns `(new_trace, accepted)`; accept/reject is standard MH.

Key enabler: Gen.jl keeps a full **trace** (`:intended_sent=>i=>:w`, `:noisy_sent=>tt=>:action`,
`:idx`, params), so a move can address "the intended word aligned to observed word `tt`" and
`Gen.update` re-scores the whole model consistently.

---

## 3. How genjax supports rejuvenation natively — and why we can't drop it in

genjax *does* have first-class rejuvenation:

- **`Rejuvenate` EditRequest** (`genjax/_src/inference/requests/rejuvenate.py`): an SMCP3 move on
  a **`Trace`**. Given a proposal generative function + an `argument_mapping`, it does
  `proposal.propose → request.edit(Update) → proposal.assess`, returning weight
  `w + bwd_score − fwd_score` (the MH ratio, *without* accept/reject — you fold the weight into
  the particle, or do accept/reject yourself).
- Usage pattern (from `tests/inference/test_requests.py:168–193`): wrap addresses in a
  `StaticRequest({addr: Rejuvenate(proposal, arg_map)})`, call `request.edit(key, tr, ())`, then
  MH-accept with `check = log(uniform) < w; tr = where(check, new_tr, tr)`.
- `SMCAlgorithm` / `ChangeTarget` (`genjax/_src/inference/smc.py`) compose these into SMC.

**Blocker:** all of this operates on genjax **traces of `@gen` generative functions** with
addressed random choices. **Our port has none of that** — it's a hand-rolled JAX SMC over a
penzai LM with manual particle buffers (`intended_buf`, `i_len`, `log_action_prior`) and
hand-computed weights. There is no trace, no choicemap, no `@gen` model. We went hand-rolled
deliberately (penzai LM in-graph + `vmap` over particles + dedup performance); representing the
variable-length BPE alignment as a genjax `@gen` model with a penzai LM inside is a large,
unproven rewrite. So we **borrow the genjax/Gen.jl math (MH / SMCP3 ratio) but implement the
moves by hand**, exactly as we did for the forward SMC.

---

## 4. Recommended approach: hand-rolled MH rejuvenation on particle state

Mirror Gen.jl's involutive MH, but on our particle buffers, vmapped over particles. For a chosen
revisit position, propose a local change, compute the MH acceptance ratio from our existing LM +
noise scorers, and accept/reject per particle (`jnp.where`-select, like the genjax test). Two
flavors to decide between (§12): **MH accept/reject** (Gen.jl style; leaves particle weights
untouched — simplest after resample) vs **SMCP3 reweighting** (genjax `Rejuvenate` style; no
accept/reject, fold weight into the particle). Default recommendation: **MH accept/reject**.

---

## 5. Prerequisite (the big one): record a per-particle alignment trace

Our unified filter currently **discards** the information rejuvenation needs. It keeps only the
running `intended_buf` / `i_len`; the per-word decision (which action, which intended tokens it
emitted, where) is used to emit and thrown away. To rejuvenate "the intended word aligned to
observed word `tt`" we must add a per-particle, per-observed-word **decision record**:

- action taken (copy / sub / insert / delete),
- the intended token span emitted (start index + length in `intended_buf`),
- enough to reconstruct the alignment (observed-word → intended-span map).

This is the analog of Gen.jl's trace. It's a fixed-size per-particle array indexed by observed
word (we know `W` up front), so it stays `vmap`-friendly. **This is the first build step and a
precondition for everything below.**

---

## 6. The rejuvenation moves (port of Gen.jl's two move families)

1. **Substitution flip (fixed length — do first):** at revisit word `tt`, propose copy↔sub or
   swap among the SymSpell candidates for that word; recompute the joint and MH-accept. No length
   change → no buffer reshaping. Reuses `noise_word.word_sub_candidates` and our LM scorer.
2. **Add/Delete (variable length — harder):** insert an omitted intended word, or delete a
   posited one, at `tt`. Changes `i_len` and shifts the suffix → ragged per-particle buffer
   rewrite. This is the reversible-jump move; reversibility via the `add ↔ !add` symmetry as in
   `involution_add_delete`.
3. **(Optional) parameter moves:** MH on `log_action_prior` / `SUB_PARAM` analog, like Gen.jl.

---

## 7. Triggering & scheduling

Port Gen.jl's **surprisal-conditioned** trigger: after each word, compare its incremental
surprisal to a baseline (unigram or running mean); high surprisal ⇒ higher probability of running
a rejuvenation sweep over a `lookback` window of recent words. Knobs: `lookback`, sweep order,
number of MH steps per position, and the trigger sigmoid (`center`/`spread`). Cheapest useful
default: a small fixed lookback, rejuvenate every step (unconditional) first to validate, then
add the surprisal gate.

---

## 8. Hard parts / risks

- **LM suffix re-scoring (the core cost).** The LM is causal, so changing the intended word at
  position *k* changes `LM(token | context)` for **all** later positions. The MH ratio's LM term
  is `LM(new suffix)/LM(old suffix)` from *k* to the current frontier *N* — the prefix cancels.
  Cost ∝ (N−k); the `lookback` window bounds it (k ≥ N−lookback ⇒ suffix ≤ lookback). The whole
  point of reanalysis is to let later context (up to *N*) vote on word *k*, so we *must* re-score
  that suffix — `lookback` is the reach/cost knob. Dedup + bounded windows make this tractable.
- **Variable-length moves** (add/delete) require ragged per-particle buffer shifts and `i_len`
  bookkeeping, plus correct reverse-move probabilities for the MH ratio.
- **`vmap` accept/reject**: per-particle `where`-select of (buffer, i_len, trace record), like
  `tree_map(where)` in the genjax test — straightforward but must select *all* particle state
  atomically.
- **Correctness of the MH ratio** (forward/reverse proposal probabilities + joint ratio) is the
  easiest thing to get subtly wrong; we should unit-test detailed balance on a tiny example.

---

## 9. Integration into the unified filter

Add a **rejuvenation phase** after each word's resample step in `particle_filter_unified`
(guarded by a `rejuvenate=False` flag so the forward filter is unchanged by default):
`… emit word → resample → [if triggered: for tt in lookback window: MH move(s) per particle] →`
next word. Keep moves behind the same injectable-LM seam so dedup applies.

---

## 10. Validation

- **Detailed-balance unit test** for each move on a 2–3 word toy (MH leaves the target invariant).
- **Reanalysis behavior test:** a sentence where an early word is only disambiguated later — show
  the forward-only filter keeps the wrong early reading while rejuvenation flips it. Candidate:
  a typo whose correction is implausible locally but favored once the following words arrive.
- **Degeneracy test:** a case where the correct reconstruction is pruned by resampling at low P;
  show rejuvenation recovers it (higher recovery rate / ESS) where forward-only fails.
- Re-confirm no regression on the existing sub/deletion/insertion/clean suite.

---

## 11. Phasing (direction confirmed)

- **R0** — *architecture decision from the §0 spike*, then (if hand-roll) the per-particle
  alignment/decision trace (§5). Precondition.
- **R1** — **substitution-flip move + unconditional sweep** over a (customizable) lookback window;
  detailed-balance test; reanalysis test. Fixed length — lowest risk, highest signal. **[v1]**
- **R2** — **add/delete reversible-jump move** (§6.2). Variable length. **[confirmed wanted]**
- **R3** — **surprisal-conditioned trigger** + tunable `lookback` distance (§7). **[confirmed wanted]**
- **R4** — (optional) parameter moves; SMCP3 vs MH tuning.

`lookback` is a knob from R1 onward (default small); the surprisal *gate* lands in R3.

---

## 12. Open questions

1. **Architecture: genjax-native vs hand-roll (§0)** — *decide from the de-risking spike.* PIVOTAL.
2. *(resolved)* **MH vs SMCP3:** MH accept/reject as the default for the fixed-length
   substitution-flip (clean fit with resample-every-step); revisit SMCP3 for the add/delete move
   where acceptance is harder. If we go genjax-native, `Rejuvenate` provides both from one primitive.
3. *(resolved)* **Move set:** substitution-flip for v1; **add/delete is confirmed for R2.**
4. *(resolved)* **Trigger:** start **unconditional**; surprisal gate + customizable `lookback` in R3.
5. **Reach vs cost:** practical max `lookback`? Is full-history reanalysis ever needed, or is a
   small window enough? (Affects LM suffix re-scoring cost, §8.)
6. **Should the trace also expose per-word action posteriors** (like Gen.jl's logs) — worth
   designing in now? (Free if genjax-native.)
7. **Word vs token granularity** for moves — word level (matching the unified scan)? (Lean yes.)
