# Plan: rejuvenation + efficient KV-caching in the unified pair-HMM paradigm (redesign)

**Supersedes** the `REJUV_GOAL{1,2,3}` docs and the archived `rejuv_bridge` / `rejuvenation*` /
`pythia_rejuv` stack. Those were written for the **sampled-alignment, multi-token** paradigm, which
we have since left. This is a from-scratch design for adding rejuvenation to the **certified
RB-SMC pair-HMM filter** (`pairhmm_smc.py`, `pythia_word_caprop.py`), reusing only the *ideas* from
the goal docs (full-conditional proposal, SMCP3 reweight, prefix-KV fork, dedup/compaction).

**Env / rules:** `ncgenjax` arm64 conda env; `python -m genjax_port.<mod>` from `src/`
(`PYTHONPATH=src`). Toy-first correctness (no LM load, large P, ms). Pythia runs are smoke tests.
Flushed per-step progress to a file, never pipe expensive output. (Carry the project's standing rules.)

---

## 0. Why this is tractable now (the reframing the redesign turns on)

The archived rejuvenation stack broke for four reasons (`planning/FRUSTRATIONS.md`):
raggedness of particle latents, non-jittable/vmappable moves, runtime ballooning, and re-deriving
GenJAX machinery by hand. **Three of the four are artifacts of the old paradigm, not of
rejuvenation itself.** The current filter removes them by construction:

| Old-paradigm problem | Why it existed | Status in the new paradigm |
|---|---|---|
| Ragged latents | multi-token words + a flat buffer → a flip **resized** the buffer | **Contained, not assumed away.** The move's addressable unit is a **word = a bounded token span** (capacity `T_max`); a flip replaces a word's span and re-packs by cumsum-scatter — fixed-shape for *any* `T_max`. Single-token is just `T_max = 1`, the first instance we run, **not** a commitment (see §0.1). |
| Trans-dimensional / non-vmappable moves | the **alignment was a latent** (add/delete alignment cells) | **Gone.** The alignment is **Rao-Blackwellized** (`log_alpha` forward DP). A move only touches the intended-word tokens; `log_alpha` recomputes deterministically, fixed-shape. |
| Hand-rolled MH/SMCP3 ratios | no trace to `edit` | **Addressable.** genjax `Rejuvenate` is an SMCP3 move (weight `w + bwd − fwd`); our @gen step kernel already has the `"action"`/`"ev"` addresses to drive it. |
| Runtime balloon | K-candidate × full LM forward, no prefix reuse, over all P | **The one real residual.** This is what the KV-cache + gating/compaction phase targets. |

So in this paradigm **rejuvenation = a Gibbs/SMCP3 resample of a *word* (a bounded token span) from
a fixed-size candidate set, conditioned on the rest of the fixed-shape buffer**, with the channel
marginal from the existing forward DP and the weight from genjax. The hard parts that sank the old
attempt (alignment trans-dimensionality, *unbounded* raggedness) are absent; the remaining work is
clean wiring + one focused perf phase.

## 0.1 Multi-token is a capacity parameter, not a future redesign

We will eventually need multi-token intended words (single-token is a real handicap), so **no design
decision here may assume single-token permanently.** It does not have to: the DP paradigm already
made words sequences, not random variables, so two of the model's three layers are token-count-
agnostic *today*, and the third is contained by the representation choice above.

- **Channel (`channel_logpdf`) — already agnostic.** It aligns *character ids of surface forms*. A
  multi-token intended word is just a longer surface string; the channel never sees token boundaries.
- **Word-level forward DP (`_word_row_update`) — already agnostic.** One row per intended *word*,
  emission = `channel(observed surface | intended surface)`. 1 vs 3 tokens adds **no** DP dimension.
- **Where single-token actually lives (all incidental, none in the DP):** (1) candidate generation
  (`_candidate_ids` / SymSpell index only single word-initial tokens — the M:N case in
  `noise_word.py`); (2) LM scoring (a step scores `P(token|prefix)`; a multi-token word needs the
  chain-rule product over its tokens); (3) buffer layout (a flat buffer makes a *splice* variable-
  length).

**Design commitments that keep multi-token open:**
- The forward filter stays **flat / append-only** and is already token-agnostic (a step may append
  `k` tokens). Multi-token *forward* support = chain-rule LM scoring within a step + a candidate
  generator that yields multi-token surfaces. That is **Phase D of `PAIRHMM_RBSMC_PLAN.md`**, not
  this plan — but nothing here blocks it.
- The rejuvenation move is written **capacity-parametric** from R1: it addresses **words** via a
  `word → token-span` index and replaces a word's span with a candidate's tokens under a fixed
  splice capacity `T_max`, re-packing the flat buffer by cumsum-scatter (fixed-shape, vmappable).
  We *run* it at `T_max = 1` until Phase D, but the interfaces (candidate = token span, emission =
  surface, weight = chain-rule) never assume a single token. `T_max = 1` must be a config value,
  never a hard-coded `[:, i].set(tok)` shortcut.

The genuinely-new work multi-token adds (real, but not a rewrite): an M:N candidate generator,
chain-rule LM scoring of a candidate word, and the variable-length re-pack — all fixed-shape with a
bounded `T_max`, and the KV-cache becomes *more* valuable (more tokens to amortize), not less.

This is also the fix for the failure mode diagnosed in `planning/kv_cache_spikes/` today:
`The cat sat on mat.` collapses (ESS → ~3.6 at step 2, ~4 ancestors) onto `The car sat on mat.`,
even though the model's MAP is to leave it unchanged. A post-resample Gibbs sweep over recent words
can flip `car → cat` back (`cat` has the higher full-conditional) — the principled cure for
impoverishment that more particles only delay.

---

## 1. What "use genjax-native rejuvenation" means here

genjax `Rejuvenate.edit` (`genjax/_src/inference/requests/rejuvenate.py`):
1. propose a change from a user proposal `@gen` fn (`fwd_proposal_score`),
2. apply it as an `Update` to the trace (`w` = the model reweight),
3. score the reverse proposal (`bwd_proposal_score`),
4. return SMCP3 weight `w + bwd − fwd` and the new trace — **no accept/reject coin**.

For a full-conditional ("Gibbs") proposal this weight is ≈ 0 (REJUV_GOAL2/3); for asymmetric moves
it carries real mass into the next resample (REJUV_GOAL3 benefit (b)). Either way **we stop deriving
the ratio by hand** — genjax owns it. Two integration levels:

- **Level A — genjax weights, external LM (DO THIS FIRST).** Keep the certified Python SMC loop and
  the external batched penzai LM untouched. A rejuvenation move at word `i` recomputes `scores`
  (LM + `dZ`) externally for the new prefix, then drives the existing @gen kernel's `"action"`
  through genjax `Update`/`Rejuvenate` so genjax produces the weight delta. Lowest risk; the LM stays
  external (it must — it's penzai), only the bookkeeping goes native.
- **Level B — full `@gen Scan` (LATER, OPTIONAL).** Express the whole intended sentence as
  `genjax.Scan` over words so the trace is first-class and `Update(action[i]=…)` re-runs the suffix
  incrementally. Most native, but needs the LM reachable from inside the Scan (cached deterministic
  call or arg-threading) and risks the certified path + compile time. **Defer**; only pursue if
  Level A's external bookkeeping proves clumsy. Do **not** start here.

`genjax.Scan` / `Update` / `EditRequest` / `Target` are top-level; `Rejuvenate` / `ImportanceK` live
under `_src.inference` (import directly). Confirmed present in the installed editable genjax.

---

## 2. The move, precisely (single-word full-conditional / Gibbs)

State per particle (unchanged from the filter): `(ctx_buf [LCTX], ctx_len, log_alpha [M+1], done)`,
plus a derived **`word→span` index** (word `w` occupies `ctx_buf[start_w : end_w]`; for `T_max = 1`
this is `start_w = w + seed_len`, trivially). A sweep revisits **words** `w` in a bounded window of
the last `lookback` words. For each `w`:

1. **Candidate set `C_w` (fixed size K, padded).** Each candidate is a **token span** (capacity
   `T_max`; at `T_max = 1` a single token): the COPY (the word currently at `w`), plus the SymSpell
   neighbours of the observed word `w` aligns to (reuse `_candidate_ids`; at `T_max > 1` this is the
   M:N generator from §0.1), plus a few top-J LM bridges. Cap K (`MAX_SUB_CANDIDATES` ladder) —
   K is the dominant cost knob.
2. **Recompute `log_alpha` from `w` forward.** Emission columns for words `< w` are unchanged; word
   `w`'s column changes (channel of the candidate **surface**, so token count is irrelevant here);
   the band `t`-indexing is unchanged (word *count* fixed). Re-run `_word_row_update` for
   `w..n_words` with out-of-range masked. Fixed-shape.
3. **Full conditional** `q(x) ∝ P_LM(x | prefix) · channel(x) · P_LM(suffix | prefix, x)` over `C_w`
   (REJUV_GOAL2). `P_LM(x|prefix)` is the **chain-rule product over the candidate's tokens** (one
   token at `T_max = 1`). The suffix term is the only new LM cost: a forward over the words after `w`
   for each of K candidates → the **KV-cache target** (R3). Sample `x_new` ~ `q`.
4. **Splice + re-pack.** Replace word `w`'s span with `x_new`'s tokens and re-pack `ctx_buf` by a
   cumsum-scatter (recompute `start`/`end` and `ctx_len` from per-word token counts), masking past
   the new `ctx_len`. At `T_max = 1` (equal lengths) this degenerates to a one-index `.set`; the
   general path is the *same code* with a variable span — **do not special-case `T_max = 1` away.**
5. **Weight.** Via genjax `Rejuvenate` (Level A): `w + bwd − fwd`. Gibbs ⇒ ≈ 0; assert it as a
   correctness check, then route into `log_w` *before* the next resample so mass can flow
   (REJUV_GOAL3 placement: pre-resample, not post).

No alignment surgery (DP recomputes), no *unbounded* raggedness (the splice is bounded by `T_max`),
fully vmappable over `[P, K, LCTX]`.

---

## 3. Phasing (incremental, certified, measure-before-optimize)

### R0 — Reframe + toy correctness harness (no LM, cheap, do first) ✅ DONE
The move lives in **`src/genjax_port/pairhmm_rejuv.py`** (additive; the certified `pairhmm_smc` is
untouched): `gibbs_sweep` resamples each word from its exact full conditional
`softmax(LM_prior + channel alpha[M])` over a candidate set (toy: the whole vocab → true Gibbs),
scored by the *same* pieces the enumeration uses. Two gates added to `tests/test_pairhmm_exact.py`:
`test_rejuv_leaves_exact_posterior_invariant` (cloud drawn from exact → 3 sweeps → MAP preserved,
`TV(after, exact) < 0.08`; observed 0.037) and `test_rejuv_recovers_collapsed_cloud` (all particles
forced onto the wrong same-length `the dog sat` → 6 sweeps → recovers exact MAP `the cat sat` at
p > 0.5; observed 0.91). **All 6 gates pass** (4 forward-filter + 2 rejuv) via the function-runner,
~61s. T_max=1, band=None (matches enumeration), no KV-cache / no genjax-`Rejuvenate` weight yet
(Gibbs needs none) — those are R1/R3.

### R1 — Fixed-shape Gibbs move, genjax weights, NO KV-cache ✅ DONE
Implement §2 on the fixed-shape state, scores recomputed via the existing external `lm_fn` (eat the
O(T) suffix cost for now — correctness/shape first). **Write it capacity-parametric (`T_max`) per
§0.1** — candidate = token span, surface-based channel, chain-rule LM, cumsum-scatter splice — even
though it runs at `T_max = 1`. The `T_max = 1` path must fall out of the general code, not be a
hard-coded single-index swap. Wire the weight through genjax `Rejuvenate`/`Update` (Level A).

**What was built.** `pairhmm_rejuv.py` rewritten capacity-parametric (additive; certified filter
untouched): the flat state is unpacked into per-word slots `word_tok [P,Wmax,T_max]`/`word_len`; a
move replaces a word's span by a fixed-shape slot `.set` and rebuilds the flat LM buffer with a
gather-by-cumsum re-pack (`_pack`) that handles unequal spans; the LM prior loops over **token**
positions (chain-rule), the channel DP over **word** slots (surface-based, token-count-agnostic).
`T_max=1` falls out of the general code (no single-index swap). The SMCP3 weight is produced by
genjax via the `Rejuvenate` recipe **inlined** as `_smcp3_move` (propose from the full conditional →
`Update` the trace → assess the reverse → `w + bwd − fwd`): the `Rejuvenate` *class* can't be used
directly here — its `argument_mapping(chm)`-only signature can't thread per-particle target scores,
and its proposal re-addresses the selected address (`MissingAddress`, confirmed in
`planning/kv_cache_spikes/rejuv_smcp3_spike.py`); the inline form uses the same genjax GFI primitives
and the identical SMCP3 formula. `gibbs_sweep` returns `(ctx_buf, move_logw)`; per-particle COPY is
prepended (index 0) and pool duplicates masked so the conditional isn't double-counted.

**Validated.** All **7 toy gates pass** (4 forward-filter + 3 rejuv) via the function-runner (~55s),
including a new `test_rejuv_smcp3_weight_zero` (full-conditional SMCP3 weight `max|w| < 1e-3` — the
`accept ≈ 1` built-it-right check; genjax owns the ratio). The collapse-cure itself is certified by
the **toy recovery gate** (cloud forced onto a wrong same-length reading → sweeps recover the exact
MAP). The **`cat/mat` case** is validated via `pythia_rejuv.recover` (run filter → resample to equal
weights → sweep → decode), which surfaced exactly the `noisy-channel-test-examples` caveat — the
collapse-cure only works where the **LM's target actually prefers the truth**:
- **pythia-70m, P=128:** the filter collapses (4/6 seeds before); the sweep recovers **5/6** —
  seed 2's `The cat cat on mat`→`The cat sat on mat` collapse is cleanly fixed (p 0.46 BAD → 0.81 OK).
  The seed-5 residual is **not an inference bug**: a direct conditional probe shows 70m's LM *slightly
  prefers the repetition* `The cat cat` over `The cat sat` (−50.45 vs −55.13), nearly cancelling the
  channel's correct 4.6-nat pull toward `sat` (conditional 0.42 `cat` vs 0.40 `sat`).
- **pythia-410m** (the `run_example_native` LM): the **move scores correctly** — the same conditional
  probe at the collapsed slot is now decisive (`sat` post **1.000**; LM −43.76 and channel agree). End
  to end 410m barely impoverishes at P=128 (5/6 before), so the sweep has little to cure and **preserves
  the good readings without regression** (5/6 after); its one non-truth seed *restores the dropped*
  `the` (`The cat sat on the mat`), an orthogonal word-count inference the substitution sweep doesn't
  touch (and arguably correct). So the cure is a 70m-at-low-P phenomenon; 410m needs it less.

**Done:** move is jit+vmap clean, weights produced by genjax and ≈0 for Gibbs, toy invariance holds,
and the collapse case is fixed at small P wherever the LM's full conditional prefers the truth
(toy recovery gate + 70m seed 2 + the decisive 410m conditional probe).

### R2 — Interleave into the Pythia loop + measure honestly ✅ DONE
Add a post-resample windowed sweep behind a flag (`rejuv="off"|"gibbs"`), pre-resample weight fold.
Run `run_example_native.sh` on the flat-posterior sentences + the eval set. **Report wall-clock +
LM-forward counts vs. no-rejuv — expect a balloon here; that measurement justifies R3.** Also run the
degeneracy counter (REJUV_GOAL1 Step 0: median `unique_gated / n_gate`) to choose dedup vs.
compaction.

**What was built.** Flag-gated `rejuv="off"|"gibbs"` threaded `pythia_word_caprop.run` →
`pairhmm_smc.run` (the certified forward filter is untouched — the sweep only runs inside the
`rejuv=="gibbs"` branch, **after each resample**, and the exact-enum gates run with it off). After a
resample the equal-weight cloud gets a windowed sweep (`pairhmm_rejuv.make_sweep`, last
`rejuv_lookback` words; jitted once per run, reused across resample events), `log_alpha` is recomputed
and the move's SMCP3 weight is folded into `log_w` **before the next resample** (REJUV_GOAL3 (b) —
mass flows). The mid-loop scorer is **done-aware** (new in R2): not-done particles score the partial
forward mass `logsumexp(log_alpha)` with no EOS term, done particles the terminal `alpha[M]` + EOS —
so the conditional is correct mid-sequence, not just at the end. A `rejuv_stats` dict collects the
LM-forward + degeneracy counters; `pairhmm_rejuv.build_pool` makes the per-slot SymSpell pool.

**Measured** (`pythia_rejuv.bench`, `The cat sat on mat.`, pythia-70m, P=128, lookback=3, 4 seeds):
- **Quality — rejuv improves the flat-posterior inferences.** MAP==truth went **3/4 → 4/4**: seed 2's
  collapse `The cat cat on mat`→`The cat sat on mat` (0.48→0.71) is fixed *in-loop*, and the already-
  correct seeds sharpen (0.74→0.86, 0.61→0.86) or hold (0.98→0.95). (The in-loop windowed sweep cures
  seed 2, which R1's end-of-loop sweep left wrong — mid-loop placement reaches the collapse earlier.)
- **Cost — the balloon, as predicted.** LM **row-forwards ×151** (filter 1152 = 9 steps·128 P; sweep
  172 800 = Σ over 5 resample events of window·(n_out+1)·P·Kt). Wall-clock ~3.6s→~29s (~8×), but that
  is **compile-heavy** — the sweep step recompiles per run (no bucketing/build-once yet). Both are
  exactly what **R3** targets: the KV-cache removes the per-candidate whole-sentence O(T) re-score
  (→ O(tail)), and compile-time control (bucket shapes, build-once) removes the recompile.
- **Degeneracy — dedup wins.** Median **unique/P ≈ 0.07** at sweep points (20 events) — the cloud is
  highly degenerate post-resample, so per GOAL1's rule the move's forward cost should scale with
  `unique` (a ~14× cut), i.e. **dedup**, not compaction (compaction is redundant when `n_gate ≈ P`).

**Done:** rejuvenation improves the flat-posterior inferences (3/4→4/4, others sharpened); the cost is
quantified (×151 row-forwards, unique/P≈0.07 ⇒ dedup) — the measurement that justifies R3.

### R3 — KV-cache the move's suffix re-scoring (the perf phase) ✅ DONE (KV win landed; dedup remains for <2×)
Apply the validated penzai prefix-KV fork (`rejuv-prefix-kv-cache-spike` memory: drive +
unbind/bind/fork) to the **K-candidate × suffix** forward only — the one place the win is structural
(shared prefix across K and across particles).

**The reframing that unlocked it.** The rejuvenation conditional is `q(x) ∝ LM(x|prefix) +
LM(suffix|prefix,x) + channel(x)`; the **prefix LM is identical for every candidate** (the move only
touches word `w`) so it **CANCELS** in the softmax / SMCP3 weight and need never be computed. So R2's
whole-sentence re-score (`_lm_logprior`, all token positions, per candidate) collapses to scoring the
short **suffix tail** `[x, suffix words, EOS?]` — exactly what a KV cache accelerates.

**What was built.** A `tail_logprobs` injection on `PairHMMModel`; the sweep (`make_sweep`) now scores
only the suffix tail. Pythia injects the **KV-cached** scorer (`lm_penzai.batch_tail_logprobs`,
`use_kv=True`) which **prefills the prefix ONCE per particle and shares its K/V across the K
candidates** (one full forward + K cheap single-token tails); the toy/default uses a generic uncached
chain-rule (`_tail_chain_uncached`). **Bug found + fixed:** the existing `_batch_tail_logprobs_kv`
rewound to `ilen-1` (overwriting the last ctx token, off-by-one) — **5.9 nats wrong**; rewritten to
the validated prefill-once convention (read `P(tail[0]|ctx)` from prefill logits at `ctx_len-1`, feed
the rest), now matches a hand chain-rule + the uncached scorer to ~1e-3
(`planning/kv_cache_spikes/tail_scorer_verify.py`). KV model **built eagerly** (pre-warmed in
`pythia_word_caprop.run`) — building it under the jitted step leaks a tracer. `max_tail = lookback+1`
(bounded suffix). The done-aware scoring (partial forward-mass vs terminal `alpha[M]`) carries over.

**Measured** (cat/mat, pythia-70m, P=128, lookback=3, 4 seeds): **full-forward balloon ×151 → ×2.67**
(filter 9 forwards + 15 shared prefills, + 60 cheap tail-steps) with **identical** quality (4/4 MAP,
logZ and posteriors match R2 to float — the prefix-cancel + KV is exact). All **7 toy gates pass**
(the suffix-tail conditional == the whole-sentence one). Wall-clock ~3.6s→~17s (~4.7×) is still
**compile-dominated** (the `make_sweep` step recompiles per run — see below).

**Remaining for the strict <2× target (scoped, not done here):** (1) **dedup** — median unique/P≈0.07,
so the prefills should run on the ~handful of unique post-resample buffers (≈×2.67→×1.1), but dedup is
host-side and forces un-jitting the step (GOAL1 tension); (2) the **per-run `make_sweep` recompile**
(it rebuilds the jitted step every `run`; pass `ctx`/pool as step args or memoize so it compiles once)
— this is what inflates wall-clock today; (3) the **surprisal gate**. Compile-time bucketing is already
handled by the fixed `max_tail`/`cache_len` (one KV specialization).

### R4 — Multi-token intended words (sequenced after Phase D, NOT a redesign)
Because R1 built the move capacity-parametric (§0.1), turning on multi-token rejuvenation is a
**parameter + ingredients bump**, not a rewrite: raise `T_max`; supply the M:N candidate generator
(multi-token surfaces); use chain-rule LM scoring of a candidate word (already the shape the suffix
forward uses); the cumsum-scatter splice already handles unequal lengths. **Prerequisite:** Phase D
of `PAIRHMM_RBSMC_PLAN.md` must first give the *forward* filter multi-token emission (so the cloud
contains multi-token words to rejuvenate). The KV-cache (R3) carries over and helps *more* (more
tokens to amortize). Add a toy multi-token gate mirroring R0. Nothing in R0–R3 may bake in
`T_max = 1`; this phase only flips the parameter and adds the two ingredients.

---

## 4. Risks & guardrails (carry the scars forward)

- **Never edit the certified forward filter.** Rejuvenation is additive + flag-gated; re-run the
  exact-enumeration gates after every change (they guard the refactor).
- **Weight NaNs.** `-inf − -inf = NaN` silently poisons categorical sampling — guard every move weight
  (existing gotcha).
- **SMCP3 variance → ESS collapse.** A bad proposal injects high-variance weights. Mitigation: the
  move is full-conditional (Gibbs), SMCP3 weight ≈ 0, and resample sits right behind it
  (REJUV_GOAL3 variance guard). Log ESS-after-move; if it collapses, fix the proposal, don't clamp.
- **Compile time is a tracked metric, not an afterthought.** The old KV-cache "drastically increased
  compile time." Bucket shapes, build-once, flag-gate, bounded suffix — and *report* compile time in
  R3, treat a regression as a failure.
- **Don't rebuild genjax by hand.** Use `Rejuvenate`/`Update` for the weight; only the external LM
  forward + the forward DP are ours.

## 5. Read-these-first (for whoever implements)
- `src/genjax_port/pairhmm_smc.py` — the certified filter; `_make_kernel` (@gen step, `"action"`/`"ev"`),
  `_word_row_update`, `run()` loop, terminal correction. The backbone; do not break it.
- `src/genjax_port/pythia_word_caprop.py` — `_candidate_ids` (single-token candidates), `channel_logpdf`,
  the surprisal gate / unigram machinery.
- `genjax/_src/inference/requests/rejuvenate.py` — the SMCP3 `Rejuvenate.edit` we wire to.
- `planning/REJUV_GOAL2…` / `…GOAL3…` — the full-conditional proposal + SMCP3 placement ideas (the
  math survives the paradigm change; the code references in them point at the archived stack).
- `planning/kv_cache_spikes/` — today's collapse diagnosis (the motivating bug) + the validated KV
  fork spike pattern.

## 6. Done when
A flag-gated Gibbs/SMCP3 rejuvenation sweep, weights via genjax `Rejuvenate`, (a) leaves the toy
exact posterior invariant, (b) recovers the `cat/mat` collapse at P=128, (c) holds/improves
`eval_rejuv`, (d) adds bounded overhead with the KV-cache and an acceptable compile time, and (e)
never touches the certified forward path. Update `run_example_native.sh` if rejuv becomes default
(`keep-run-example-script-current` memo).
