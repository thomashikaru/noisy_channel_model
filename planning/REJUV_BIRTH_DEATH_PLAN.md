# Birth/death rejuvenation — a single involutive add/remove-a-word move

**Status:** drafted 2026-06-21, **not yet implemented.** Branch: `rejuv-birth-death` (forked from
`align-action-channel` @ `efe204d`). Self-contained — a fresh session should execute it end-to-end.
**Reversibility contract:** all work is on this branch; the new move is a **separate sweep** gated behind a
new `rejuv` selector value (`rejuv="gibbs+bd"`), so the existing substitution-only `rejuv="gibbs"` path and
the certified `rejuv="off"` exact-enumeration path stay byte-for-byte untouched; no defaults change until a
final user-approval phase. To abandon: `git checkout align-action-channel` and delete this branch.

---

## 0. The gap we're closing

The current rejuvenation sweep (`pairhmm_rejuv.py`) is **substitution-only**: for each existing intended-word
slot `w` it resamples the word from `[COPY] ++ pool[w]` (`_candidates`, `pairhmm_rejuv.py:273`) and the move
is gated by `active = w < n_words` (`_apply_move`, `:345`). It **never changes `n_words`** — it can neither
remove a slot (delete a word) nor add one (insert a word). The number of intended words is fixed once the
forward filter commits it (`kernel`, `pairhmm_smc.py:298`, `n_words + 1` on advance).

Consequence, flagged in `ALIGN_OPT_RESULTS.md:127`: a duplicated intended word (`INS-01a "handed handed"`)
is never removed, because nothing downstream of the forward pass can drop a word. More generally, all
word-count corrections during rejuvenation are impossible — restoring a dropped word can *only* happen in
the forward pass (via the `alpha` del-action), and removing a spurious one can happen *nowhere*.

**The fix:** a rejuvenation move that changes `n_words` by ±1, leaving the posterior invariant.

### Why one observed word can be "removed" with no new channel code

The channel forward DP (`word_dp.channel_carry` → `alpha[M]`) already marginalizes the full edit alignment,
including channel **insertions** (a spurious *observed* word, cost `WINS`). So when a death move removes an
intended "handed", the now-unmatched observed "handed" is automatically re-explained by the DP as a WINS
insertion — the target `π(y')` for the shorter sentence is already correct. No alignment bookkeeping is added;
we only need to *propose* the shorter (or longer) sentence and score it through the existing DP.

## 1. The design — one involutive birth/death move

A single move that, at a chosen position, **either inserts or removes one intended word**, with an
involution `φ` that flips the two directions. Self-inverse ⇒ dimension matching is automatic and the
"Jacobian" is 1 (everything is discrete).

Augmented variable `(y, d, w, x)`:
- `y` — the intended-word sequence (the latent we rejuvenate),
- `d ∈ {birth, death}` — direction,
- `w` — a position/gap index,
- `x` — a word; the **auxiliary that matches dimensions**.

The involution:

```
φ(y, birth, w, x) = ( insert(y, w, x), death, w, ∅   )
φ(y, death, w, ∅) = ( delete(y, w),    birth, w, y_w )
```

`φ∘φ = id`. Birth *draws* a word (randomness +1, state +1); death *recovers* the deleted word into `x`
deterministically (randomness −1, state −1) — so total (state+randomness) dimension is preserved, the
involutive-MCMC / reversible-jump requirement. This is genuinely **one** move (direction sampled inside it)
and **its own inverse**, per the request.

It runs as a **separate sweep** beside the substitution sweep; both target the same `π` and compose. Keeping
them separate (rather than folding DELETE/INSERT candidates into the existing per-slot categorical) avoids
mixing a fixed-dim full-conditional Gibbs step (weight ≈ 0) with a trans-dim step (weight ≠ 0) in one
normalization, and keeps the certified substitution path bit-identical and the new move independently
gateable and testable.

## 2. The weight (SMCP3 / reversible-jump, Jacobian 1)

Target `π = P_LM^λ · P_channel` — exactly what the current sweep scores (`lm_temp * chain + chan`,
`pairhmm_rejuv.py:392`). For a **birth** proposing word `x` at gap `w` (reverse is a death at that position
in `y'`):

```
log W = [ logπ(y') − logπ(y) ]                                  ← target ratio
      + [ log p_death(y') + log q_del(w | y') ]                 ← reverse (death) proposal
      − [ log p_birth(y)  + log κ(w | y) + log q_ins(x | w, y) ] ← forward (birth) proposal
```

and the mirror for a **death** (forward `p_death·q_del`; reverse `p_birth·κ·q_ins` re-inserting the removed
word). Two efficiencies:

- **Channel:** `logπ_channel(y')` is one extra `channel_carry → alpha[M]` call — the DP already exists and is
  length-agnostic. (For DONE particles read `alpha[M]`; for mid-loop ones `logsumexp(alpha)`, exactly as
  `_chan_scores`, `pairhmm_rejuv.py:291`.)
- **LM:** birth/death at `w` changes the sentence only from `w` onward, so the LM **prefix cancels** in the
  ratio — score only the suffix tail, reusing the KV suffix-tail scorer the sweep already drives
  (`_tail_inputs`/`tail_fn`, `:312`/`:391`).

Unlike the substitution sweep (full-conditional Gibbs, weight ≈ 0 — asserted as a build-it-right check in
`_smcp3_move`), this move carries **real weight**; the plumbing to fold `move_logw` into `log_w` before
resampling already exists (the sweep returns `move_logw`; the filter folds it pre-resample). Correctness is
independent of proposal quality; **proposal quality only controls weight variance / ESS** (the main practical
risk), tuned in §3.

I will **hand-compute this weight vectorized over P** (as `_smcp3_move` already inlines the recipe rather than
threading per-particle scores through genjax's trace API) — the trans-dim ratio is the clean closed form above.

## 3. Proposals (efficiency, not correctness)

- **Death** `q_del(w | y)` — near-conditional: enumerate removing each current word, score each `(n−1)`-word
  target, sample `w ∝ softmax`. Concentrates on duplicates / low-LM-gain / WINS-explainable words. Suffix-LM
  sharing makes the `n` evals cheap.
- **Birth** `q_ins(w, x | y)` — place `w` where the channel currently pays an alignment/WDEL penalty; draw `x`
  from the **forward filter's own inventory**: the channel-compatible pool plus the top-J LM "bridge" words
  (`_rejuv_pool_from_inventory`, `pairhmm_smc.py:178`; the `_caprop_scores` bridge set, `:213`). Word choice can
  itself be a local full conditional over that pool.
- `p_birth/p_death` = ½, forced at the boundaries `n = 0` (birth only) and `n = Wmax` (death only); the weight
  formula absorbs the boundary asymmetry.

## 4. Implementation mapping (where the code goes)

- **Buffers** (`word_tok [P,Wmax,T_max]`, `word_len`, `word_surf`, `n_words`): insert/delete a slot = a shifted
  copy of those arrays + the existing `_pack` cumsum re-pack (`pairhmm_rejuv.py:116`), which already handles
  variable spans. `n_words ± 1`. Births need headroom `n_words < Wmax`; `Wmax = M + slack` — may need to bump
  `slack` (cheap) if births are frequent.
- **New code**, all additive to `pairhmm_rejuv.py`: `insert(y,w,x)` / `delete(y,w)` + auxiliary recovery; the
  position/word kernels (`q_del`, `q_ins`, `κ`); the vectorized `log W`; and a `birth_death_sweep(...)`
  returning `(state', move_logw)` in the same shape contract as `make_sweep`'s `sweep`.
- **Hook:** new `rejuv="gibbs+bd"` selector in `pairhmm_smc.run` (`:462`) that runs the substitution sweep then
  the birth/death sweep per resample event. `rejuv="off"` and `rejuv="gibbs"` unchanged.

## 5. Phased execution

The move is **atomic** — both directions exist from the first working version. A death's SMCP3 weight
references the birth/insert reverse density `q_ins` (the `+ reverse_prop` term in §2), and a one-directional
kernel cannot be π-invariant (it only ever shrinks `n_words`). So there is **no "death-only" build**. What
stages is (a) **proposal sophistication** and (b) **which cases we validate against** — not direction.

- **Phase 0 — involution core.** `insert(y,w,x)` / `delete(y,w)` + auxiliary recovery on the buffer reps, and
  the `φ` round-trip — **both** directions (the involution *is* the pair). No scoring yet. Gate: **involution
  round-trip test** (apply, re-apply with recovered aux, assert exact state + fwd/bwd density consistency) —
  the analogue of the `weight ≈ 0` build-it-right assertion in `_smcp3_move`.
- **Phase 1 — the complete move, cheap proposals.** Both directions, the trans-dim weight, folded into
  `log_w`; start with cheap proposals (uniform `κ`, near-conditional `q_del`, top-of-pool `q_ins`). Validate
  first on the **death-dominated** motivating case (`INS-01a "handed handed"` duplicate removed) and the toy
  **posterior-invariance** test (small enumerable variable-length target: equally-weighted-in ⇒ correctly
  reweighted); certified `off`/`gibbs` paths bit-identical.
- **Phase 2 — informed proposals.** Upgrade `q_ins` to the forward-filter inventory + LM bridges and `q_del`
  to the full near-conditional. Gate: **birth-dominated** restorations the forward filter already does
  (`DEL-of`, `INS-02a`, the `ALIGN_OPT_RESULTS.md:65` wins) don't regress, and ESS isn't materially worse
  than `gibbs`.
- **Phase 3 — battery + writeup.** 40-item battery (E/L/junk), `gibbs` vs `gibbs+bd`; record in
  `REJUV_BIRTH_DEATH_RESULTS.md`. Recommend default only on a clean win.

## 5b. Phase-1 status / finding (2026-06-21)

Phase 1 is **built and proven correct**, and surfaced the expected variance wall:

- **Correctness chain closed.** (i) `_bd_log_weight` passes an EXACT transition-sum reversible-jump
  invariance test (`Σ_y π(y)q_fwd exp(W)=π(y')`, err <1e-6); (ii) `_make_bd_score_fn` reproduces the exact
  enumeration joint `LM+channel` to <1e-4 over varied-length sentences (`test_bd_score_fn_matches_exact_joint`).
  Weight ∘ correct-target ⇒ the integrated move is correct. Certified `off`/`gibbs` paths bit-identical
  (32 gates green).
- **Uniform proposals are too noisy for the live filter.** Wired `rejuv="gibbs+bd"` (move after the
  substitution sweep, done-only, n_attempts=2, pool = observed surfaces). On a toy duplicate (`the cat cat
  sat`, target prefers the deduped `the cat sat` by +0.81 nats) `gibbs+bd` *degraded* the posterior vs
  `gibbs` and shifted logZ ~1.3 nats — the signature of high weight-variance (logsumexp downward bias),
  NOT bias: every done particle does 2 random add/remove moves per resample, ~half of them bad births of
  random pool words. This is exactly the §7 risk.
- **Consequence for the phase boundary.** The behavioral win (duplicate removal *improving* the posterior,
  `handed handed`) needs the variance down, i.e. **Phase 2's informed proposals** (near-conditional `q_del`
  that targets the removable duplicate; channel/LM-informed `q_ins`). The uniform Phase-1 move is the
  correct-but-not-useful scaffold; Phase 2 makes it help. (Tune: also consider fewer attempts / running
  bd only near terminal.)

## 6. Tests / correctness ledger

1. **Involution round-trip** (Phase 0) — `φ∘φ = id` on state + fwd/bwd density consistency.
2. **Posterior invariance on the toy** (Phase 1) — enumerable variable-length target; the move leaves it
   invariant.
3. **Certified path untouched** — `rejuv="off"` exact-enumeration gates and `rejuv="gibbs"` results
   bit-identical with the move off.
4. **Motivating case** — `INS-01a "handed handed"` duplicate removed (Phase 1).
5. **No-regression** — forward-pass restorations preserved under `gibbs+bd` (Phase 2).

## 7. Risks

- **Weight variance / ESS** — the only real risk; trans-dim weight ≠ 0. Mitigated by near-conditional proposals
  (§3); measured in Phase 2/3.
- **Two restoration paths** — birth duplicates the forward filter's deletion-restoration. Same posterior, but
  watch for double-restoration instability (Phase 2 no-regression gate).
- **Headroom** — births capped at `Wmax`; bump `slack` if needed.

## 8. Open choices (decide before Phase 2)

- Run birth/death **per position** (like the substitution sweep) vs a fixed **K attempts** per resample event.
- Whether `q_ins` word choice is a full local conditional (lower variance, costlier) or a cheap top-1 from the
  pool.

## 9. Phase 2 — KICKOFF (start here next session)

**Goal:** drive the weight variance down so the move *helps* — the toy duplicate (`the cat cat sat`) improves
under `gibbs+bd` (deduped reading ≥ `gibbs`, no `logZ` blow-up), the live `handed handed` duplicate is removed,
and forward-pass restorations don't regress. Phase 1 proved the move correct; Phase 2 makes it useful.

**State of the code (all committed, branch `rejuv-birth-death`):**
- `pairhmm_rejuv.birth_death_move` — proposals are currently **uniform**: direction p=½ (forced at
  boundaries), gap `κ`=1/(#gaps), `q_ins`=1/Kc over a fixed pool, `q_del`=1/D over deletable positions.
- `pairhmm_rejuv._bd_log_weight` — **hardcodes those uniform densities** (`log(n+1)`, `log Kc`, `log D` are
  baked in). ⚠️ **This is the key refactor:** for non-uniform proposals the weight must use the *actual*
  forward/reverse proposal log-densities, not the uniform ones.
- `pairhmm_rejuv.make_bd_sweep` wraps the move (done-only, `n_attempts=2` hardcoded at the call site in
  `pairhmm_smc.py`'s `rejuv=="gibbs+bd"` build block). `_make_bd_score_fn` (the target) is **done — proven
  exact**, reuse it as-is.

**Step 1 — generalize the weight (do this first). ✅ DONE (2026-06-22).** `_bd_log_weight` is now the
proposal-agnostic `W = (logπ(y') − logπ(y)) + log q_bwd − log q_fwd` (a 4-arg pure ratio); `birth_death_move`
computes, for the chosen move, `log q_fwd` (= `log p_dir + log κ + log q_ins` for a birth; `log p_dir +
log q_del` for a death) and the **reverse** `log q_bwd` (density of the move that undoes it from `y'`,
recomputing the direction rule at `y'`). The old uniform formula is now the explicit special case computed in
the move (proven bit-identical term-by-term). `test_rj_weight_invariance_exact` was rewritten to build the
forward (`qf`) AND reverse (`qb`) densities and feed them in — it now certifies densities + weight END-TO-END
(err <1e-6) and is the regression guard for Step 2's informed densities. Gates: birth/death 6/6 + full suite
38/38 (off/gibbs bit-identical). **Resume at Step 2.**

**Step 2 — informed proposals. ✅ CORE DONE (2026-06-22).**
- `q_del` **near-conditional** (`_del_logq`): score `π(y with w removed)` for each deletable position
  (~`n_words` `score_fn` calls via `lax.map`), `q_del(w) ∝ softmax`. Concentrates deaths on the removable
  duplicate. Its reverse density (a death at y') is `_del_logq` re-evaluated at y', indexed at the inserted slot.
- `q_ins` **near-conditional WORD** (`_ins_logq`): gap stays uniform, but the inserted word `x ∝ softmax` over
  the pool of `π(y + x @ gap)`. Balancing q_ins against q_del is the variance lever — informed births stop
  proposing easy-to-undo spurious words (the Phase-1 uniform asymmetry was the logZ-depressing source). Pool is
  still observed-surfaces; broadening to the full `_rejuv_pool_from_inventory` + LM bridges is a later refinement.
- Both directions' ACTUAL densities are fed to the Step-1 weight; certified by the rewritten exact
  invariance test `test_rj_weight_invariance_informed` (both informed, err <1e-6). `n_attempts=1` (one
  targeted move/event). Gates: birth/death 7/7 + full suite 39/39 (off/gibbs bit-identical).
- **Toy gate-3 result** (`the cat cat sat`, `P=3000`, `planning/bd_toy_gate.py`): deduped `the cat sat` mass
  **off 0.412 / gibbs 0.428 / gibbs+bd 0.468** (PASS, ≥ gibbs); duplicate `the cat cat sat` **0.178 / 0.169 /
  0.122** (driven down — the behavioral win). logZ depression vs `gibbs` went **1.3 (Phase-1 uniform) → 0.72
  (informed q_del) → 0.56 (both informed)**: improved but NOT fully closed. The residual is the inherent
  variance of a ½/½ trans-dim move applied at EVERY resample event to already-converged done particles → Step 3.
- **NaN guard (live bugfix).** The live band-limited align channel can make a proposed/source parse impossible
  (logπ = −inf), so the target ratio is −inf−(−inf) = NaN, which poisons logZ + every softmax. Fix in
  `birth_death_move`: `W = where(isnan(W) | (W==+inf), −inf, W)` — degenerate moves get −inf weight (discarded
  by resampling); finite weights and clean −inf rejections pass through. Toy (`band=None`) is bit-identical.
- **Live gate-4 result** (`planning/bd_live_gate.py`, pythia-70m, align channel). Warmup `the the dog ran`
  (`P=64`): `gibbs` keeps `The the dog ran` (0.891, top1=miss); `gibbs+bd` → **`The dog ran` (0.685, top1=HIT)**
  — duplicate REMOVED (the behavioral goal the substitution-only sweep cannot reach). logZ −39.6 (gibbs) →
  −43.6 (gibbs+bd): **~4-nat depression, much larger than the toy's 0.56** (band-limited channel ⇒ noisier
  move) — the strongest motivation for Step 3. Cost: ~18s gibbs vs ~124s gibbs+bd (the O(Wmax²) un-jitted
  scoring). Full INS-01 (9 words, Wmax≈12) not yet run — slow; do it after Step 3 cuts aggression/cost.

**Step 3 — tune aggression. ✅ INVESTIGATED (2026-06-22) — near-terminal gating BACKFIRES.** Added the
tunable `bd_min_done` (`pairhmm_smc.run`; fire bd only once a fraction ≥ bd_min_done of particles are done;
0.0 = every event = original). Toy `the cat cat sat` (`planning/bd_toy_step3.py`):

| config | logZ | dedup `the cat sat` | dup `the cat cat sat` |
|---|---|---|---|
| gibbs (baseline) | −9.43 | 0.428 | 0.169 |
| gibbs+bd `min_done=0.0` (every event) | −9.99 | **0.468** | **0.122** |
| gibbs+bd `min_done=0.5` / `0.9` | −9.65 | 0.336 | 0.293 |

Gating to near-terminal cuts the logZ depression (0.56→0.22) but **destroys the behavioral win** — dedup
drops BELOW the gibbs baseline and the duplicate mass rises. The interleaved **resampling between bd events
is load-bearing**: it amplifies good deaths and discards the junk births the ½ direction-coin makes; fire bd
only at the end and that cleanup is gone. ⇒ The logZ depression is the COUPLED COST of the mechanism that
makes the move help, NOT removable by firing less. **Keep `bd_min_done=0.0` (every event).** The real
remaining variance lever is BETTER PROPOSALS — chiefly an INFORMED DIRECTION (don't fire junk births at
converged particles; the weight already supports arbitrary p_dir via p_dir_fwd/p_dir_rev, so this is a clean
extension) — but it adds the reverse-direction-prob bookkeeping and risks bias, so defer unless gate 5
(no-regression) shows the variance actually HURTS other sentences.

**GATE 5 (no-regression, live, pythia-70m align channel, P=128, seed 0) DONE — SPLIT: one win, one
regression (`planning/bd_gate5.py`).** Ran `{off, gibbs, gibbs+bd}` on the two restoration items the align
calibration cited:

| item | correction | off | gibbs | gibbs+bd | logZ (gibbs→bd) |
|---|---|---|---|---|---|
| INS-02a `the cat sat on on the mat` | remove doubled `on` (DELETION) | 0.000 | 0.000 | **0.985 (HIT)** | −53.71 → −56.98 |
| DEL-of-01a `this is one the best` | restore dropped `of` (INSERTION) | 0.016 | 0.047 | **0.000 (REGRESSION)** | −30.06 → −33.64 |

These are the two faces of ONE mechanism. **INS-02a is a THIRD behavioral win** (beyond gates 3/4): the
doubled `on` is a deletion bd can execute, and neither the forward filter nor the sub-sweep can (both stuck
at 0.000) → bd 0.985. **DEL-of-01a is a REAL regression**: the fix is inserting an *unseen* word (`of` never
appears in the observed), and bd's **birth pool is observed-surfaces-only**, so births structurally cannot
propose it; meanwhile the per-event bd reweight + resample COLLAPSES the cloud onto the dominant literal mode
(1.000 on `this is one the best`), wiping the 0.047 of restored-`of` particles the forward filter built. The
SAME "resampling between bd events is load-bearing" property that helps INS-02a hurts here. ⇒ **bd is a
DELETION specialist that damages insertion-restorations it cannot perform; NOT safe as a global default.**
This is the gate-5 trigger the Step-3 deferral named — the variance DOES hurt other sentences.

**ROOT CAUSE = an unprincipled birth pool (NOT the direction coin).** Why can't bd restore `of`? Because the
bd candidate pool is built from **column 0 only** of the rejuv inventory — the verbatim OBSERVED surface of
each word (`pairhmm_smc.py:640-647`, `for i in range(M): sid = ps[i,0]`). The forward filter's own proposal
does NOT work this way: `_caprop_scores` (`pairhmm_smc.py:243-245`) unions the per-word SymSpell neighbours
with the **top-J LM-predicted words** (`top_j = top_k(lm_word, J)`, the comment calls them "fluency/deletion
bridges") — that LM-contextual bridge pool is exactly how the forward filter proposes a missing `of`. The bd
pool throws all of that away and keeps only observed surfaces. That restriction was a Phase-1/2 SCAFFOLDING
shortcut (the Step-2 note already flagged "broadening to the full inventory + LM bridges is a later
refinement"), NOT a modeling choice. The pool `cand_surf` controls BOTH directions — births insert from it
(`_ins_logq`) AND a slot is deletable only if its surface is in it (`_in_pool`/`_deletable_count`). So INS-02a
works (`on` is observed ⇒ in pool ⇒ removable) and DEL-of regresses (`of` not observed ⇒ not in pool ⇒ births
structurally cannot reach it, and the bd reweight+resample then wipes the forward filter's restored `of`).

**FIX (Phase 2.5 — PRIMARY): unify the bd pool with the forward-filter proposal.** Enrich the bd pool from
"observed surfaces" to the full candidate inventory PLUS the per-position top-J LM bridges (the SAME `top_j`
the caprop uses), deduped + capped. The near-conditional `_ins_logq` already does CONTEXTUAL word selection
(it scores `logπ(y + cand @ gap)` for each candidate and softmaxes), so it just needs the right candidate
*present* — a fixed-but-enriched GLOBAL pool likely suffices; per-gap contextual top-J is a later escalation
only if the global pool dilutes the right word. This attacks the regression at its source and should turn
DEL-of into a WIN (bd could insert `of`, not just dedupe). It is compatible with the existing machinery: Step 1
already made `_bd_log_weight` proposal-agnostic, and a larger pool automatically (and correctly) makes more
parse words deletable, with reverse densities still well-defined (in-pool ⇒ positive). COST: `_ins_logq` is
O(Kc) `score_fn` calls (run twice, fwd+rev), so Kc growth is the price — cap the bd pool (Kc_bd ≈ 24–32) and
the deferred suffix-tail KV sharing becomes the gating perf win. Lever A (informed/gated DIRECTION — fire a
death only when a genuinely removable word exists) is now DEMOTED to an optional complementary variance lever,
not the fix. Concrete change sketch + the re-run gates are in §10.

**Gates (in order):** (1) generalized `_bd_log_weight` passes the updated exact RJ-invariance test;
(2) certified `off`/`gibbs` still bit-identical; (3) toy `the cat cat sat` — `gibbs+bd` deduped-reading mass
≥ `gibbs`, `logZ` not depressed; (4) live `handed handed` duplicate removed; (5) no-regression on
restorations (`DEL-of`, `INS-02a`).

**Quick behavioral harness** (the Phase-1 throwaway, recipe to recreate): run `pairhmm_smc.run` on a duplicate
sentence with `rejuv` in `{off, gibbs, gibbs+bd}` and compare `pairhmm_smc.decode(...)` top posteriors + `logZ`.
Toy: `_toy_model(lm_logits)` (from `tests/test_pairhmm_exact`), `OBS="the cat cat sat"`, `P≈3000`, `band=None`,
`WDEL/WINS` from `test_pairhmm_exact`. Live: a pythia `handed handed` sentence.

**Watch-outs:** births capped at `Wmax=M+slack` (bump `slack` if births are frequent); the move runs
**done-only** by design (mid-construction births would desync the forward filter's `n_words`) — keep it that way.

## 10. Phase 2.5 — enrich the bd pool with LM bridges (concrete sketch)

**Diagnosis recap.** The bd pool = observed COPY surfaces only (`pairhmm_smc.py:640-647`); the forward filter's
proposal also carries the top-J LM bridges (`_caprop_scores`, `pairhmm_smc.py:243-245`). DEL-of regresses
because `of` is a bridge word, never an observed surface, so births can't reach it. Fix = give the bd pool the
same bridges.

**The change (one place: the `rejuv=="gibbs+bd"` pool-build block, `pairhmm_smc.py:640-652`).** Today it loops
`for i in range(M): take ps[i,0]` (col-0 COPY surface, dedup). Replace with a builder that unions, deduped + capped to `Kc_bd`:
- the existing observed COPY surfaces (col 0) — keep (cheap, always-valid re-inserts; covers the duplicate-removal wins);
- the per-position **top-J LM bridges**: compute `top_k(lm_word_p, J)` at each observed-word position from the
  observed-sentence prefix (prime + words so far) — the same quantity `_caprop_scores` computes, but evaluated
  ONCE statically over the observed string before the SMC loop (not per-particle). Each bridge token id is a
  single-token candidate `((tid,), surf)`, so it slots into the `(cand_tok, cand_len, cand_surf)` triple exactly
  like the col-0 entries; multi-token bridges can wait.
- (optional) the SymSpell neighbours already in `rj_pool` cols 1..Ke — lower priority; the LM bridges are what
  fix DEL-of.

Everything downstream is UNCHANGED: `_ins_logq` softmaxes `logπ(y+cand@gap)` over the bigger `cand_*` (so word
choice stays contextual — it just has `of` available now); `_deletable_count`/`_in_pool`/`rem_ci` key off
`cand_surf`, so the enriched pool automatically makes bridge-inserted words deletable with well-defined reverse
densities; `_bd_log_weight` is already proposal-agnostic. No weight-math change.

**Pick `Kc_bd`.** `_ins_logq` is `lax.map` over `Kc` `score_fn` calls, run twice per move (fwd `w_b` + rev
`w_d`); `score_fn` is itself ~O(Wmax) LM steps. Current `Kc≈M` (5–9); a full per-position top-J union is
`≤ M·J` (with J=8 → ~40–70) → ~5–10× the dominant cost on top of the already-slow ~370–400 s/case. So CAP at
`Kc_bd ≈ 24–32` (keep col-0 surfaces + the highest-prob bridges, deduped) and treat the deferred **suffix-tail
KV sharing** as the now-gating perf win (it cancels the shared prefix so each `score_fn` scores only the tail).

**Re-run gates after the change** (`planning/bd_gate5.py` + the toy/live harnesses, all already written):
- (3 redo) toy `the cat cat sat`: dedup mass still ≥ gibbs (don't lose the win); logZ depression not worse;
- (4 redo) live `the the dog ran` / `handed handed`: duplicate still removed;
- (5 redo) DEL-of-01a: target `of` mass now ≥ gibbs (regression GONE; ideally a HIT — the new win);
       INS-02a: still ~0.985 (bridges shouldn't hurt the deletion case);
- a clean-keep guard (e.g. `INS-02b` / a plausible sentence): bridges must not induce spurious births on inputs
  that need no edit (the symmetric risk of a richer insert pool).
Then proceed to the Phase 3 battery (gibbs vs gibbs+bd, E/L/junk) + recommend default.

## 11. Phase 2.5 Step 2 — the STAY branch (the real blocker; do this BEFORE perf)

**Gate-5-redux result (bridges IMPLEMENTED, `bd_bridge_j`/`bd_pool_cap`; `planning/bd_gate5.py`, P=128).** Three
configs × three items:

| item | gibbs | bd j=0 (off) | bd j=3 (bridges) |
|---|---|---|---|
| DEL-of-01a (restore `of`) | 0.047 | 0.000 | 0.001 ❌ still collapses to literal (0.993) |
| INS-02a (remove `on on`) | 0.000 | 0.985 | 0.976 ✓ win holds |
| INS-02b (CLEAN keep) | 0.984 | **0.000** | 0.302 ❌ clean sentence DESTROYED |

The bridge fix is mechanically correct — `bd j=0` reproduced the prior gate-5 numbers EXACTLY (bit-identical),
`of` re-enters the cloud (0→0.001), INS-02b partly recovers (0→0.302), INS-02a holds, cost ~1.6× (410→665 s).
But it is INSUFFICIENT, and the **clean-keep guard exposed the real blocker: bd over-deletes on clean inputs.**
INS-02b is a perfect sentence (gibbs 0.984); `bd` collapses it to `The sat on the mat`/`The cat on the mat`
with a ~10-nat logZ depression.

**ROOT CAUSE (verified in `birth_death_move`): no STAY option.** A done particle keeps its state only when
`none` (neither birth nor death feasible). On any normal sentence both are feasible, so EVERY done particle is
forced to move off its parse, and the resample fires immediately — locking in the damage before a corrective
move can run. Fine when a move improves π (INS-02a duplicate); destructive at an already-good parse (clean
sentence, or the forward-filter-restored `of`, which is a minority the forced move also displaces).

**FIX: a 3-way direction {stay, birth, death}, `p_stay = s`.** The kernel becomes `s·I + (1−s)·K_old`. Because
the `(1−s)` factor multiplies BOTH the forward direction prob (`p_dir_fwd`) AND the reverse direction prob
(`p_dir_rev`) — the reverse of a birth/death is itself a not-stay move — it CANCELS in every weight:
`W = Δlogπ + log q_bwd − log q_fwd` is UNCHANGED. Invariance holds for any `s`:
`Σ_y π(y) q_fwd(y*|y) e^W = s·π(y*) + (1−s)·[Σ over K_old = π(y*)] = π(y*)`. So the implementation is minimal:
sample `is_stay ~ Bernoulli(s)`; stay particles keep their state with `move_logw = 0`; everything else (the
birth/death proposals, the densities, `_bd_log_weight`) is untouched. `p_stay = 0` reproduces the current
always-move behavior bit-identically (the regression guard / default).

**Why this differs from the FAILED `bd_min_done` gating (Step 3).** `bd_min_done` removed whole bd events
(near-terminal only) and DESTROYED the dedup win because the interleaved resampling is load-bearing. `p_stay`
keeps bd firing at EVERY event but lets a fraction of particles stay — so at each event the resample compares
movers vs stayers: on a duplicate the good death-mover (W>0) up-weights and wins; on a clean sentence the
stayer (W=0) beats the damage-movers (W<0) and the parse survives. It throttles aggression WITHOUT removing the
load-bearing per-event resampling.

**Expected gate effects.** INS-02b → preserved (~0.98); DEL-of → returns to ~gibbs (0.047) no-regression (a WIN
needs the model to actually prefer `of`, which pythia-70m barely does); INS-02a → dedup still wins; toy
`the cat cat sat` → dedup ≥ gibbs (maybe marginally slower). Default `p_stay` likely ~0.5 (balanced); tune on
the gates.

**Steps.** (1) add `p_stay` to `birth_death_move` + thread `bd_p_stay` through `make_bd_sweep` → `pairhmm_smc.run`
→ `pythia_word_caprop.run` (default 0.0); (2) add `test_rj_weight_invariance_with_stay` (full mixture kernel:
diagonal `s·π` + `(1−s)`-scaled moves, err < 1e-6); (3) re-run bd selftests + toy gate + gate 5 with `bd_p_stay`.

**STEP 2 RESULT — DONE, blocker FIXED (live, `planning/bd_gate5_stay.log`, bridges off).**

| item | gibbs | stay=0 (broken) | stay=0.1 | stay=0.3 |
|---|---|---|---|---|
| INS-02b CLEAN keep | 0.984 | **0.000** | **0.999** | **1.000** |
| INS-02a dedup | 0.000 | 0.985 | 0.691 | 0.721 |
| DEL-of restore `of` | 0.047 | 0.000 (junk) | 0.000 (literal 0.94) | 0.000 (0.96) |

Clean-keep FIXED at just `p_stay=0.1` (0→0.999). Dedup PRESERVED (strong HIT). DEL-of reframed: bd correctly
removes `of` because pythia-70m scores `one the best` > `one of the best` — a MODEL limit, not a move bug. Toy
dedup ≥ gibbs only for `p_stay`≤0.1-0.2 (weak-signal pessimistic case); live tolerates more. 8/8 bd gates +
27/27 exact green. Default `p_stay` ~0.1-0.2 (tune Phase 3). Bridges now orthogonal/opt-in (default off).

## 12. Perf — DONE (single-forward teacher forcing; ~4× total, ~5.7× on the bd move)

The planned "suffix-tail KV sharing" was aimed at the wrong cost. The actual bottleneck: `_lm_logprior` (the bd
score_fn's LM term) called `model.lm_fn(bufs, pos)` in a LOOP over positions, and `lm_fn` runs a FULL
`_raw_logits(bufs)` transformer forward then keeps ONE row — so the SAME forward was recomputed `n_out+1` times
per score, and `_del_logq`/`_ins_logq` call the score O(`Wmax`)/O(`Kc`) times per move.

**Fix:** one teacher-forcing forward gives every position's next-token logprob. New
`lm_penzai.seq_token_logprobs(bufs) → [N, seq]` (gather all next-token logprobs from a single pass) + new
`PairHMMModel.seq_token_logprobs` field; `_lm_logprior` reads from it when present, else the per-position loop
(toy / custom `lm_fn` → bit-identical). The EOS term needs no special case: `bufs` is padded with `eos_id`, so
the per-position logprob at the first pad slot IS `log P(EOS | tokens)`. `_lm_logprior` is used ONLY by bd, so
the change is isolated.

**Verified:** posterior + logZ match the slow path EXACTLY (INS-02a 0.691/−56.34, INS-02b 0.999/−50.51); gibbs+bd
375→87 s and 263→67 s (bd-added cost ~348→~60 s). Battery now ~1 hr/config (was ~4 hr). The suffix-tail KV
cancellation could still stack on top but is far more complex (prefix-cancel + per-position deletion rewind) and
lower marginal value now that the LM is no longer the sole cost — deferred unless the battery needs it.

**Phase 2.5 is complete and COMMITTED** (3 commits on `rejuv-birth-death`):
`0d989c0` LM-bridge pool · `d8d7c4d` STAY branch (clean-keep fix) · `a4759b1` single-forward perf.

## 13. Phase 3 — kickoff (START HERE next session)

**State.** The bd move is now SAFE and fast. `rejuv="gibbs+bd"` adds a symmetric birth/death rejuvenation move
(remove/insert a word) on top of the substitution-only sub-sweep. The clean-keep blocker is fixed by the STAY
branch; the move weights are exact (RJ invariance certified incl. stay); the LM scoring is single-forward.
8/8 bd gates + 27/27 exact green. `off`/`gibbs` are bit-identical to before; gibbs+bd with the default knobs
(`bd_p_stay=0.0`, `bd_bridge_j=0`) is the always-move/observed-only move — so **Phase 3 must pass the knobs
explicitly.**

**Knobs (in `pythia_word_caprop.run` / `pairhmm_smc.run`).**
- `bd_p_stay` — per-event STAY probability. **THE key knob.** 0.0 = broken (forced move, destroys clean
  sentences). Recommended live ~**0.1–0.3** (0.1 already fixes clean-keep to 0.999 and keeps dedup a strong HIT).
  Toy dedup wants ≤0.1–0.2, but the toy is a weak-signal pessimistic case — trust the live battery, not the toy.
- `bd_bridge_j` — per-gap top-J LM bridges added to the birth pool (0 = observed-only). Opt-in; only helps
  insertion-restoration WINS, which pythia-70m mostly can't support, so likely leave at 0 for the battery.
- `bd_pool_cap` — cap on the bd candidate-pool size (bounds the O(Kc) `_ins_logq` cost); use ~20 if bridges on.

**What Phase 3 does.** Run the 40-item plausible/implausible battery (the same set as ALIGN_OPT_RESULTS) with
`gibbs` vs `gibbs+bd`, compare E (correction mass on implausible) / L (literal retention on plausible) / junk,
sweep `bd_p_stay` (~{0.1, 0.2, 0.3}), and recommend whether to promote a gibbs+bd default (and at what
`bd_p_stay`). Expectation: gibbs+bd should help the structurally-stuck duplicate/doubling family (`INS_DUP`,
which the substitution sweep cannot fix — see INS-02a) WITHOUT regressing L on clean plausible items (the stay
branch protects them). Watch the `INS_DUP` family especially.

**How to run.** The battery harness is `genjax_port.calibration_word_action_smc` (driven by
`planning/align_opt_full40.sh`: `NC_CHANNEL=align NC_REJUV=gibbs NC_ALIGN_SLOPE=-4.5 NC_ALPHA=200,2,2`, P=128,
subsample `planning/wa_alpha_subsample.txt`). It reads `NC_REJUV` (line ~107) and calls `W.run(...)` (line
~80). **Small task first:** add `NC_BD_P_STAY` (+ optionally `NC_BD_BRIDGE_J`/`NC_BD_POOL_CAP`) env reads and
pass them into that `W.run(...)` call; then run with `NC_REJUV=gibbs+bd`. E/L/junk analysis:
`genjax_port.calibration_battery_analyze`. Cost ~87 s/item at P=128 → ~1 hr per config (the perf commit made
this tractable). NB the battery is SYNTHETIC, not the reserved human hold-out ([[human-data-reserved-holdout]]).

**Caveats / known limits.** DEL-of-style insertion-restoration is a pythia-70m MODEL limit, not a move defect
(the LM scores `one the best` > `one of the best`); a better LM would flip it. Don't over-tune `bd_p_stay` on
the toy. The suffix-tail KV cancellation (the originally-planned perf win) is still available to stack on top
if the battery proves too slow, but is far more complex and lower-value now (§12).

**Throwaway harnesses** (recreate-able): `planning/bd_{toy_gate,toy_step3,live_gate,gate5}.py`; logs
`planning/bd_gate5{,_stay}.log`.
