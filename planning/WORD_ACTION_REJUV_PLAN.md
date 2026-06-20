# Restore word-substitution rejuvenation to the word-action path

**Status:** Phase 0 + Phase 1 + Phase 2 + Phase 3 DONE (Phase 3 on 2026-06-20, branch `word-action-rejuv`).
The boolean fork is gone; channel scoring has one source of truth selected by a named `channel`. **Only the
word-action battery spot-check remains** (Phase 2/4 owed; needs Pythia) before the branch is ready to compose
with the α re-tune + the §1b production default-flip. Goal (ACHIEVED for the inference path): the
post-resample SMCP3 **word-substitution** rejuvenation move
(R2/R3/R4 — the impoverishment cure that fixes "P=128 flips a correct word to a wrong neighbour") runs on the
**word-action channel**, which is now THE model. It used to not: when the word-action channel was active the
filter ran a θ-refresh **instead of** the word sweep — Phase 2 made the sweep θ-aware and switched the filter to
**sweep-then-refresh**.

## 0. Root cause — duplication + a boolean fork (the thing to fix, not work around)
The channel forward-carry exists in **two** copies that drifted:
- forward filter, θ-aware: [`_channel_carry_action`](../src/genjax_port/pairhmm_smc.py#L415) — emission column
  `emit_form[:,s] + lp_sub + (lp_copy−lp_sub)·copy_mask[:,s]`, **per-particle** `wdel_p`/`wins_p` from θ.
- rejuv sweep, θ-blind: [`_channel_carry_a`](../src/genjax_port/pairhmm_rejuv.py#L190) — bare `emit_full[:,s]`,
  **global** scalar `wdel`/`wins`, no action split.

Because the sweep's scorer can't see θ, firing it on word-action particles would score every candidate against the
wrong channel — so it was gated off with a boolean `ON = action_alpha is not None` threaded through `run()`. **The
fork is the disease; the missing rejuvenation is the symptom.** The fix is to collapse the two carries into one
source of truth that both the filter and the sweep call, so "θ-aware" is automatic and the two can never drift
again — then delete the boolean.

## 1. Development principles for this work (per user, 2026-06-19)
- **Branch.** Do all of this on a dedicated git branch (`word-action-rejuv`), committing each phase so any step is
  revertible. Do not develop the new path as a fork living next to the old one in `run()`.
- **No simultaneous ON/OFF fork.** Unify the channel scorer into ONE shared component. The deprecated char-copy
  channel becomes one *implementation behind the same interface* (kept only as the exact-enumeration certification
  anchor), not a parallel `else` branch sprinkled through the filter.
- **Meaningful names.** Retire the non-informative boolean `ON`/`OFF`. Select the channel by a named value
  (`channel="word_action"` default; `channel="char_copy"` = deprecated/certification-only).
- **Pin before it lands.** Every new path is pinned against a reference *first*: OFF stays bit-identical
  (`test_pairhmm_exact`), and a new toy word-action gate proves the θ-aware sweep is correct, before the gate flips.
- **Keep char-copy's good features.** The case-insensitive char DP and the exact-enumeration anchor are genuinely
  good; they survive behind the unified interface — we deprecate the *path forking*, not the capability.

## Progress log (RESUME HERE)

**Branch:** `word-action-rejuv`, off the `genjax-port-unified` baseline. Commits so far:
- `dc16bd5` — Phase 0 baseline: consolidate genjax_port (all model code).
- `6bde1b5` — Phase 0 baseline: calibration substrate + planning docs.
- `4fbff44` — **Phase 1 (de-fork) DONE.** Unified the two duplicated channel carries into one
  `word_dp.channel_carry`, called by both the filter's theta-refresh and the sweep. Char-copy = the
  zero-action degenerate (`lp_copy=lp_sub=0`). Verified **bit-identical**: `test_pairhmm_exact` caprop logZ
  −7.955/−7.995/−9.230, TV 0.261/0.259/0.118; full live suite 18/18. (`sweep`'s `_channel_carry_a` is now a
  thin zero-action adapter into `channel_carry`.)
- **Phase 2 (θ-aware sweep + restore the move) DONE (2026-06-20).** Three changes, OFF bit-identical at each:
  1. `pairhmm_rejuv`: `sweep` gained an optional `theta_costs=(lp_copy, lp_sub, wdel_p, wins_p, a0p,
     copy_mask)` tuple, threaded as TRACED args into the jitted `step`/`move`; `_chan_scores` and the final
     `log_alpha` recompute now call `word_dp.channel_carry` directly (each per-particle cost `jnp.repeat(…,
     Kt)`-aligned to the P·Kt spliced rows). `theta_costs=None` builds the zero-action char-copy
     parameterization from `ctx` → bit-identical. The Phase-1 `_channel_carry_a`/`_channel_carry` thin
     adapters are now DELETED (their job is the zero-action default inside `sweep`).
  2. `pairhmm_smc.run`: dropped `and not ON` at the sweep-build guard, and replaced the resample-branch
     `if rj_sweep / elif rj_theta` with **sweep-then-refresh** — the θ-aware word sweep under the current θ
     (passing `theta_costs` assembled from the gathered per-particle costs + a fresh `a0p`), THEN the
     Dirichlet-conjugate θ-refresh on the corrected post-move parse.
  3. `tests/test_pairhmm_exact.py`: added the toy word-action ON gates — `test_wa_rejuv_*` (invariance,
     collapse-recovery, SMCP3 weight ≈0) against an exact word-action enumerator at the concentrated-α
     limit, plus `test_wa_run_gibbs_end_to_end` (the full `run` path with `action_alpha` + `rejuv="gibbs"`).
  - **Guards GREEN:** full toy suite **17/17** (13 OFF unchanged + 4 new). Live suite: see commit.
- **Phase 3 (retire the `ON`/`OFF` boolean → named `channel` selector) DONE (2026-06-20).**
  1. `pairhmm_smc.run` gained `channel="word_action"|"char_copy"` (default `None`). `ON = action_alpha is
     not None` is retired → `ON = channel == "word_action"`. `channel=None` infers the channel from
     `action_alpha` (set → word_action, None → char_copy) — a back-compat shim that makes the rename
     bit-identical for every existing caller with no edit. Contradictory combos now RAISE (word_action with
     no `action_alpha`; char_copy with one; unknown name) instead of silently mis-scoring.
  2. `pythia_word_caprop.run` gained the same `channel` param (defaults `action_alpha` →
     `ACTION_ALPHA_DEFAULT` when word_action is selected without one) and threads it down. The CLI boolean
     `--word_action` is REPLACED by `--channel {word_action,char_copy}` (default `char_copy`; an explicit
     `--action_alpha` still implies word_action). Both `run()` docstrings now state char_copy's
     exact-enumeration certification-anchor role.
  3. **DECISION — no default-flip in this phase.** The §1b production default-flip (make word_action the
     deployed default) is gated on the unfinished α re-tune (CODE_CONSOLIDATION_PLAN §1b /
     WORD_ACTION_ALPHA_SWEEP_PLAN — `(3,1,1,1)` is edit-happy), so the *effective* default is unchanged
     (char_copy when nothing is specified). Phase 3 is purely the named-selector rename.
  4. **Guards GREEN:** full toy suite **19/19** (17 unchanged, all bit-identical + 2 new `test_channel_*`
     gates pinning the pure-rename equivalence and the validation contract). CLI `--help` smoke clean.
  - **`run_example_native.sh` (skip-worktree, NOT committed):** its `WA_ARGS` line uses the now-removed
     `--word_action`; the user must update it — change `WA_ARGS=(--word_action)` →
     `WA_ARGS=(--channel word_action)` (flagged to the user, per [[keep-run-example-script-current]]).

**Only owed item (Phase 2/4 done-criteria):** a **word-action battery spot-check** (a couple of calibration
items) showing the restored sweep recovers an early dropped word that θ-refresh-alone left uncorrected — the
live behaviour this buys. Run once Pythia is loaded (heavier than the toy gates). Then this branch composes
with the α re-tune + the §1b default-flip.

## 2. Phased plan

### Phase 0 — Baseline + branch
- Commit the current working tree first (the consolidation + the pre-existing uncommitted word-action work) so the
  branch has a clean, named baseline to revert to. (Commit is the user's call.)
- `git switch -c word-action-rejuv`.

### Phase 1 — Unify the channel carry (de-fork; no behaviour change)
- Promote the θ-aware carry to a single shared home (extend `word_dp.py`, or a new `channel_scoring.py`) as **the**
  channel forward-carry. Parameterize it so the deprecated char-copy case is a degenerate call (no action offset →
  `lp_copy=lp_sub=0`, `copy_mask` all-ones-or-irrelevant, global `wdel`/`wins`), i.e. one function, two
  parameterizations — not two functions.
- Repoint the forward filter to the shared carry; **delete** the sweep's duplicate `_channel_carry_a` and repoint
  [`_chan_scores`](../src/genjax_port/pairhmm_rejuv.py#L312) and the sweep's final-`log_alpha` recompute
  ([pairhmm_rejuv.py:554](../src/genjax_port/pairhmm_rejuv.py#L554)) to it.
- **Guard:** `test_pairhmm_exact` (incl. rejuv + multi-token gates) bit-identical. This phase changes structure
  only, not numbers.

### Phase 2 — θ-aware sweep + restore the move
- Add per-particle θ-costs `(lp_copy, lp_sub, wdel_p, wins_p, a0p)` + `copy_mask` as **traced args** to
  [`sweep`](../src/genjax_port/pairhmm_rejuv.py#L527) / the jitted `step`/`move` (mirroring how `emit_full`/`a0`/
  `wins` are already threaded so the compile is reused). The Kt candidates per particle take `jnp.repeat(…, Kt)` of
  each particle's costs. When the costs are absent, the shared carry runs the char-copy parameterization → OFF
  unchanged.
- In `run()`: build the sweep for the word-action channel, and run **word-sweep then θ-refresh** each resample
  event (replacing the `if/elif`). Rationale: the θ-refresh alone can only re-estimate θ on a *frozen* parse (the
  documented deletion mode-collapse); the word move can *change* the parse — restore a dropped word — after which
  the refresh re-estimates θ on a corrected parse. Sweep-then-refresh gives the particle that escape route. (See §3
  decision.)
- The SMCP3 weight machinery ([`_smcp3_move`](../src/genjax_port/pairhmm_rejuv.py#L272)) is target-agnostic — no
  change; the full-conditional weight should still collapse to `≈0` (built-in self-check that the target is right).
- **New ON certification gates** (toy, exact): a word-action analog of `test_rejuv_leaves_exact_posterior_invariant`
  (seed a cloud at the toy word-action posterior with `action_alpha` set → sweep leaves MAP/posterior invariant,
  weight ≈0) and of `test_rejuv_recovers_collapsed_cloud` (a cloud collapsed onto a wrong neighbour is pulled back),
  at the concentrated-α limit. The toy already has the FORM-channel mirror needed for word-action scoring.

### Phase 3 — Retire the `ON`/`OFF` boolean
- Replace `ON = action_alpha is not None` with an explicit named `channel` selector on `run()` (default
  `"word_action"`; `"char_copy"` deprecated, used by the certification anchor + opt-out). Update the CLI flags and
  `run_example_native.sh` to the named selector. Document char-copy's role as the exact-enumeration anchor in the
  `run()` docstring.

## 3. Decided
- **Refresh/sweep ordering: SWEEP-THEN-REFRESH** (user, 2026-06-19). Each resample event runs the θ-aware word
  moves under the current θ, then refreshes θ from the corrected post-move parse — giving the particle the escape
  route from the θ mode-collapse that refresh-alone lacks. The toy collapse-recovery gate must confirm it actually
  escapes (a dropped word gets restored, not just re-θ'd).

## 4. Guards / done criteria
- OFF path bit-identical at every phase (`test_pairhmm_exact`).
- New toy word-action sweep gates green (invariance + collapse recovery + SMCP3 weight ≈0).
- A word-action battery sanity check (a couple of the calibration items) shows the restored word sweep recovers an
  early dropped word that θ-refresh-alone left uncorrected — the concrete behaviour this is meant to buy.
- Lands as a reviewable branch, each phase a commit; the boolean fork is gone, channel scoring has one source of
  truth. Then this composes with the α re-tune (`planning/WORD_ACTION_ALPHA_SWEEP_PLAN.md`) and the §1b default-flip.
