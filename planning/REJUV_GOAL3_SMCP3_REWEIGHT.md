# Goal 3 — SMCP3 reweighting instead of MH accept/reject

## Problem this addresses
MH accept/reject has two failure modes for reanalysis:
1. **Discarded work.** On rejection the full LM forward is computed and thrown away — wasted
   compute proportional to the reject rate.
2. **Cannot move mass (the bigger one).** MH is weight-preserving. A particle stuck in the wrong
   reading that *rejects* the corrective flip stays wrong **and keeps its weight**. MH-rejuvenation
   can only diversify within what survives; it cannot transfer probability mass from a wrong reading
   to a right one. This is a plausible cause of "rejuvenation doesn't consistently improve
   inferences": when the move is rejected (the common case under the old local proposal), nothing
   happens at all.

SMCP3 (always-apply the move, carry its weight, let resampling reallocate) fixes both: every
forward contributes to a kept particle, and a move that the suffix reveals to be good/bad gets
up/down-weighted and multiplied/culled at the next resample — **mass flows between readings**, which
MH structurally cannot do.

## Important scoping (read after Goal 2)
If Goal 2 makes the substitution flip a **near-Gibbs** move (acceptance ≈ 1), there is little
rejection waste left for SMCP3 to reclaim **on that move**, and a pure Gibbs move's SMCP3 weight is
≈ 0 (no reweighting). So SMCP3's real payoff concentrates on:
- **(a) Asymmetric / trans-dimensional moves** (add/delete), where you *can't* cheaply make the
  proposal exact and rejection waste is real. The hand-assembled SMCP3 weight already exists there
  (`rejuvenation_r2.add_delete_step`, `W = w_upd + s_bwd - s_fwd`, ~L187).
- **(b) Folding any move's weight into resampling** so the move can *reallocate mass*, not just
  diversify — the part that helps late-evidence inferences even when the within-particle move is
  exact.

Decide scope based on Goal 2's outcome: if the sub-flip is Gibbs, focus Goal 3 on (a)+(b); if you
keep a stochastic (non-exact) sub-flip, Goal 3 also reclaims its rejection waste.

## Why this is cheap to try now
The weight is **already computed**. `manual_subflip_move` computes the exact MH ratio `w`
(`rejuv_bridge.py` ~L471); for this K/L pair the MH ratio and the SMCP3 incremental weight are the
**same number**. The add/delete move already assembles its SMCP3 weight by hand
(`rejuvenation_r2.py` ~L187). So the conversion is: stop the coin flip (`accept = log u < w`), keep
the move unconditionally, and route `w` into the particle log-weights.

## Read these first
- `src/genjax_port/rejuv_bridge.py`: `manual_subflip_move` (~L431, the weight `w` ~L471, accept
  ~L472); `_make_aligned_subflip_hook` / `run_smc_conditional_rejuv_aligned` (~L598/~L646) — the
  hook is currently a **post-resample** hook.
- `src/genjax_port/smc_substitution.py`: `run_smc_substitution` (~L125) — the SMC loop. Note the
  per-word weight assembly `log_w` (~L220), `log_marginal` (~L222), and **resample** (~L227), and
  that the rejuv hook runs **after** resample (`post_resample_hook` ~L260). This placement matters.
- `src/genjax_port/rejuvenation_r2.py`: the existing hand-built SMCP3 weight for add/delete.
- `docs/model.tex` eq smcp3 / Thm 2 — `W = w_upd + s_bwd - s_fwd`.

## The placement change (the crux)
Today the move runs **after** the word's resample, so its weight cannot influence that resample —
mass can't flow. For SMCP3 to deliver benefit (b), the move's weight must feed a resample. Two
options:
- **Preferred — move pre-resample, fold into `log_w`:** run the windowed move *before* the word's
  resample and add each particle's accumulated SMCP3 `w` to `log_w` (smc_substitution ~L220) before
  the `jax.random.categorical` resample (~L227). The move's verdict then immediately culls/multiplies
  — exactly the mass-flow benefit. This is a structural change to the hook contract (currently
  `post_resample_hook`); introduce a `pre_resample_hook` (or a hook that returns `(buf, delta_logw)`).
- **Simpler — extra resample after the move:** keep the post-resample hook but follow it with a
  second resample on the move weights. More compute (two resamples/word); use only if the
  pre-resample refactor is too invasive for a first cut.

## Implementation steps
1. Add a flag (e.g. `rejuv_mode="mh" | "smcp3"`) to `run_smc_conditional_rejuv_aligned`. Default
   stays `mh` until validated.
2. In SMCP3 mode, have the move **always apply** (`buf = buf_new`, drop the accept mask) and return
   the per-particle accumulated `w` (summed over window columns / sweeps) as a weight delta.
3. Wire the delta into resampling per the placement choice above. Ensure `log_marginal` accounting
   stays correct: an always-applied SMCP3 move contributes its mean weight to the marginal just like
   any SMC reweight — verify against the `@gen` oracle (`rejuvenation_r2` /
   `MaskCombinator.edit` path) on a small case where the weight is known exactly.
4. **Variance guard.** SMCP3 trades "reject bad moves" for "downweight bad moves"; a bad proposal
   injects high-variance weights and can collapse ESS. This only behaves with a good proposal and a
   resample right behind it — so **do Goal 2 first**. Add per-step ESS-after-move logging; if ESS
   collapses, the proposal (Goal 2) is the fix, not clamping the weights.

## Validation
- **Marginal correctness:** on a small fixed case, the SMCP3 run's `log_marginal` must match the
  oracle (the `@gen` `Update`/`MaskCombinator.edit` weight from `rejuvenation_r2`) to float-tiling
  noise (~1e-2, as elsewhere in the port).
- **Mass flow (the point):** construct/keep a late-disambiguation case where exactly one particle
  carries the soon-to-be-correct reading. Under MH it should fail to spread; under SMCP3 +
  resample, mass should concentrate on the corrected reading. Assert the posterior difference.
- **ESS:** log per-step ESS-after-move; SMCP3 must not systematically collapse it vs. MH.
- **Eval:** `eval_rejuv.py` mean under SMCP3 ≥ MH. If not, inspect whether the proposal (Goal 2) or
  the placement (pre- vs post-resample) is the cause before abandoning.

## Done when
SMCP3 mode reproduces the oracle marginal, demonstrably moves mass on the late-evidence case where
MH cannot, holds or improves `eval_rejuv.py`, and does not collapse ESS. Keep MH selectable for
regression comparison. Update `run_example_native.sh` if SMCP3 becomes the default path (see the
`keep-run-example-script-current` memo).
