# Vectorized rejuvenation plan (TODO #1: rejuvenation integrated into the real filter)

## Context

Bridge v1 (`rejuv_bridge.py`) connects the filtering sweep to the R1 rejuvenation move, but as a
**post-sweep, per-particle Python loop** — a stopgap. The point of the genjax port is to **vectorize
over particles** so the penzai forward batches (the ~6×-not-P× win); a per-particle loop forfeits
exactly that. The Gen.jl reference (`src/gen_inference.jl`) supplies the high-level ideas only
(conditional/surprisal-gated + second-pass rejuvenation, lookback window) — its `@threads` +
per-particle `Gen.mh` execution model is what we are moving away from. Resampling is **every word**
(settled; `ESS_THRESH = Inf` in `config.jl:122` means the reference also always resamples) — no
adaptive-resampling change.

Goal: rejuvenation that runs **inside** the every-word SMC loop, **vmapped across all P particles**,
surprisal-gated with a lookback window, so early-commitment errors are corrected mid-sweep before the
next resample — at batched-forward cost.

## Phase 0 — de-risk: vmap `Rejuvenate.edit` over particles ✅ DONE

Spike (`/tmp/genjax_spike5_vmap_rejuv.py`, pythia-70m, `make_chain_model`/`rejuv_step` from
`rejuvenation.py` **unchanged**):
- **(A)** `jax.vmap` of both `materialize` (`model.importance`) and `rejuv_step` over the P axis RUNS
  — no Python control-flow break in `StaticRequest`/`Rejuvenate.edit` under vmap.
- **(B)** It **batches the forward**: a full W-position sweep costs P=64 → 4.3× of P=1 (not 64×). The
  batching win holds for the rejuvenation move, exactly as for the filtering-sweep forwards.
- **(C)** vmap result == per-particle loop, **bit-for-bit** (P=8, shared keys).

Conclusion: the vectorized path is viable and the existing R1 functions already vmap. Everything below
builds on `jax.vmap(rejuv_step)`.

## Phase 1 — vectorized conditional (interleaved) rejuvenation

Scope inherits bridge v1: single-token words, substitution-only, fixed `W` ⇒ every particle's chain
trace is **structurally identical** (addresses `x0..x{W-1}`, `o0..o{W-1}`, scalar), so a batched trace
is just those arrays with a leading P axis — the regime vmap wants.

**Mechanism (materialize-on-demand, vmapped).** Keep the fast deduped buffer sweep
(`smc_substitution.run_smc_substitution`) as the filtering backbone. Inside the word loop, *after* the
(every-word) resample at word `t`:
1. **Surprisal gate.** We already compute `log_mean_weight = logsumexp(log_w) - log P` per word; set
   `surprisal = -log_mean_weight`. Form `cond_rejuv_p = custom_sigmoid(surprisal - unigram_surp,
   logprob_thresh, logprob_spread)` (mirrors `gen_inference.jl:409`). The reference draws a
   per-particle `Gen.bernoulli(cond_rejuv_p)`; vectorized that is a **`[P]` Bernoulli mask**.
2. **Materialize** the batched chain trace over the **lookback window** `[max(0, t-lookback) .. t]`
   via `jax.vmap(model.importance)` over `(key[P], x_window[P,w], cand_ls[P,w,K])` (buf0 seeded from
   the committed prefix before the window; `cand_xs`/buffer shared). Reuses
   `rejuv_bridge._word_candidate_tables` (already matches the sweep's evidence — the keystone).
3. **Move.** `jax.vmap(rejuv_step)` over the window positions (BACKWARD/FORWARD/shuffle order, an
   arg). The MH accept is a `[P]` mask; combine with the gate mask so ungated/rejected particles are
   unchanged (`tree_map(where(gate & accept, new, old))`).
4. **Write back** decoded window tokens into `intended_buf`, continue the sweep.

Because the gate fires on a minority of (surprising) words, the batched-trace path is hit rarely; the
common path stays the deduped vmapped buffer sweep.

**New dependency — unigram surprisal baseline.** Reference uses `unigram_probs`/`get_vocab_idx`
(`gen_inference.jl:408`); the port has none. Start with an absolute surprisal threshold (no unigram
term), add a unigram table later if the gate needs calibration.

**Refactor.** Replace bridge v1's per-particle loop in `rejuvenate_particles` with the vmapped move
(proven in Phase 0). Second-pass rejuvenation (bridge v1) then becomes the special case: gate forced
on, window = whole sentence — so one vmapped primitive serves both the `--rejuvenate` post-pass and
the new interleaved mode.

**Files:** `smc_substitution.py` (interleave hook + surprisal in the loop; likely a
`rejuvenate=...`/`lookback=...`/gate-params arg), `rejuv_bridge.py` (vmapped
materialize+move+writeback primitive, windowed; replaces the loop), `rejuvenation.py` (unchanged or
minor — already vmaps), `run.py` (`--conditional_rejuv`, `--lookback`, gate-param flags), tests.

**Validation:** (i) vmapped move == loop (carry the spike's parity check into the suite, small P);
(ii) interleaved-with-gate-off == sweep-only (identity); (iii) the gate fires on high-surprisal words
(behavioral, print surprisal/`cond_rejuv_p` like the reference); (iv) timing shows batched cost
(P-sweep ratio ≪ P). Correctness of the move itself is already covered (R1 detailed balance + Phase 0
parity + the bridge keystone).

## Phase 2 — vectorized trans-dimensional moves (couple with R2)

Add/delete changes `W` per particle ⇒ ragged traces ⇒ breaks the homogeneous batching Phase 0/1 rely
on. The fix is **fixed-max-`W` padding + the `Mask` combinator** so the batched trace stays
rectangular with inert slots, and the reverse-jump bookkeeping runs masked. This is one coupled
problem with R2's move math — do not start until R2 (`rejuv_proposal_add_delete` analog) exists. This
is the genuine frontier; Phase 1 deliberately stays dimension-preserving.

## Sequencing
Phase 0 ✅ → Phase 1 (vectorized conditional sub-flip; folds in bridge v1 as the post-pass special
case) → R2 move math → Phase 2 (vectorized trans-dimensional). Throughout: vmap over particles;
resample every word.

## Verification (env)
```bash
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh && conda activate ncgenjax
export TOKENIZERS_PARALLELISM=false
NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python -m src.genjax_port.tests.run
NC_LM=EleutherAI/pythia-410m PYTHONPATH=. python -m src.genjax_port.run \
  --filter native --conditional_rejuv --lookback 4 --particles 64 --sentence "<single-token sentence>"
```
