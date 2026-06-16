# R2 — add/delete reversible-jump rejuvenation (design + build plan)

> Picks up from `MIGRATION_PLAN.md` (M5/R1 done; R2 is NEXT) and `REJUVENATION_PLAN.md` §6.2/§11
> and `docs/model.tex` §`sec:moves` (R2 paragraph). R1 (`rejuvenation.py`) is the template: a
> standalone, non-vmapped, unrolled per-word chain validated by detailed balance, *then* a vectorized
> bridge in a later phase (`VECTORIZED_REJUV_PLAN.md` Phase 2). R2 follows the same shape.

## What R2 adds

R1 flips a word's intended token (fixed word count). R2 is **trans-dimensional**: it inserts an
omitted intended word (`add`, = reversing a deletion) or removes a posited one (`delete`), changing
the length of the intended sentence `z`. This lets rejuvenation *reanalyse the alignment* — e.g. the
filtering sweep committed to "no omission here", but later context shows a word was dropped, so `add`
recovers it. Mirrors Gen.jl's `rejuv_proposal_add_delete` + `involution_add_delete`
(`src/gen_inference.jl:193-284`).

## The representation: a MASKED deletion-gap chain (keeps the address set FIXED)

The naive view "the trace changes dimension" is the wrong mental model for genjax. Instead use a
**fixed-max-length masked chain** so every address always exists and only a per-gap boolean (plus the
threaded buffer) changes — which is what makes `Update`/`Rejuvenate` apply and (later) vmap.

Per observed word `t` (`make_gap_chain`, prototyped in `/tmp/genjax_spike6_maskflip.py`):

| address | dist | meaning |
|---|---|---|
| `del{t}` | `genjax.flip(p_del)` | is there an omitted intended word *before* observed word `t`? |
| `gap{t}/xd` | `mask(lm_token)` | the omitted token (scored only when `del{t}` true; masked → score 0) |
| `x{t}` | `lm_token` | the intended token aligned to observed word `t` |
| `o{t}` | `obs_dist(x, cand)` | the channel (observed token, constrained) |

The LM **buffer is threaded deterministically from the choices** inside the `@gen` body:
`buf = where(del{t}, buf.at[il].set(xd), buf); il += del{t}; … ; buf.at[il].set(x{t}); il += 1`.
So flipping any `del{t}` shifts the buffer for the whole suffix ⇒ all later `x` LM terms re-score
automatically through the `Update`. (Scope as in M2: one omitted token per gap, `MAX_DELETIONS=1`.)

## Keystone result (spike 6, 2026-06-15, pythia-70m) — DE-RISKED ✅

`genjax.mask`'s `MaskCombinator.edit` (`combinators/mask.py:214-253`) supplies the reversible-jump
birth/death weight **for free**:
- **False→True (ADD)**: `final_weight = final_trace.get_score()` — full score of the born slot.
- **True→False (DELETE)**: `final_weight = -original_trace.get_score()` — minus the old score.
- Jacobian unity (discrete overwrite), exactly model.tex R2.

Verified on the masked-gap chain: an `Update` flipping `del{k}` returns `w_upd` **equal to the full
`logp(t')-logp(t)`** — i.e. `Bernoulli(del) prior Δ + LM(omitted token) + suffix re-score Δ`, all
assembled by the library — and the delete is exactly `-add`, round-tripping bit-exactly to the literal
score. So **add/delete is a fixed-address masked edit; no manual trans-dimensional/Jacobian math.**

## The move math (manual SMCP3 — why not plain `Rejuvenate`)

R1 used genjax `Rejuvenate` (one proposal `Q` for both forward `K` and backward `L`). R2 **can't**:
the forward (add) draws a token, the backward (delete) draws none — the proposals are *asymmetric*
over different address sets. So assemble the SMCP3 weight by hand (this is also what Gen.jl does via
`involution_add_delete`), using `model.edit(Update(...))` for the free `w_upd`:

```
W = w_upd + s_bwd - s_fwd                     # docs/model.tex eq (smcp3)
```
Per gap `k`, forced-toggle (binary slot ⇒ exactly one move type available per state; the two are
mutual reverses, so no add-vs-delete coin needed — note the Gen.jl 0.5 coin generalises to multi-slot):

- **ADD** (`del{k}` F→T): propose `xd ~ q(·)` = the deletion-gap proposal (LM top-`LOOKAHEAD_K`
  reweighted by one-step lookahead toward the next observed token — reuse `smc_substitution.deletion_gap`'s
  proposal). `s_fwd = log q(xd)`; reverse move (delete) draws no token ⇒ `s_bwd = 0`.
- **DELETE** (`del{k}` T→F): `s_fwd = 0`; reverse (add) would propose exactly the removed token ⇒
  `s_bwd = log q(xd_removed)`.

Consistency: `W_delete(t') = -W_add(t)` for the reverse pair ⇒ detailed balance (Thm 2). MH-accept
`log u < W` (default), or fold `e^W` as SMCP3 reweight.

## Build steps (this is `rejuvenation_r2.py`, mirroring `rejuvenation.py`)

1. `make_gap_chain(W, p_del)` + `gap_chain_inputs(obs_ids)` (candidate tables; buffer seed). ✅ proto.
2. `_gap_proposal_sample/_logpdf` — the LM-lookahead omitted-token proposal `q` (factor out of
   `smc_substitution.deletion_gap` so the sweep and the move share it; the keystone parity from the
   bridge).
3. `add_delete_step(key, tr, k, ...)` — pick available move from `del{k}` state, build the `Update`
   choicemap, get `w_upd` from `model.edit`, add `s_bwd - s_fwd`, MH-accept, `tree_map(where)` select.
4. `add_delete_sweep(...)` over gap positions × `n_sweeps`.

## Validation (gate)

- **Detailed balance** on a 2-word toy: MH stationary histogram over `{del configs}` == brute-force
  exact posterior (enumerate the `2^W` gap on/off configs × candidate tokens) to ≤1e-3. (R1 has the
  analogous test in `tests/test_rejuvenation.py`.)
- **Reanalysis behaviour**: a sentence with a genuinely omitted word (e.g. "he wants go home" with the
  "to" dropped) where the *substitution-only* chain can't recover it but an `add` sweep inserts "to".
  Show the add move's MH weight is ≈0 without the disambiguating suffix and strongly positive with it.
- Reverse: a spuriously-posited deletion is removed by a `delete` move.
- Add tests to `tests/run.py` (pythia-70m), like R1.

## After R2: vectorization (Phase 2 of `VECTORIZED_REJUV_PLAN.md`)

The standalone move proven, the bridge vectorizes it over particles. Because the representation is
already fixed-address + masked, the batched trace stays rectangular (the `del{t}`/`gap{t}` arrays just
gain a leading P axis) — the ragged-`W` problem is absorbed into the mask. That is the coupled
"R2 move math + Phase 2" the plan flagged; do it only after the gate above is met.

## Status

- [x] Keystone de-risk (spike 6): masked add/delete `Update` weight == RJ birth/death incl. suffix.
- [x] `rejuvenation_r2.py` (masked-gap model + LM-lookahead proposal + `add_delete_step`/`_sweep`).
- [x] Tests in the runner (`tests/test_rejuvenation_r2.py`, pythia-70m): reanalysis (add recovers the
  omitted "to" → "he wants to go home"), suffix-participates, and **detailed balance** (add/delete MH
  histogram matches the enumerated exact posterior to ≤0.07 — Thm 2 / R2 in practice). All green.
- [ ] **NEXT — Vectorized bridge (Phase 2 of `VECTORIZED_REJUV_PLAN.md`):** vmap the standalone move
  over particles and interleave it into the filtering sweep (couple with R3's surprisal gate). The
  representation is already fixed-address + masked, so the batched trace stays rectangular (the
  `del{t}`/`gap{t}` arrays gain a leading P axis); ragged `W` is absorbed by the mask. Then wire
  `--add_delete` into `run.py`. Also: R2 currently runs as a second pass on its own chain (like R1's
  `rejuvenation.py` did before `rejuv_bridge.py`); the bridge unifies it with the substitution sweep.

  **RECONCILE with `rejuv_model.py` first (important — avoid two parallel masked chains).** That
  module (Phase 2b, commit f0f086b) is a **vmappable** masked *autoregressive* carrier:
  `present_k ~ flip(p)` gates `mask(_token_slot)`, buffer threaded over active slots only,
  `importance == manual joint` **and** vmap-over-particles both verified (`tests/test_rejuv_model.py`).
  It already solved the hard Phase-2 infra (masked AR + vmap + weight parity) — but has **no channel
  (`o`), no proposal, and no move**. This module (`rejuvenation_r2.py`) is the complement: the actual
  add/delete **move** (proposal `q`, SMCP3 weight, MH step, detailed balance) on a channel-bearing
  gap-chain, **non-vmapped**. Two representational choices to merge: `rejuv_model`'s uniform
  "every intended token is a toggleable `present` slot" vs this module's "a `del{t}` gap before each
  always-present observed word". Phase 2 = port the (representation-agnostic) move math here onto a
  channel-augmented, vmapped `rejuv_model` chain — not a third rewrite. The keystone (`MaskCombinator`
  birth/death weight) and the move math are identical for both layouts.
