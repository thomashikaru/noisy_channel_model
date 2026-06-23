# Gibbs insertion/deletion rejuvenation — results

Branch `rejuv-birth-death` (off `align-action-channel`, transposition-fix merged in `f9ca4b9`).
Goal: handle the noisy-channel model's insertion/deletion failures (missing PP / function words,
spurious insertions) **without over-editing clean sentences**. Compute treated as scarce
(pythia-70m only, P≤128, battery subsets).

## The problem and the diagnosis

Baseline `rejuv=gibbs` (substitution-only sweep) on the deployed align channel (K=−4.5, α=(200,2,2),
P=128, seed 0, the 40-item battery `planning/align_opt_full40_K4.5_a2.txt`): **keeps 20/20 (L=0.95),
edits 7/20 (35%)**. The failing edits are mostly **deletion-restoration** (a dropped function word):
DELFROM, DELTO, DEL-the, DEL-a, LADDER → E≈0.

**Are the failures inference-limited or signal-limited?** `planning/indel_signal_check.py` computes the
true joint `lm_temp·logP_LM(intended|prime) + channel-action-cost` for the literal vs the target reading
(deletion = one DEL action, no content cost; spurious-insertion removal = one INS action that pays the
removed word's `−unigram_surprisal`, which my first pass wrongly omitted). Verdict:

| item | kind | JOINTΔ (target−literal) | gibbs E | status |
|---|---|---|---|---|
| DELFROM-01a | restore `from` | **+4.50** | 0.00 | INFERENCE-limited (winnable) |
| DELTO-02a | restore `to` | **+4.41** | 0.00 | INFERENCE-limited |
| DEL-of-01a | restore `of` | +2.85 | 0.70 | already handled by the forward filter |
| DELFROM-02a | restore `from` | +1.77 | 0.12 | INFERENCE-limited |
| LADDER-send-2 | restore `to` | +1.37 | 0.15 | INFERENCE-limited |
| DELFOR-01a | restore `for` | +0.60 | 0.00 | marginal |
| DEL-the-01a, DEL-a, DELTO-01a, LADDER-give-2 | restore | ≤ 0 | ≈0 | **signal-limited — literal is correct, must NOT edit** |
| INS-01a `handed handed`, INS-to-* | remove spurious | < 0 | low | **signal-limited** (insertion content cost > LM gain) |

So there is genuine **inference-limited headroom** (~5 deletion-restoration items the true posterior
prefers to edit but SMC misses), and the over-editing constraint is exactly the signal-limited set.

## Why the existing birth/death move failed, and the two new moves

The legacy `bd_mode="smcp3"` move (Phase 3, rejected) ALWAYS applies a birth/death and folds a trans-dim
importance weight; its weight-variance collapses the cloud and **over-edits clean sentences**.

* **`bd_mode="mh"` (Gen.jl `Gen.mh` design).** `W = _bd_log_weight` IS the log MH acceptance ratio, so
  accept the proposal w.p. `min(1,e^W)` and inject NO weight (`move_logw=0`). A bad move on a clean parse
  is rejected → **cannot over-edit** (certified: `test_mh_accept_reject_and_zero_weight`; INS-01a held).
  But it proposes ONE word at ONE **uniform** gap, so restoration is weak and a content word at a
  locally-fluent wrong gap slips in as junk (DELFROM with bridges: E=0.04 at 1 attempt → junk 0.96 at 6).

* **`bd_mode="gibbs"` (DEFAULT, the effective move).** `gibbs_indel_move` resamples the **single edit**
  from its full conditional over `{no-op} ∪ {insert c@gap g} ∪ {delete word i}` ∝ `π(resulting parse)`.
  Because no-op is in the set, the conditional self-regulates:
  - a CLEAN parse draws no-op w.p. ~1 (every edit lowers π) → cannot over-edit;
  - a DROPPED-word parse draws the restoring insertion (the conditional concentrates on the one high-π
    edit) → restores it;
  - a spurious π-lowering insertion gets ~0 mass → no junk (the true joint confirms `from` beats the junk
    readings by 6–15 nats: `planning/delfrom_joint.py`).
  `move_logw=0` (Gibbs preserves the target). Fired ONCE post-loop over the all-done cloud, `bd_attempts`
  sweeps (each sweep restores ~15% of the cloud → ~5 sweeps reach the posterior fraction).

## Results (gibbs+bd, gibbs mode, P=128, K=−4.5, α=(200,2,2), bridges_j=4, post-loop attempts=5)

| item | role | gibbs baseline | gibbs+bd | verdict |
|---|---|---|---|---|
| DELFROM-01a | restore `from` (target) | E=0.00 | **E=0.61, q_smc=0.88 PASS** | restored ✓ |
| INS-02b | clean keep | L=0.80 | **L=1.00, junk=0.00** | held (improved) ✓ |
| DEL-the-01a | signal-limited (literal correct) | E≈0.00 | **E=0.07** | correctly NOT edited ✓ |

This is the goal: a substantial increase in target behaviour on a genuinely inference-limited
deletion-restoration case, with NO over-editing of the clean sentence and NO spurious editing of the
signal-limited case. (Further confirmatory items in `planning/bd_gibbs_confirm.log`.)

## Perf / caveats

* The DEFAULT `rejuv=gibbs` is **unchanged / bit-identical** — `gibbs+bd` is opt-in. So the deployment
  default is not slowed.
* `gibbs+bd` (gibbs mode) is ~250 s/item at P=128 (vs ~15–30 s for `gibbs`): the move scores a
  `O(Wmax·Kc)` candidate grid per sweep × ~5 sweeps. Scoring is **sequential at the native P batch** — a
  full `Kc·P` batch was tried and **thrashed 32 GB** (each forward materialises a `Kc×` larger vocab-logit
  tensor), so it is reverted. A memory-bounded chunked-batch scorer is the obvious future perf win.
* Restoration needs `NC_BD_BRIDGE_J>0`: the dropped word is not an observed surface, so the move needs the
  LM-bridge candidates in its insertion pool. `bridge_j=0` only re-inserts observed words (duplicate
  removal). `bridge_j=4` covers the function-word targets here; too small (j=3) drops `from`/`to`.
* Signal-limited items (DEL-the, INS-01a `handed handed`) are pythia-70m LM limits, not move defects — a
  better LM would flip them. The move correctly leaves them literal.

## Knobs (`NC_*` in `calibration_word_action_smc`; args in `pythia_word_caprop.run` / `pairhmm_smc.run`)

`NC_BD_MODE` = gibbs (default) | mh | smcp3 · `NC_BD_BRIDGE_J` (LM-bridge insertion candidates, default 0)
· `NC_BD_POOL_CAP` · `NC_BD_ATTEMPTS` (post-loop Gibbs sweeps, default 1; ~5 for full restoration).

Throwaway harnesses: `planning/indel_signal_check.py`, `planning/delfrom_joint.py`,
`planning/bd_gibbs_*.log`.
