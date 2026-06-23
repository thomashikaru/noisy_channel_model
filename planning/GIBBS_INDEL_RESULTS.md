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

## The function-word coverage fix (pool part 3)

The top-J LM bridges are the LM's NEXT-token prediction at a gap, so a dropped function word is often NOT
in the pool even when the full-sentence joint wants it. DELTO-02a: after "soup" the LM ranks `and` above
`to`, so the move restored "served the soup AND the customers" (junk 0.40) instead of the target `to` —
even though the joint prefers `to` by ~1.6 nats (`planning/delfrom_joint.py`-style LM check: TO −49.91 >
AND −51.49 > literal −56.91). Fix: a fixed **closed-class function-word set** (articles / prepositions /
conjunctions, single-token) ALWAYS in the insertion pool (`bd_funcwords`, default on). Function-word
omission is the dominant production error, so this is a principled channel prior, NOT battery tuning — it
guarantees the targets are in the pool and lets the move's full-sentence conditional pick CORRECTLY among
them. With it, DELTO restores `to` and the `and` junk drops to 0.11.

## Results (gibbs+bd, gibbs mode, P=128, K=−4.5, α=(200,2,2), bd_funcwords on, post-loop attempts=4–5)

| item | role | gibbs baseline | gibbs+bd | verdict |
|---|---|---|---|---|
| DELFROM-01a | restore `from` (target, JOINT +4.5) | E=0.00 | **E=0.58–0.61, q_smc=0.85–0.88** | restored ✓ PASS |
| DELTO-02a | restore `to` (target, JOINT +4.4) | E=0.00 | **E=0.56, q_smc=0.68** | restored ✓ PASS (funcwords) |
| INS-02b | clean keep | L=0.80 | **L=1.00, junk=0.00** | held / improved ✓ |
| DEL-the-01a | signal-limited (literal correct) | E≈0.00 | **E=0.07** | correctly NOT edited ✓ |

Edit pass **2/2** (q_smc>0.5), keep **1/1**, junk>0.5 on **0/3+1**. This is the goal: a substantial
increase in target behaviour on genuinely inference-limited deletion-restoration cases, with NO
over-editing of the clean sentence (even with the richer function-word insertion pool the Gibbs
full-conditional draws no-op w.p. 1) and NO spurious editing of the signal-limited case. Logs:
`planning/bd_gibbs_pl5.log` (bridges), `planning/bd_gibbs_fwjit2.log` (funcwords),
`planning/bd_gibbs_broad.log`.

## Perf / caveats

* The DEFAULT `rejuv=gibbs` is **unchanged / bit-identical** — `gibbs+bd` is opt-in. So the deployment
  default is not slowed.
* `gibbs+bd` (gibbs mode) is ~180 s/item at P=128 once warm (vs ~15–30 s for `gibbs`; first item pays a
  ~175 s JIT compile): the move scores a `O(Wmax·Kc)` candidate grid per sweep × `bd_attempts` sweeps.
  Scoring is **sequential at the native P batch** — a full `Kc·P` batch was tried and **thrashed 32 GB**
  (each forward materialises a `Kc×` larger vocab-logit tensor), so it is reverted; the **per-move is
  JIT'd** so the nested-`lax.map` grid fuses into one program compiled once. A memory-bounded chunked-batch
  scorer is the obvious further perf win.
* Restoration needs the word in the insertion pool. `bd_funcwords` (default on) covers the function-word
  targets with NO per-gap bridge computation (`bd_bridge_j=0` suffices for the battery). `bd_bridge_j>0`
  adds the LM's local top-J bridges (content-word restorations); but the local top-J often MISSES the
  function-word target (the `to`/`and` case), which is exactly why the fixed funcword pool is needed.
* Signal-limited items (DEL-the, INS-01a `handed handed`) are pythia-70m LM limits, not move defects — a
  better LM would flip them. The move correctly leaves them literal.

## Knobs (`NC_*` in `calibration_word_action_smc`; args in `pythia_word_caprop.run` / `pairhmm_smc.run`)

`NC_BD_MODE` = gibbs (default) | mh | smcp3 · `NC_BD_FUNCWORDS` (fixed closed-class insertion pool, default
on) · `NC_BD_BRIDGE_J` (extra LM-bridge insertion candidates, default 0) · `NC_BD_POOL_CAP` ·
`NC_BD_ATTEMPTS` (post-loop Gibbs sweeps, default 1; ~4–5 for full restoration).

Throwaway harnesses: `planning/indel_signal_check.py`, `planning/delfrom_joint.py`,
`planning/bd_gibbs_*.log`.
