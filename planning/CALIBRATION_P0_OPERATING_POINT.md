# P0 — The current operating point (single source of truth)

Companion to `CALIBRATION_PLAN.md`. This pins the **live** parameter values of the production
noisy-channel pair-HMM so calibration has a defined point to move *from*. The audit's headline
finding: parameter values in the tree depend on **which entry point you read**, and several
look-like-parameters are *legacy decoys* that the production pair-HMM never touches.

**Production entry point** = `pythia_word_caprop.run` (the channel-aware pair-HMM SMC on real
Pythia). The toy-bigram path is for certification, not behavior. Default LM = **pythia-70m**
(`lm_penzai.py:31`, `NC_LM` override).

---

## Group A — Model (channel) parameters  → these are what we CALIBRATE

| Param | Symbol | Live value | nats (log) | Source | `model_current.tex` |
|---|---|---|---|---|---|
| char copy | θ_copy | 0.90 | −0.105 | `pythia_word_caprop.py:34` | 0.90 ✓ |
| char substitution | θ_sub | (1−0.9)/26 ≈ 0.00385 | −5.557 | `:37`, `ALPHA=26 :33` | (1−0.9)/26 ✓ |
| char insert / delete | θ_ins, θ_del | 0.05 | −2.996 | `:35` | 0.05 ✓ |
| char transposition | θ_tr | = θ_sub | −5.557 | `:44` | = θ_sub ✓ |
| word deletion (missing-word) | `WDEL` | **−9.0** | — | `:60 WDEL_DEFAULT` | −9 ✓ ⚠ runner uses −8 |
| spurious-word (insertion) | `WINS` | log(ρ_ins) − unigram_surp(w), **ρ_ins=0.02** | per-word | `:268`, default freq-aware | freq-aware ✓ |
| LM-prior temperature | λ (`lm_temp`) | 1.0 | — | `run()` default | 1.0 ✓ |
| the LM (the prior itself) | — | EleutherAI/pythia-70m | — | `lm_penzai.py:31` | "Pythia/GPT-NeoX" ✓ |

The model `model_current.tex` reference table **agrees with the library defaults** on every Group-A
parameter. Good — the documented model is the production model. The one live tension is `WDEL`
(see Discrepancies).

---

## Group B — Inference / approximation parameters  → SET FOR FIDELITY, never fit to humans

| Param | Symbol | Library default | Source | Runner override |
|---|---|---|---|---|
| alignment band half-width | b | 2 | `run()` | 2 (same) |
| particles | P | 64 | `run()` | **128** |
| top-J LM bridge candidates | J | 8 | `run()` | — |
| candidate words / observed word | Ke | 12 | `run()` | — |
| candidate frontier window | cwin | 1 | `run()` | — |
| max char edit distance (SymSpell) | max_dist | 2 | `run()` | 2 (same) |
| slack word-steps beyond M | slack | 3 | `run()` | — |
| rejuvenation | rejuv | **"off"** | `run()` | **"gibbs"** |
| rejuv lookback window | ℓ | **3** words | `run()` | **5** |
| rejuv candidates / slot | rejuv_Ke | 8 | `run()` | — |
| candidate-retrieval cap | `MAX_SUB_CANDIDATES` | 32 | `config.py:55` | — |

`MAX_SUB_CANDIDATES` is the one `config.py` constant the production path *does* honor (it caps
SymSpell retrieval, which the pair-HMM shares). Everything else in `config.py` is Group C.

---

## Group C — LEGACY / DECOY values  → NOT used by the production pair-HMM

These are easy to mistake for live channel parameters. They belong to the **token-level**
`smc_substitution` / `particle_filter_unified` path and the `poc_*` prototypes — not the pair-HMM.

| Looks-like | Value | Where | Why it's a decoy |
|---|---|---|---|
| deletion prior | `P_DELETE_PRIOR = 0.005` (−5.30) | `config.py:39` | token-filter deletion prior; the pair-HMM uses `WDEL=−9.0`, not this |
| deletion proposal rate | `P_DELETE_PROPOSAL = 0.20` | `config.py:40` | token-filter proposal exploration only |
| consecutive-deletion cap | `MAX_DELETIONS = 1` | `config.py:23` | token-filter; pair-HMM bounds indels with the **band**, not this |
| lookahead candidates | `LOOKAHEAD_K = 6` | `config.py:45` | token-filter deletion proposal only |
| generic SMC defaults | `wdel=log(0.1)=−2.30`, `wins=log(0.05)=−3.00` | `pairhmm_smc.py:348` | **always overridden** by `pythia_word_caprop` (passes −9.0 + freq-aware); dead in production |
| form-sub sharpness | `SUB_PARAM = 0.1` (−2.30/char) | `noise.py:26` | legacy `word_sub_loglik` (`char_dist·log 0.1`); the pair-HMM scores substitution by the **char DP** (θ_sub above), and `_candidate_words` sorts by raw distance, so `SUB_PARAM` never enters the production path |
| action Dirichlet | `ACTION_ALPHAS = [3,1,1]` | `config.py:15` | original Gen.jl action prior; no explicit edit action exists in the pair-HMM (edits are channel events), so this is not a live pair-HMM parameter |

---

## Discrepancies found (the reason P0 exists)

1. **`WDEL` has four values in the tree:** library/doc **−9.0**, the user's runner **−8**,
   `config` legacy −5.30, generic/poc −2.30. Live production = −9.0; the demos the user actually
   runs (`run_example_native.sh:50`) use −8. → This is the headline knob to calibrate; the baseline
   is **−9.0**, but note the as-operated demos differ by 1 nat.

2. **Library vs. runner diverge on inference settings:** library runs `rejuv=off, ℓ=3, P=64`;
   the runner runs `rejuv=gibbs, ℓ=5, P=128`. These are Group-B (fidelity) knobs, but **rejuvenation
   changes the *quality of the posterior approximation***, so the calibration's inference-fidelity
   gate (Plan §4) must fix one setting. Rejuv only ever *improves* the approximation toward the same
   target — it never changes the model — so calibrate either (a) with rejuv ON at the runner setting,
   or (b, cleaner) via per-item exact/near-exact enumeration (Plan §4.1), which sidesteps the choice.

3. **`config.py`, `noise.py`, and the generic `pairhmm_smc.run` defaults are legacy** w.r.t. the
   pair-HMM and are the most likely source of future "which value is live?" confusion.

---

## Recommended canonical operating point (the baseline we calibrate *from*)

Adopt the **library `pythia_word_caprop.run` defaults** (= the `model_current.tex` table) as the
documented baseline, with these explicit calls:

- **`WDEL = −9.0`** as the baseline (reconcile the runner's −8 — pick one; recommend −9.0 to match
  the library + doc, or consciously adopt −8 if the demos are the reference).
- **Inference fixed for the fit:** band=2, max_dist=2; resolve rejuv via the fidelity gate (Plan §4)
  — recommend per-item near-exact enumeration so the fitted costs see model misfit, not MC/rejuv noise.
- **Demote the decoys:** clearly label `config.P_DELETE_PRIOR / P_DELETE_PROPOSAL / MAX_DELETIONS /
  LOOKAHEAD_K`, `noise.SUB_PARAM`, the generic `pairhmm_smc.run` wdel/wins defaults, and
  `ACTION_ALPHAS` as legacy/non-pair-HMM (a one-line comment each), so nobody calibrates a dead knob.

**Code-hygiene follow-ups (flagged, not done here):**
- `run_example_native.sh` is `git --skip-worktree` (personal runner — do **not** commit edits to it).
  Its `WDEL=-8`, `REJUV=gibbs`, `REJUV_LOOKBACK=5`, `PARTICLES=128` differ from the library
  defaults; if you want the demos to reflect the calibrated baseline, you'll need to update your
  local copy yourself.
- Consider adding `legacy:`-tagged comments to the Group-C constants (a production-path change — I'll
  do it on your say-so rather than silently).
