# Hierarchical noisy-channel calibration — plan (pick up here)

**Status (2026-06-19):** the calibration substrate is built and a key pivot is validated — the channel
rates are now treated as **latent variables with priors** and **marginalized** (no λ tempering), which
produces graded, uncertainty-aware inferences with zero over-editing. The fork is **resolved toward
Option 2** (rates latent in the production filter). The immediate task is to **settle the priors against
the calibration battery's target inferences** — and the key realization (2026-06-19) is that *settling the
priors is entangled with building Option 2 and must be done against the hierarchical posterior, not the
offline prior-average* (see §6, rewritten). We do **not** anchor the centers to external typo/disfluency
corpora; the battery's desired inferences set them.

**Channel redesign (2026-06-19, settled with the user) — read `planning/WORD_ACTION_CHANNEL_PLAN.md`.** The
latent is **not** a character/per-word `copy` rate (the §3 three-Beta framing and the §6.2 per-word-copy
result are now a superseded stepping stone). It is the **word-level Dirichlet action distribution**
`(p_copy, p_sub, p_insert, p_delete)` from the original Gen.jl model, certified at *word* granularity, with
the pair-HMM scoring only substitution *form*. That doc is the self-contained design + code-map +
implementation plan for the next session; this doc holds the calibration framing and the targets.

**Read first:** this doc, then memory `calibration-substrate-status.md`. Earlier docs are still valid for
detail: `CALIBRATION_PLAN.md` (the original framework + P0–P8 phasing), `CALIBRATION_P0_OPERATING_POINT.md`
(live param audit), `CALIBRATION_BATTERY_DRAFT.md` (battery design). **Reserved:** all of `data/`
(gibson2013, ryskin2021, qian2023, ncgp2) is a sealed hold-out — see `human-data-reserved-holdout.md`.

---

## 1. The pivot, in one paragraph

We started trying to *fit point values* for the free channel parameters (`WDEL`, `ins_rate`/ρ_ins,
`lm_temp`/λ, substitution sharpness) to an intuitive calibration set. That fought us at every turn: the
dials are entangled, "moderate confidence everywhere" forced λ→~0.15 (nearly flattening the LM and
killing the subtle meaning-based edits), and `WDEL` ran to its bound. The fix (the user's: restore the
original Gen.jl design) is to **not pick point values at all** — make the channel rates *latent with
priors* and **integrate over them**. Doing so produces naturally graded, hedged predictions; the
*width* of the prior sets the confidence instead of a hand-tuned λ. **λ is dropped.** Validated on the
battery at 410m: predictions are graded, controls stay at ~0 (no over-editing), and the
deletion>insertion asymmetry (a Gibson/Bayesian fact about humans) falls out for free.

---

## 2. Locked decisions

- **Target is a distribution, not a label.** Match the model's edit-*probability* to human edit-*rates*,
  uncertainty on both sides. Never fit to a MAP/accuracy match.
- **Reserved hold-out.** `data/` is sealed; design blind; it is spent once, at the end, for the test.
- **410m is the calibration LM** (meaning-based cases need it; 70m too weak).
- **Asymmetry is a desideratum.** Deletion (restore a missing word) > insertion (remove a spurious one)
  matches Gibson's human data and rational Bayes. Emerges from the content cost, not hand-built.
- **Channel rates are LATENT with PRIORS; λ is DROPPED.** Calibrate the *priors* (centers + widths), not
  point values.
- **Priors are settled against the BATTERY's target inferences, not external corpora.** The objective is
  operational and battery-internal: for each matched pair, the *implausible* member's posterior mass on the
  designed correction must be **> 0.5**, and the *plausible* member's posterior mass on the literal reading
  must be **> 0.9** (edit-prob < 0.1). These targets *shape the prior*; they are **design desiderata, not
  the human-rate objective** — so this does not violate "never fit to a MAP/label" (decision #1): we are not
  declaring the battery the human distribution, we are choosing a deliberately-shaped prior that the reserved
  human data will later *test*. (Supersedes the abandoned "anchor centers to typo/disfluency statistics"
  idea.)
- **Priors are settled against the HIERARCHICAL posterior, not the offline prior-average** (the 2026-06-19
  realization — see §6). The deployed model (Option 2) averages the edit-probability under the θ-*posterior*,
  which the bulk clean channel reweights toward a clean θ; settling on the offline prior-average would
  systematically miss in deployment.
- **No separate malapropism channel.** "antidote→anecdote" is plain edit-distance 2; the char channel
  handles it. (Corrected mid-session — don't re-propose this.)
- **Transposition discount is in and principled.** A transposition is determined (no "which wrong letter"
  choice) so it should not pay the 1/26 alphabet penalty — ~log(26) cheaper than a substitution.
- **Battery scope:** include Gibson spurious-*function*-word insertions (INS_TO: delete a spurious "to");
  EXCLUDE doubled-word items (INS_DUP) — removing an exact rare-content duplicate costs the word's full
  unigram surprisal, a known model gap (a duplicate-aware channel is the eventual fix).

---

## 3. The model as it stands (the "offline reduced model")

`src/genjax_port/calibration_marginalize.py`. For each (one-edit) item it reduces the posterior to **two
readings** (literal vs. the designed correction) and computes the edit-probability marginalized over the
channel-rate priors:

```
logit_i(θ) = g_i + channel_i(θ)
q_i        = E_{θ ~ priors}[ sigmoid(logit_i(θ)) ]      (Monte-Carlo over prior draws; λ dropped, =1)
```

- `g_i` = LM preference = slp(correction) − slp(observed), **cached** from the gate at 410m (no LM work
  at predict time).
- `channel_i(θ)`:
  - **substitution:** `n_sub·κ_letter + n_indel·κ_letter + n_trans·κ_trans`, where
    `κ_letter = log((1−copy)/26) − log(copy)` and `κ_trans = log(1−copy) − log(copy)` (transposition
    discount). Op counts from an OSA decomposition (`edit_ops`).
  - **deletion:** `log(p_del)`.
  - **insertion:** `log(ρ_ins) − surp_uni(spurious word)`  (the content cost gives the asymmetry).

**Current (placeholder) priors — TO BE SETTLED:** `copy ~ Beta(8,2)` (mean 0.80), `p_del ~ Beta(1.5,18)`
(mean 0.077), `ρ_ins ~ Beta(1.5,18)` (mean 0.077).

**Validated behavior (410m, S=20000):** edits lean-edit 26/33 (mean 0.74); keeps 10/10 lean-keep (mean
0.00, **max keep q = 0.01** → no over-editing); 23/43 strictly graded in (0.05,0.95). Transposition typos
fixed ("recieve" 0.31→0.88; mountian/suop/causal→1.00). Residual uncertain cases ("definately" 0.23,
"president" 0.42, the daughter/candle insertion, the mid ladder rung) are all genuine LM-weakness or
designed-borderline — the model is uncertain exactly where the evidence is weak (a feature).

---

## 4. Artifacts + how to run (needs the `ncgenjax` arm64 conda env)

| file | what |
|---|---|
| `planning/calibration_battery_v0.csv` | 87 items, matched implausible/plausible pairs (SUBW, SUBN, DEL_*, INS_TO, INS_DUP[excluded], LADDER, CTRL) |
| `src/genjax_port/calibration_gate.py` | P3 fit-readiness gate; writes `calibration_battery_v0_gated.csv` (70m) / `_gated_410m.csv` |
| `src/genjax_port/calibration_identifiability.py` | P4 simulation-based parameter recovery (2-reading GLM); not central post-pivot |
| `src/genjax_port/calibration_marginalize.py` | **the current model** — offline marginalized predictions at 410m |
| `src/genjax_port/calibration_fit_intuitive.py` | the ABANDONED point-fit (kept for reference; shows why points fail) |

```
# regenerate 410m gains (already cached) and the marginalized predictions:
NC_LM=EleutherAI/pythia-410m PYTHONPATH=src conda run -n ncgenjax python -u \
  -m genjax_port.calibration_gate planning/calibration_battery_v0.csv planning/calibration_battery_v0_gated_410m.csv
PYTHONPATH=src conda run -n ncgenjax python -u -m genjax_port.calibration_marginalize
```
(Redirect long runs to a file; see `never-pipe-expensive-output.md`.)

---

## 5. Option 1 vs Option 2 (the relationship)

Both share the **settled priors** as their input. The difference is *where the integration over θ happens*
and whether the data can update θ.

- **Option 1 — offline.** Keep the offline reduced model (§3). Settle the priors, run prior-predictive
  checks, then run the **human test offline**: score each reserved item once for `g_i`, apply the channel
  formula, average over the priors, compare predicted edit-prob to human edit-rate. The production SMC is
  **untouched** (stays point-valued, unused). *Sufficient for the scientific question* — the reserved
  items are one-edit corrections, so the two-reading reduction is faithful. Averages over the **prior**
  (the observation does NOT update θ).
- **Option 2 — rates latent in the production filter.** The settled priors become the prior over a
  genuine latent θ carried per particle; the SMC infers the joint posterior over (sentence, θ). The same
  `Beta(α,β)` are reused verbatim — settling them is literally Option 2's input spec. **Adds** the ability
  for the observation to UPDATE θ (posterior, not prior-average): a garbled sentence is evidence the
  channel is noisy, so the model edits more readily, and it can learn the channel across a corpus.
  Needed for a **deployable general-text** model. Bigger build (touches `pairhmm_smc`, per-particle state,
  trace schema, likely a rejuvenation step on θ). Not much extra compute (θ only hits the cheap channel
  DP, not the LM).

**Correction (2026-06-19):** I previously claimed the two predictions are close for a single short stimulus
("one sentence barely moves a global rate"). That is wrong when θ is inferred *per item*: a single
sentence's own bulk clean channel (`copy^N`, `(1−rate)^M`) strongly determines its θ-posterior, so Option 2
diverges from the Option-1 prior-average **even for one stimulus** — and conservatively (toward keeping; see
§6.1). So Option 1 is *not* a faithful stand-in for what we deploy; the priors must be settled against the
Option-2 posterior `q_full`. We compute `q_full` cheaply offline (the §6.3 preview) without building the
SMC, which keeps the fast iteration of Option 1 while being faithful to Option 2. **The prior's WIDTH
propagates into Option 2:** wide → the filter does real θ-inference (garbled→edit, clean→keep); narrow →
collapses toward old point-value behavior.

---

## 6. IMMEDIATE NEXT TASK — settle the priors against the battery, in the hierarchical model

**Reframed 2026-06-19.** Do **not** anchor centers to external typo/disfluency corpora. Use the calibration
battery and its *target inferences* (the "informed peak + reasonable uncertainty" comes from these):

> **Targets.** For each matched pair: implausible member → posterior mass on the designed correction
> **> 0.5**; plausible member → posterior mass on the literal reading **> 0.9**. Prefer the **widest**
> priors (most uncertainty) that still satisfy the targets with margin.

### 6.1 Why this is entangled with Option 2 (settle against the *hierarchical posterior*)

Write each item as a two-reading contest, literal `L` (= observed) vs the designed edit `E`, with
`g = logP_LM(E) − logP_LM(L)` (cached, θ-independent) and channel term `ch(θ)`. The two models differ in
*which distribution over θ they average the edit-probability under*:

- **Offline reduced model** (current `calibration_marginalize.py`): averages the per-θ conditional under
  the **prior** — `q_off = E_θ[ σ(g + ch(θ)) ]`. (The bulk clean channel cancels in the L-vs-E *ratio*, so
  it never enters here.)
- **Deployed hierarchical model (Option 2)**: averages under the θ-**posterior** —
  `q_full = E_θ[ W(θ)·σ(g+ch(θ)) ] / E_θ[ W(θ) ]`, where the evidence weight `W(θ) ∝ ` the item's marginal
  likelihood, and `W(θ)` is **dominated by the bulk clean channel**: `copy(θ)^(#matched chars)` (and
  `(1−p_del)^(#kept words)`, `(1−ρ_ins)^(#kept words)`).

The bulk term is the catch. A 38-char sentence with one suspicious word is ~37 votes for a *clean* channel
against 1 for a noisy one, so the θ-posterior concentrates on high `copy`, making the edit **more expensive
than the prior mean implies**. E.g. `copy~Beta(8,2)` (mean 0.80) reweighted by `copy^37` ≈ `Beta(47,3)`
(mean ≈0.94); at copy≈0.94 a 2-char real-word correction (antidote→anecdote, g≈10.7) drops from q≈0.81 to
q≈0.2 — **below target**, purely from the clean context certifying the typist. Two consequences:

1. **Settling on `q_off` and deploying `q_full` would systematically miss** — the deployed model is more
   conservative, and the gap *grows with sentence length*. Settle against `q_full`.
2. **Prior width flips meaning.** In `q_off`, width = hedging toward 0.5. In `q_full`, width = *capacity for
   the data to move θ*: a garbled item pulls θ noisy and edits confidently; a clean control pulls θ clean and
   keeps. That data-driven sharpening is the whole point of going hierarchical and is invisible offline.

### 6.2 The crux design question this surfaces (decide empirically, via 6.3)

A single **global** `copy` latent is over-certified by clean context (`copy^N`), so few-character real-word
corrections get suppressed even when the LM strongly prefers them. (Genuine nonword typos have huge `g` and
survive; real-word malapropisms are the casualties.) Responses to weigh:

- **(a) Permissive global prior** — low-ish `copy` center / heavy low tail so post-reweighting mass still
  allows editing. Simplest, but a literally low global copy is hard to justify and risks over-editing short
  items.
- **(b) Local / hierarchical noise** — per-word (or per-region) noise so *one* suspicious word can be noisy
  without indicting the whole sentence. Principled (a malapropism is a local slip), matches Option 2's
  per-particle latent story, bigger build.
- **(c) Source mixture** — a sentence-level "clean source vs noisy source" indicator, so a cluster of errors
  can be attributed to a noisy source without forcing the global copy rate up. Middle ground.

**Recommendation:** measure the over-certification first (6.3). If only real-word malaprops suffer, prefer
(b)/(c) over a cosmetically-low global copy.

**MEASURED → RESOLVED to (b), scoped to the character channel (2026-06-19, `calibration_prior_preview.py`).**
The preview (placeholder priors, 410m, S=20000) shows the over-certification is **entirely substitution-
specific** and exactly as predicted:

| family | `q_off` | `q_glob` (one global copy) | `q_local` (per-word copy) | implied post-copy |
|---|---|---|---|---|
| SUBW (real-word malaprop) | 0.84 | **0.68** | 0.81 | 0.94 |
| SUBN (nonword typo) | 0.82 | 0.77 | 0.80 | 0.92 |
| DEL_* / INS_TO / LADDER | — | ≈ `q_off` (±0.02) | ≈ `q_off` | 0.91–0.96 |

Named: antidote→anecdote `0.67 → 0.19 → 0.66`; president→precedent `0.42 → 0.05 → 0.25`; nonwords
(experimemt) stay `1.00`. Under the **global** copy latent the whole sentence certifies copy≈0.94–0.96 and
the multi-char real-word edits collapse; the **word-level** rates (`p_del`,`ρ_ins`) are untouched (their
bulk `(1−rate)^M` over ~7 words is far too weak to over-certify). Making **copy per-word** (the suspect word
certifies only itself, `copy^(len word)`, length-independent) **recovers the offline behavior** (SUBW
`0.68→0.81`; aggregate implausible>0.5 `25/33 → 26/33`, = `q_off`; keeps stay 0.00). **Decision: copy is a
PER-WORD latent; `p_del` and `ρ_ins` stay global.** Option (b), but scoped only to the character channel —
not a whole new mechanism. (`president` stays 0.25 — but it was 0.42 offline, a designed weak-`g`
borderline, not a hierarchical regression.)

**SUPERSEDED 2026-06-19 → the latent is the word-level ACTION distribution, not a per-word character copy.**
Discussion with the user (who wrote the Gen.jl model) reframed the fix at the right level of abstraction. The
per-word-*copy* result above is a valid *diagnostic* — it proved the over-certification is substitution-
specific and that localizing the noise to the word fixes it — but "per-word character copy" is a clumsy
re-derivation of a structure the original model already has. The over-certification is an artifact of the
**port scoring substitution with a character-level pair-HMM** (~38 certification events) instead of Gen.jl's
**word-level action model** (~7 events). The faithful fix: the latent noise rate is the **Dirichlet-
distributed word action distribution** `(p_copy, p_sub, p_insert, p_delete)`, certified per word; the
character pair-HMM is demoted to scoring only the substitution *form* (which neighbor), conditional on an
edit. Deletion and insertion are *already* word-level rates in the port (`WDEL`, `WINS`) — which is exactly
why the preview showed them untouched — so only copy and substitution change. **Full design + code-surface
map + implementation plan: `planning/WORD_ACTION_CHANNEL_PLAN.md`.** The offline preview should be re-run with
the word-action channel to re-confirm the targets before the SMC build.

**Consequence for settling priors:** because `q_local ≈ q_off`, the existing **offline marginal model is a
faithful proxy** for the per-word hierarchical model, so the prior search (6.3 step 2) can run on it and will
transfer to Option 2. The only residual hierarchical effect is the mild, arguably-correct word-length
certification (`copy^(len word)`), which the search should fold in.

### 6.3 Procedure

1. **Build the cheap `q_full` preview ✅ DONE → `calibration_prior_preview.py`.** Computes the hierarchical
   posterior edit-prob on the **existing Monte-Carlo draws** (same cached `g_i`, same channel DP, plus the
   bulk weight `copy^N`/`(1−rate)^M`), in three columns: `q_off` (prior-average), `q_glob` (one global
   copy), `q_local` (per-word copy). No SMC, no new LM forwards. **Result: §6.2** — over-certification is
   substitution-specific; per-word copy recovers offline behavior. Output: `calibration_prior_preview_out.txt`.
2. **Decide the structural question (6.2) ✅ DONE — then SUPERSEDED:** the diagnostic said per-word copy;
   discussion with the user settled the faithful version — the latent is the **word-level Dirichlet action
   distribution**, with the pair-HMM scoring only substitution *form*. **The remaining steps move to
   `planning/WORD_ACTION_CHANNEL_PLAN.md`** (the full design + code map + build order).
3. **Re-run the offline preview with the word-action channel** (`WORD_ACTION_CHANNEL_PLAN.md §4`): confirm the
   over-certification is gone at word granularity and the targets hold, *then* settle the priors.
4. **Settle the priors = the Dirichlet `α` (+ `SUB_PARAM` form sharpness)** as the widest prior hitting the
   targets with margin; inspect implied per-item action-posteriors for sensibility (battery only — never the
   reserved data). Document the chosen `α` and sharpness here as a short addendum.
   **DONE (2026-06-19, `calibration_word_action_prior_search.py`) → `α = (3,1,1,1)` over (copy,sub,ins,del),
   `SUB_FORM_LP = log(1/26)`.** Hits 29/33 implausible>0.5, 10/10 keeps<0.1, DEL(0.81)>INS(0.57). Two findings:
   `SUB_FORM_LP` is uncalibratable from this battery (fixed by first principles); and the battery has no
   over-editing counter-pressure, so a **copy-mode floor** (`mean p_copy ≥ 0.5`) is needed to keep the prior a
   sensible deployment channel — full writeup in `WORD_ACTION_CHANNEL_PLAN.md §4` (prior-search result).
5. **Build the hierarchical SMC** (§7 / `WORD_ACTION_CHANNEL_PLAN.md §5`) and confirm it reproduces the
   offline word-action preview on the battery; the settled `α` is its input verbatim.

Optional refinement (medium effort, not required): a **character confusion matrix** for substitutions
(a→i, e→i cheap) — would lift "definately"; the transposition discount already handled the common cases.

---

## 7. Build the hierarchical SMC (the fork is resolved)

The fork (Option 1 offline vs Option 2 latent-in-the-filter) is **resolved toward the latent-in-the-filter
model**: the priors are settled against the hierarchical posterior anyway (§6), and the deployable model is
the goal. **The full design + exact code-surface map + build order live in
`planning/WORD_ACTION_CHANNEL_PLAN.md`** (self-contained for a fresh agent). In brief:

- The latent is the **word-level Dirichlet action distribution** `(p_copy, p_sub, p_insert, p_delete)` per
  particle (faithful to the original Gen.jl model), certified at *word* granularity — not a character copy
  rate. The character pair-HMM scores only substitution **form** (which neighbor), conditional on an edit.
- Only **copy and substitution** change; **deletion (`WDEL`) and insertion (`WINS`) are already word-level
  rates** in the port (which is why they never over-certified) and are just reinterpreted as `log p_delete`
  and `log p_insert − unigram_surprisal`.
- The change is localized to **how the emission table is built** (`pairhmm_smc.py:393`): COPY column →
  `log p_copy`; SUB columns → `log p_sub + form` (char-DP with `COPY_LP=0`). Cheap per-particle add.
- **Closed-form Dirichlet rejuvenation** of `θ_action` given the alignment's action-counts; trace carries
  `θ_action`; concentrated-`α` limit recovers (and must stay bit-compatible with) today's point behavior.
- **Close the loop:** the SMC must reproduce the §4 offline word-action preview on the battery.

**Then the human test (the unlock, unchanged from the original plan):** pre-commit metrics (calibration
curve, correlation, log-loss), unseal `data/` ONCE, score, compare. No re-tuning after unsealing. This now
tests the *hierarchical* model directly (not an offline proxy).

---

## 8. Parking lot / open items

- Doubled-word (INS_DUP) removal — needs a **duplicate-aware channel** (an exact adjacent duplicate is a
  cheap dittography slip regardless of the word's frequency). Deferred.
- Char **confusion matrix** for substitution typos (the "definately" case).
- ~~Ground `p_del`/`ρ_ins` centers in actual disfluency statistics~~ — **dropped** (2026-06-19): centers
  come from the battery's target inferences (§6), not external corpora.
- The two substitution parameterizations from P0 are effectively reconciled: use the char-DP with the
  transposition discount; `SUB_PARAM`/`config` legacy values remain decoys (see P0 doc).
- Multi-token DELETION still deferred (single-token function words to/for/from/of/a/the are fine).

## 9. Working discipline

- Reserved `data/` stays sealed until the pre-committed test; design + prior-setting blind to it.
- Calibrate the **priors** (distributions), never collapse to point values.
- `run_example_native.sh` is `git --skip-worktree` (personal runner) — do not commit edits; flag
  production-path changes to the user.
- Long runs → redirect to a file with flushed progress; use the `ncgenjax` env.
