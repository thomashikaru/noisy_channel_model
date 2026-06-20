# Calibrating the noisy-channel model's free parameters to human inference

**Status:** framework design (Phase 1). Written *blind* to the reserved human data under
`data/` — see "Pre-registration discipline" below. Nothing here was tuned against
`gibson2013/`, `ryskin2021/`, `qian2023/`, or `ncgp2/`.

---

## 0. The problem, stated carefully

The model has a handful of free **channel** parameters (deletion cost, insertion rate,
character edit rates, LM-prior temperature). We want to set them so the model's inferences
look like the inferences *people* draw when reading noisy/implausible sentences — without
over-editing (correcting things a human would leave alone) and without missing the corrections
a human would readily make.

The central methodological commitment, which shapes everything below:

> **Human noisy-channel inference is graded and uncertain, so the calibration target is a
> *distribution*, not a label.** We match the model's *posterior probability* of an
> interpretation to the *proportion of people* who adopt it, with uncertainty represented on
> **both** sides. We never fit to "did the model's single best reconstruction equal the human
> modal answer" — that throws away the gradation and rewards exactly the overconfident
> over-editing we want to avoid.

This is the noisy-channel comprehension paradigm of the psycholinguistics literature
(Gibson/Levy and successors): the empirical quantity is the *rate* at which people adopt a
non-literal interpretation as a function of (a) the edit needed to reach it and (b) the prior
plausibility of literal vs. inferred meaning. That rate is what we calibrate to.

---

## 1. Five separations that keep the calibration honest

These are the load-bearing distinctions. Most ways to get calibration wrong are a failure to
hold one of them.

**1.1 Model parameters vs. inference/approximation parameters.**
Only the first set is calibrated to humans.
- **Model (calibrate to humans):** char rates θ_copy, θ_sub, θ_ins, θ_del, θ_tr; word
  deletion cost `WDEL`; insertion rate `ρ_ins` (`ins_rate`); LM-prior temperature `λ`
  (`lm_temp`); possibly `ACTION_ALPHAS`. These *define the posterior*.
- **Inference / approximation (set for *fidelity*, never fit to humans):** band `b`, particle
  count `P`, `J`, `Ke`, `MAX_SUB_CANDIDATES`, `MAX_DELETIONS`, rejuv lookback `ℓ`. Fitting
  these to behavior would launder *inference error* into the "model" and is a category error.

**1.2 Model fit vs. inference error.** (The project's standing rule — see
`memory/pythia-caprop-smc-band-fix.md`: "separate model-correctness from inference BEFORE
tuning.") We calibrate the **model's posterior**, not the SMC estimate of it. Every calibration
item must first pass an inference-fidelity gate (§4) so that any model–human gap is genuine
misfit, not Monte-Carlo noise or a degenerate intermediate target.

**1.3 Model capacity vs. calibration.** Some apparent over-/under-edits are *representational*
limits, not parameter settings, and must be excluded from the fit (or they drag the costs to
compensate for a missing capability). Multi-token words are **fully supported** for copy,
substitution, and insertion (the `(token span, surface id)` representation; faithful COPY from the
observed span). The one residual limit is multi-token **deletion**: restoring an *omitted* word
that itself tokenizes to several BPE pieces is deferred. Common function-word omissions ("to",
"the", "a", "of") are single-token and unaffected; only an item whose correct reconstruction
requires restoring a dropped *multi-token* word hits this limit.

**1.4 The LM is part of the prior.** `pythia-70m` vs `410m` vs `gpt2` is effectively a *prior*
choice, not a continuous knob. Treat it as a **discrete model-selection axis**: fit the channel
params *given* each LM, then compare LMs by *held-out* predictive — do not tune the LM choice on
the same data as the fit.

**1.5 Parsimony.** Prefer the simplest parameterization that predicts within human noise. Every
free knob must earn its keep (Occam, via marginal likelihood / LOO), because several of the
knobs are mutually entangled (§5) and a small dataset cannot identify all of them.

---

## 2. Pre-registration discipline (why this doc is blind)

All human behavioral data in `data/` is a **locked hold-out test set**
(`memory/human-data-reserved-holdout.md`). Designing or tuning the framework against the data we
will later validate on overfits the *framework* to the test set and destroys the generalization
claim.

So the protocol is pre-registration in spirit:

1. **Commit, blind:** the observable, the model→response aggregation map (§3), the
   stimulus/condition design (§3), the objective (§6), the free-vs-pinned parameter sets and
   their ranges/priors (§5), the validation metrics (§7), and the train/validation/test split
   (§8 — the big open decision).
2. **Only then unlock** the reserved data, at the pre-committed validation step.
3. **No re-tuning after the unlock.** If held-out prediction fails, that is a *finding*; any
   change opens a new, freshly pre-registered round. We do not iterate against the test set.

---

## 3. The observable and the parameter→observable map

**Human observable (primary): interpretation proportion.** For each item, the fraction of
participants who adopt the literal reading vs. one (or more) edited alternative(s). This is the
cleanest match to the model's output and to the literature's paradigm.

**Human observable (secondary): processing signal.** Self-paced-reading / eye-tracking measures
(and an explicit "I noticed an error" response) give a graded *processing-cost* signal, which
links to the model via posterior P(error) / expected surprisal rather than a single
interpretation probability. Keep this as a *separate* likelihood channel (§6) — do not force a
reading time and an interpretation proportion into one number.

**The aggregation map `A` (commit in advance, per item).** The model emits a posterior over
fine-grained intended sentences; humans give a coarse response. Define, per item, a partition of
the candidate intended-sentence support into **response classes** (literal / specific
alternative(s) / other), then

  q_i(θ, class) = Σ_{x ∈ class} P_θ(x | o_i).

`A` is part of the pre-registration: it must be fixed before the fit, not chosen to make a given
θ look good.

---

## 4. Inference-fidelity gate + the cheap-refit substrate (prerequisite to any fit)

Before an item can constrain a parameter, we must be able to compute the **model** posterior on
it faithfully.

**4.1 Gate.** For each candidate item: confirm the SMC posterior ≈ the *exact* posterior on the
candidate-restricted support (the channel marginal is a forward DP and the support is finite, so
near-exact per-item posteriors are computable — drive `P` up, or enumerate, or use the joint
scorer). **Exclude/flag** items where (a) SMC can't reproduce the model posterior (inference
problem, §1.2) or (b) the correction is unrepresentable or the LM doesn't distinguish the readings
(capacity problem, §1.3): the LM-noticeability check `slp(alt) ≠ slp(literal)`, plus the one
representability caveat that the correction must not require restoring a dropped *multi-token* word
(multi-token copy/substitution/insertion are fine). Result: `q_i(θ)` is the (near-)exact model posterior,
so the fit sees model misfit only.

**4.2 The key efficiency insight (makes a full Bayesian fit tractable).** The LM chain-rule
log-probs of the candidate intended sentences are **independent of the channel parameters**
(θ_copy/sub/ins/del, `WDEL`, `ρ_ins`): only `λ` rescales them, and only the channel
forward-DP / character-edit structure depends on the channel θ. So **precompute once per item**:
the candidate support, each candidate's LM score, and the character-edit-distance emission
structure. Then re-evaluating `q_i(θ)` for a new θ is just the cheap forward DP + softmax —
**no LM forwards**. A full grid or MCMC over the channel parameters is therefore essentially free
after a one-time per-item LM pass. (`λ` only rescales the cached LM vector — still no new
forwards.) This is what turns "calibrate by re-running SMC thousands of times" into "re-run a
vectorized DP thousands of times."

---

## 5. Which parameters, and can the data identify them?

**5.1 Candidate free set and what each is identified by.** Design the stimulus conditions (§3.1
below is the taxonomy) so that each free parameter is constrained by its *own* cell:

| Parameter | Meaning | Identified by condition |
|---|---|---|
| θ_sub (and the d-slope) | char substitution sharpness | substitution items across edit distance d = 1,2,3 |
| θ_ins = θ_del (char) | char indel rate | sub-vs-indel-distance items (typo shape) |
| `WDEL` | word-deletion (missing-word) cost | omitted-word items (esp. function words) |
| `ρ_ins` | spurious-word insertion rate | doubled / spurious-word items |
| `λ` (`lm_temp`) | LM-prior temperature | **prior-plausibility** contrast (the Gibson manipulation): literal-implausible-but-grammatical vs. edited-plausible |

**5.2 Pinned by first principles (recommended).** θ_copy = 0.90, the `/26` substitution
normalization, and θ_tr = θ_sub are structural (the normalization sharpness is principled — see
`memory/pythia-caprop-smc-band-fix.md`; un-normalizing it was a past bug). Pin these; free the
rest.

**5.3 Identifiability is a real risk — analyze it before fitting.** `λ`, `WDEL`, and `ρ_ins` are
*known to trade off*: `λ<1` curbs substitution over-editing but buys word-deletion over-editing
(`memory/lm-temp-prior-tempering.md`). So:
- Run a **simulation-based identifiability check** on the battery first: inject known θ, generate
  synthetic human proportions, and confirm the fit recovers θ with the planned conditions. If two
  params are non-identified (posterior ridge), either pin one by principle or add a condition that
  breaks the degeneracy.
- Report posterior **correlations** among the fitted params, not just marginals.

---

## 6. The objective: hierarchical and uncertainty-aware on both sides

**Human side.** For item `i` with `k_i` of `N_i` participants adopting the edited reading, model
the counts as **Beta-Binomial** (binomial + an item-level over-dispersion / random-intercept term
`φ`), *not* a point proportion — humans have item idiosyncrasy beyond binomial sampling noise.

**Model side.** `q_i(θ)` is the near-exact model posterior class-probability from §4, with
residual MC error driven *below* the human binomial SE (we control `P`; human data we don't).

**Likelihood & priors.**

  L(θ, φ) = Σ_i  BetaBinom( k_i | N_i, q_i(θ), φ ),

with first-principles priors on θ (θ_copy≈0.9 etc.; weakly-informative on `WDEL`, `ρ_ins`, `λ`).
Because dim(θ) is small and refits are cheap (§4.2), do a **full Bayesian fit** (NUTS, or grid +
importance weights) and report the **posterior over θ with credible intervals** — *uncertainty
on the parameter side too*. Shipping a posterior (not a point) is precisely how we honor "don't
treat a specific inference as gold": the fit integrates over item-level and parameter
uncertainty.

**Avoiding over-edits, concretely. [DECIDED]** Use a **balanced battery with explicit no-edit
controls** (clean plausible sentences, target adopt-edit ≈ 0) under the plain Beta-Binomial
likelihood above. An honest likelihood on a base-rate-matched battery *already* punishes a θ that
over-edits the controls — over-editing is penalized by the data, not by a hand-set loss. An
explicit asymmetric utility (weighting false-edits worse than missed-edits) is **deferred / not in
scope**; revisit only if controls + honest likelihood prove insufficient after a real fit.

---

## 7. Validation (pre-committed, then unlock)

**Internal (on the fitting battery).**
- **Leave-one-condition-out CV:** does θ fit on substitution items predict deletion items? This
  tests whether the *channel structure* is right, not merely flexible enough to interpolate.
- **Reliability diagram:** predicted edit-probability vs. observed proportion, binned — direct
  read on calibration.
- **Proper scoring** (log-loss / Brier) on held-out items; **posterior-predictive** checks with
  credible bands.

**External (the unlock — `data/`).** Only after the framework, metrics, param posteriors, and
split are committed: run the **frozen** model on the reserved datasets and report held-out
predictive (log-loss, calibration, correlation of model P(edit) with human proportion). No
re-tuning. Cross-LM and free-vs-pinned-param-set choices are made by *this* held-out predictive
(+ LOO/WAIC Occam), never by training fit.

---

## 8. Phasing, with gates

**Committed immediate scope = P0–P4 + the internal-validation machinery (the "substrate").**
The fit-data choice (§9.1) is **deliberately deferred** until that substrate exists; the real fit
(P5), the unlock (P7), and the lock-in (P8) wait on that later decision. We build the apparatus
first so the eventual fit-data decision is made with the identifiability/fidelity results in hand.

- **P0 — Housekeeping: pin the current operating point. ✅ DONE → `CALIBRATION_P0_OPERATING_POINT.md`.**
  Audited every channel + inference parameter across the production path, `config.py`, `noise.py`,
  the generic filter, and `model_current.tex`. Findings: the doc table **agrees with the production
  library defaults** on all *model* params; the live tensions are (1) `WDEL` has four values in the
  tree (library/doc **−9.0**, the user's runner **−8**, `config` legacy −5.30, generic −2.30), and
  (2) the personal runner diverges from the library on inference settings (`rejuv=gibbs/ℓ=5/P=128`
  vs `off/3/64`). Several look-like-parameters (`config.P_DELETE_PRIOR`, `MAX_DELETIONS`,
  `noise.SUB_PARAM`, the generic `pairhmm_smc.run` defaults, `ACTION_ALPHAS`) are **legacy decoys**
  the pair-HMM never reads. Recommended baseline = library defaults with `WDEL=−9.0`.
- **P1 — Framework spec (this doc) + pre-registration.** Commit §3, §5, §6, §7, §8-split, blind.
- **P2 — Battery construction.** Factorial conditions (§5.1) + no-edit controls; per-item gating
  (§4.1): LM-noticeability (`slp` gain) + the multi-token-deletion representability caveat
  (multi-token copy/sub/insertion are supported).
- **P3 — Inference-fidelity gate + per-item LM/edit-structure cache** (§4) — the cheap-refit
  substrate.
- **P4 — Simulation-based identifiability ✅ DONE → `calibration_identifiability.py`.** Recovered
  known-injected θ=(λ, WDEL, log ρ_ins, κ_sub) from synthetic counts (no human data; pure numpy on the
  cached gate output, via the 2-reading logistic reduction = a binomial GLM). **Verdict: GO.** Recovery
  is unbiased (|bias|≤0.04) with calibrated 1σ intervals (coverage ~0.66); design rank 4/4 — all dials
  separately identified. Caveats, both fixable by battery expansion: (1) **ρ_ins under-powered** (only
  3 insertion items → 1σ ±0.82, vs λ ±0.08, WDEL ±0.24, κ_sub ±0.47); (2) **strong λ↔cost coupling**
  (r ≈ −0.9 of λ with every cost) — the dials are jointly identified but individually correlated. To
  decouple: add (a) more insertion items, (b) keep-side "tempting-edit" anchors for DEL/INS (negative-g
  points, as the SUBW keeps already provide for substitution), and (c) wider within-type spread of LM
  preference g. λ itself is well-pinned (shared across types + gain variation).

  **Expansion done (battery v1, 79 items / 53 fit points):** added 6 insertion pairs + 5 high-preference
  deletion pairs, and generalized the gate to compute INS keep contrasts. Result: ρ_ins 1σ **±0.82 →
  ±0.44** (no longer under-powered), and all dials tightened (λ ±0.047, WDEL ±0.16, κ_sub ±0.31); still
  unbiased, coverage ~0.72. Correlations eased (λ↔WDEL −0.90→−0.80; cost–cost 0.84–0.89 → 0.71–0.84) but
  the **λ↔cost coupling stays ≈−0.9 for ρ_ins/κ_sub** — intrinsic to the 2-reading reduction (λ scales the
  LM preference everywhere, costs are per-type offsets). Mitigation is not more 2-reading items but (i)
  always reporting the **joint** posterior (already mandated), and (ii) the full multi-candidate model at
  P5, where λ also shapes EOS/length and the candidate normalization and so gains independent leverage.
- **P5 — Fit** → posterior over θ with CIs (§6).
- **P6 — Internal validation** (§7).
- **P7 — UNLOCK**: frozen held-out evaluation on `data/`. Report.
- **P8 — Lock defaults** = posterior medians + documented uncertainty; collapse `config` /
  pythia-path / doc to one source of truth; update the `model_current.tex` reference-params
  table; keep the battery as a regression suite.

---

## 9. Open decisions for you (recommendations inline)

1. **Data partition / fit source — DEFERRED (chosen 2026-06-18).** Decision: **methodology-first**
   — build the substrate (P0–P4) before choosing what to fit on. `data/` stays fully reserved
   throughout the substrate work. Leading candidate for the eventual fit set is **literature-derived
   aggregate proportion tables** (keeps all of `data/` as held-out test, strongest generalization
   claim); a small fresh collection or a pre-committed split of `data/` remain on the table. The
   intuitive `golden_targets.json` is a *smoke test only*, never a fit target. Revisit once P4
   (identifiability) results are in.
2. **Over-edit handling — DECIDED:** honest Beta-Binomial likelihood + balanced no-edit controls.
   Asymmetric utility deferred (§6).
3. **Primary fit paradigm:** interpretation-proportion *(rec primary)*; add the reading-time /
   "noticed-an-error" channel as a secondary likelihood only if wanted. *(Settle at P5, not needed
   for the substrate.)*
4. **Free vs. pinned set:** *(rec)* free {`WDEL`, `ρ_ins`, `λ`, char-indel θ}; pin {θ_copy, `/26`
   normalization, θ_tr}. *(Locked at P4 via the identifiability check.)*
5. **Param-posterior usage downstream:** ship the **posterior median + CI** and propagate
   parameter uncertainty into reported inferences *(rec, matches the uncertainty-aware ethos)*
   vs. a single point default. *(Settle at P8.)*
