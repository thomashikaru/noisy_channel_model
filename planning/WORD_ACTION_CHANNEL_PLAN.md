# Word-action channel + Dirichlet action prior — design & implementation plan

**Status (2026-06-19):** design settled with the user; **not yet implemented**. This doc is self-contained
for a fresh agent. Read it, then `planning/HIERARCHICAL_CALIBRATION_PLAN.md` (the calibration context) and
the memory `calibration-substrate-status.md`. Reserved hold-out discipline still applies: all of `data/` is
sealed (`human-data-reserved-holdout.md`).

---

## 0. One-paragraph summary

The genjax port scores word **substitution** with a character-level pair-HMM (`copy=0.9` per matched char).
If we make that character `copy` a latent inferred from the text — the natural "hierarchical channel" move —
it is **certified by character counts** (~38 per sentence): a mostly-clean sentence drives the inferred copy
rate to ~0.95 and *vetoes* real-word corrections (antidote→anecdote dropped from 0.67 to 0.19 in the offline
preview). This is the wrong granularity. The fix restores the **original Gen.jl structure**: the latent noise
rate is a **word-level action distribution** `(p_copy, p_sub, p_insert, p_delete) ~ Dirichlet`, certified by
*word* counts (~7 per sentence, mild), and the character pair-HMM is demoted to scoring only the **form** of a
substitution (which neighbor / how far), conditional on an edit having occurred. Deletion and insertion are
**already** word-level rates in the port (`WDEL`, `WINS`) — which is exactly why they never over-certified —
so the change only touches **copy and substitution**, and is localized almost entirely to how the emission
table is built.

---

## 1. The model (the decomposition)

Per intended word `w`, the channel score of mapping `w` to its realized observed span factors into a
**word-level action cost** (the latent rate) plus a **conditional form cost** (given the action):

```
score(w → observed) = log P(action | θ_action)  +  log P(form | action, w)

θ_action = (p_copy, p_sub, p_insert, p_delete) ~ Dirichlet(α)        # the LATENT, per particle

form costs:
  copy   :  0                      # deterministic: w transmitted verbatim
  delete :  0                      # deterministic: w replaced by nothing
  insert : −unigram_surprisal(x)   # frequency content cost of the spurious word x (ALREADY in the model)
  sub    :  edit-distance form     # the char pair-HMM, base-rate-decoupled, ∝ sharpness^(edit distance),
                                   #   normalized over neighbors (Gen.jl: SUB_PARAM**dist / Z(w))
```

- **The latent is `θ_action`** (a Dirichlet-distributed simplex), carried per particle. It is certified at
  **word granularity**: a sentence of `M` words contributes `M` action observations to its posterior, so a
  few clean words barely move it — no over-certification. This is the whole point.
- **The pair-HMM keeps its job**: scoring substitution *form* (graded edit distance, transposition discount,
  indels, multi-token). All its advantages live in `log P(form | sub, w)` and are untouched.
- **`p_copy` may be a single global simplex per particle** (not per word). Word-count certification is mild,
  so global is expected to suffice; confirm in the offline sanity check (§4) before committing to per-word.

### 1.1 Why this is faithful to Gen.jl and fixes the bug

The original Gen.jl model: per-word action `~ Categorical(action_probs)`, `action_probs ~ Dirichlet`
(`config.ACTION_ALPHAS = [3,1,1]`); conditional on `sub`, a neighbor `~ SUB_PARAM**dist / Z(intended)`
(`noise.py:16,26`, `SUB_PARAM=0.1` = mode of `Beta(2,11)`); conditional on `sub` there is **zero**
probability of copying `w` as-is. The port replaced this with character channel events and **froze** the
action probs to points (`copy=0.9`, `sub=(1-0.9)/26`), then this calibration work proposed making the
*character* copy latent — which over-counts evidence. Putting the latent back on the **word action** (the
Dirichlet) is the faithful and correct fix.

**The tell that this is the right cut** (measured, `calibration_prior_preview.py`): deletion and insertion
never over-certified, because the port already scores them as word-level rates; substitution was the only
character-level-scored channel and the only one that broke. Symmetric fix: give substitution the same
word-level-rate treatment del/ins already have.

---

## 2. The key technical subtlety — decoupling base rate from form

The character DP currently returns `copy^(matched) · sub^(changed) · …` — it **bundles** the base rate of
editing (the `copy^matched` reward for faithful transmission) with the form (which neighbor). For the new
model the `copy^matched` *absolute reward must not appear* — the base rate lives in `p_copy`/`p_sub` now.

- A **copy** word pays `log p_copy` (a word-level constant), **not** `copy^(word length)`. ← THE change that
  moves certification from char-count to word-count.
- A **sub** word pays `log p_sub + form`, where `form` excludes the `copy^matched` reward and depends only on
  the edit operations (distance, op types).

**Implementation options for `form`:**
- **(a) Set `COPY_LP = 0` in the char-DP** (matched chars free). The DP then returns the pure edit-op cost
  (`SUB_LP^changes · INS_LP^ins · …`). Simplest. The form is then unnormalized over surfaces, but its
  per-intended-word partition folds into the calibrated `p_sub`, so it's fine for the fit. **Recommended for
  the offline sanity check.**
- **(b) Normalize per intended word**: `form = charDP(COPY_LP=0) − logZ(w)`, `Z(w) = Σ_neighbors`. Matches
  `noise.py`'s `SUB_PARAM**dist / Z(w)` exactly. More faithful; needs the neighbor partition. Decide whether
  (b)'s normalization matters after (a) is validated.

`SUB_LP` stays as the **form-sharpness** (neighbor-distribution decay), NOT the base rate. It is a candidate
calibration parameter alongside the Dirichlet `α`.

---

## 3. Map to code (exact surface)

**The change is localized to how the emission column is built; the word-level DP recurrence is unchanged.**

| piece | file:line | role | change |
|---|---|---|---|
| char pair-HMM | `pythia_word_caprop.py:70 channel_logpdf` | `log P(obs chars \| intended chars)`, uses `COPY_LP/SUB_LP/INS_LP/DEL_LP/TRANSP_LP` (`:36–44`) | add a **form variant with `COPY_LP=0`** (matched chars free) for substitution form; copy column no longer uses it |
| toy char-HMM | `poc_pairhmm_channel.channel_logpdf` | same, toy/certified path | mirror the form variant |
| **emission table** | `pairhmm_smc.py:393 emit_full = vmap(vmap(model.channel_logpdf))` | `(M × Vc)` channel score per (slot, candidate surface) | **main edit:** COPY column → `log p_copy`; SUB columns → `log p_sub + form`. See §3.1. |
| word-level DP row | `poc_word_indel.py:77 _word_row_update(log_alpha, emit_col, wdel, wins)` | one word step: `diag`=copy/sub (`+emit_col`), `up`=delete (`+wdel`), `ins`=insert (`+wins`) | **structure unchanged**; `wdel`≡`log p_delete`, `wins` rate part≡`log p_insert` (reinterpret) |
| production entry | `pythia_word_caprop.py:225 run()` | builds `WDEL` (`:60 = −9.0`), `WINS` (`:266 = log ins_rate − unigram_surp`, `ins_rate=0.02`), `CH_COPY=0.90 :34` | draw/pass `θ_action`; map `WDEL→log p_del`, `WINS→log p_ins + (−surp)`, add `p_copy`,`p_sub` |
| SMC driver | `pairhmm_smc.py:382 run`, `_make_kernel:259`, `_caprop_scores:185` | kernel applies `_word_row_update` with `WDEL/WINS`; proposal scores candidate `dZ` (`:234`) | thread per-particle `θ_action`; see §3.2 |
| rejuvenation | `pairhmm_rejuv.py` (`RJ.RejuvCtx`, used `pairhmm_smc.py:441`) | Gibbs/SMCP3 refresh of the alignment | **add a `θ_action` refresh** (Dirichlet-conjugate; §3.3) |
| Dirichlet prior | `config.py:15 ACTION_ALPHAS = [3,1,1]` (copy,sub,insert) | original Gen.jl action prior; **live only in the token-filter path** (`particle_filter_unified.py:78`, `smc_substitution.py:184`) | **extend to 4 actions** (add delete); reference those two call-sites for the `jax.random.dirichlet` draw idiom |
| form reference | `noise.py:16,26 SUB_PARAM=0.1` | `SUB_PARAM**dist / Z(intended)` normalized neighbor distribution | conceptual target for option (b) normalization |
| offline substrate | `calibration_{marginalize,prior_preview,gate,identifiability}.py` + `planning/calibration_battery_v0_gated_410m.csv` | cached LM gains + the 2-reading reduction | extend to the word-action channel for §4 |

### 3.1 The emission-table edit (the crux, and why it's cheap)

`emit_full[:, surf]` is the per-(slot, surface) channel score. Decompose it into a **shared form table** plus
**per-particle action offsets**:

```
emit_form[m, surf] = charDP_form(obs_word_m, surface)   # COPY_LP=0; shared across particles (M × Vc)
action_offset(surf) = log p_copy   if surf is the slot's COPY (verbatim) surface
                    = log p_sub     if surf is a substitution neighbour
emit_col(particle)  = emit_form + action_offset(particle's p_copy, p_sub)     # per-particle, but only a
                                                                              # scalar add by column type
```

Because the action offset depends only on the **action type of the column** (copy vs sub) and the particle's
two scalars `(p_copy, p_sub)`, you do **not** need a full per-particle `(M×Vc)` table — just the shared
`emit_form` plus a cheap per-particle add. (The COPY column is identified today as the first candidate; see
`pythia_word_caprop.py:151,172` "COPY comes FIRST … the observed span itself".)

### 3.2 Threading θ_action through the SMC

- Draw `θ_action ~ Dirichlet(α)` per particle at init (idiom: `particle_filter_unified.py:78`,
  `smc_substitution.py:184`). Carry it in the particle state tuple (`_make_kernel`'s `state`,
  `pairhmm_smc.py:269`) and the trace.
- Replace the shared scalars: `WDEL → log p_delete[particle]`, the `WINS` *rate* part `→ log p_insert[particle]`
  (keep the `−unigram_surprisal` content part as-is), and feed `(p_copy, p_sub)` into the emission column
  (§3.1). `_word_row_update` and `_caprop_scores` then read per-particle action costs.
- Default to a **concentrated** `α` (e.g. matching the current frozen point values) so the existing
  point-valued behavior is the narrow-prior limit (a regression anchor — must stay bit-compatible at the
  concentrated limit).

### 3.3 Rejuvenation on θ_action (closed-form)

Given a particle's current alignment (which intended words were copied / substituted / inserted / deleted —
counts `n_copy, n_sub, n_ins, n_del`), the Dirichlet posterior is conjugate:
`θ_action | counts ~ Dirichlet(α + (n_copy, n_sub, n_ins, n_del))`. Resample from it (a Gibbs step). Cheap —
no LM, only the channel DP. Wire alongside the existing alignment rejuvenation (`pairhmm_rejuv.py`). For the
calibration battery (one-edit items) the alignment is near-deterministic, so counts are unambiguous; general
text needs the sampled/MAP alignment's counts.

---

## 4. Offline sanity check — DO THIS FIRST (cheap, no SMC, no LM)

Validate the design in the 2-reading offline substrate before touching the filter. Extend
`calibration_prior_preview.py` (it already has the machinery: cached `g_i`, MC draws, `q_off`/`q_glob`/
`q_local`). Replace the channel term with the **word-action** model:

- per draw, sample `θ_action ~ Dirichlet(α)`; the per-word action cost is `log p_copy` (copy) or
  `log p_sub + form` (sub), `form` via option (a) (`COPY_LP=0` edit-distance), `WDEL→log p_del`,
  `WINS→log p_ins − surp`;
- compute the hierarchical posterior correction-prob, with the bulk weight now at **word granularity** (the
  certification term is the product of per-word action probs over the ~`M` words, NOT `copy^N` over chars).

**Pass criteria:**
1. **Over-certification gone:** the antidote case stays ≈ its offline value (~0.66) under the *full
   hierarchical posterior* (because certification is ~7 word-actions, not 38 chars). Contrast with the
   `q_glob` (char-copy) column that drops it to 0.19.
2. **Targets hold:** implausible members > 0.5 (correction), plausible members < 0.1 (literal); controls
   untouched.
3. **Asymmetry sensible:** deletion easier than insertion (the content cost), nonword typos confident,
   real-word malaprops graded.

**RESULT — PASS (2026-06-19, `calibration_word_action_preview.py`, out: `calibration_word_action_preview_out.txt`).**
Built with the closed-form Dirichlet-multinomial (no MC needed — the action marginal is exact), at the
UNCALIBRATED prior `α=(3,1,1,1)` over (copy,sub,ins,del). Reports `q_point` (action probs at the prior mean,
shared copies cancel → no certification) vs `q_hier` (action probs marginalized → word-level certification):

| | antidote (SUBW-01a) | SUBW family | president (SUBW-03a) | mean sub gap | targets (q_hier) |
|---|---|---|---|---|---|
| **word-action** | 0.96 → **0.89** | 0.96 → 0.91 (−0.05) | 0.82 → **0.63** | **−0.05** | 29/33 edit>0.5, 10/10 keep<0.1 (keep max 0.00) |
| char-copy (prior preview) | 0.67 → **0.19** | 0.84 → 0.68 (−0.16) | 0.42 → **0.05** | −0.16 | — |

**Criterion 1 ✅** over-certification gone (antidote barely moves; gap ~3× milder on average, ~7× on the worst
case). **Criterion 2 ✅** targets hold even uncalibrated (the 4 sub misses are the usual weak-`g` cases:
definately 0.13 etc.). **Criterion 3 ✅** nonwords confident (experimemt 1.00), transposition fixed
(recieve 0.87), keeps 0.00, del/ins graded. **Caveat:** absolute `q` is *higher* than the char model (SUBW
0.96) — the word-action form is cheaper (one action/word, no per-char `(1−copy)`) and `α` is edit-happy
(mean `p_copy`=0.5). That is what the prior search sets; the sanity check is about the certification
*mechanism* (the `q_point→q_hier` gap), which is mild at word granularity, as designed.

Then run the **prior search** (HIERARCHICAL_CALIBRATION_PLAN.md §6.3 step 3, retargeted): settle the
Dirichlet `α` (+ `SUB_PARAM` form sharpness) as the widest prior that hits the targets with margin. Document
the chosen `α` and sharpness.

**PRIOR SEARCH RESULT — SETTLED (2026-06-19, `calibration_word_action_prior_search.py`, out:
`planning/calibration_word_action_prior_search_out.txt`).** Closed-form `q_hier` swept over the action prior
`α = (copy, sub, ins, del)` and the form-sharpness `SUB_FORM_LP`, copy kept as the mode and the three error
pseudo-counts symmetric (Gen.jl's single `error_alpha`):

> **`α = (3, 1, 1, 1)`** over (copy, sub, insert, delete) — the faithful 4-way extension of Gen.jl's `[3,1,1]`,
> copy the mode (prior-mean `p_copy = 0.50`, `α0 = 6`). **`SUB_FORM_LP = log(1/26) ≈ -3.258`** ("which of 26
> letters" per edited char; transposition free).

Hits the targets: **29/33** implausible edits `> 0.5` (28 with margin `> 0.55`, mean 0.79), **10/10** plausible
keeps `< 0.1` (max 0.00), **DEL (0.81) > INS (0.57)** asymmetry. The 4 residual misses are the documented
weak-`g` / designed-borderline cases (`INS-to-01a` content-swap 0.09, `definately` 0.13, the ambiguous ladder
rung 0.18, `he is good man` 0.50) — not prior-fixable without distorting.

**Two findings to carry forward:**
- **`SUB_FORM_LP` is uncalibratable from this battery** — `1/15`…`1/40` are tied on every metric because nearly
  all battery sub-edits are distance-1 or transpositions (the sharpness only bites on multi-char edits). It is
  therefore fixed by first principles at `log(1/26)`, not fit.
- **The battery has no over-editing counter-pressure.** The matched *keeps* are separated from their *edits* by
  the LM gain `g`, never by the channel, so unconstrained width-maximization always pushes `p_copy → ~0.4`
  (the `hi` rows: e.g. `(2,1,1,1)` buys one extra borderline item at a 60%-error channel mean). A **copy-mode
  sensibility floor** (`mean p_copy ≥ 0.5`) is what keeps the settled prior a sensible *deployment* channel.
  This is fine because in the deployed hierarchical filter the per-item θ-*posterior* supplies the
  data-responsiveness that prior width is meant to provide (a clean sentence's own bulk channel pulls its
  θ-copy up; §3.2/§6.1 of HIERARCHICAL_CALIBRATION_PLAN). **Revisit at the reserved-data validation**, which is
  the first stage that *can* observe over-editing of clean text.

These `α` / `SUB_FORM_LP` are the SMC's default prior (§5) and the concentrated-limit regression anchor uses a
sharpened copy of them (§3.2).

---

## 5. Full SMC integration — the build (after §4 passes)

Ordered, each step independently testable:

1. **Decouple base rate from form** in the emission table (§3.1): add the `COPY_LP=0` form variant; build
   `emit_form`; COPY column → `log p_copy`, SUB columns → `log p_sub + form`. With `α` concentrated at the
   current point values, **assert bit-compatibility** with today's output (regression gate).
2. **Add the Dirichlet latent** per particle (§3.2): draw at init, carry in state/trace, default concentrated.
3. **Read per-particle action costs** in `_word_row_update` / `_caprop_scores` (`WDEL→log p_del`,
   `WINS→log p_ins − surp`, `(p_copy,p_sub)` into the emission column). Re-run the toy certification gates.
4. **θ_action rejuvenation** (§3.3): Dirichlet-conjugate refresh given alignment action-counts.
5. **Close the loop:** run the battery items through the SMC and confirm per-item correction-probs match the
   §4 offline word-action preview (same `α`). Divergence is diagnostic (model-constant mismatch vs particle
   degeneracy vs candidate competition — keep model-fit separate from inference error, per
   `pythia-caprop-smc-band-fix.md`).

**Code surface touched:** `pythia_word_caprop.py`, `pairhmm_smc.py`, `poc_word_indel.py`,
`poc_pairhmm_channel.py`, `pairhmm_rejuv.py`, `config.py`. The toy/certified paths
(`poc_word_indel.run`, the bigram gates) must stay green at the concentrated-α limit.

### 5.x BUILD STATUS (2026-06-19) — steps 1–4 DONE, gated; step 5 validated on Pythia

The word-action channel is implemented as a **gated, additive** path: `pairhmm_smc.run(action_alpha=...)`
(a length-4 Dirichlet over copy/sub/insert/delete) turns it on; `action_alpha=None` (the default) is the
**bit-identical certified char-copy path**. Pythia: `pythia_word_caprop.run(action_alpha=...)` /
`--word_action` (settled `α=(3,1,1,1)` = `ACTION_ALPHA_DEFAULT`). Key design that kept it clean and OFF
bit-identical (no code duplication): per-particle action costs are *always* threaded but trivial when OFF —
emission offset `lp_sub + (lp_copy−lp_sub)·[surf==copy_surf[m]]` is `≡0` (lp=0), `wdel_p/wins_p` are the
global `WDEL/WINS` broadcast, and `emit_full` is the bundled `channel_logpdf`. ON swaps in the `COPY_LP=0`
form table (`channel_form_logpdf`), the per-particle θ-offset, and `wdel_p=log p_del`, `wins_p=log p_ins −
surp` (so `wins` then carries only the content cost; the rate comes from θ).

- **Step 1 (decouple)** — `channel_form_logpdf` (COPY_LP=0, edited chars pay `SUB_FORM_LP=log(1/26)`) in
  `pythia_word_caprop.py` + `poc_pairhmm_channel.py`; the emission offset is applied per-particle in
  `_caprop_scores` / `extend_bootstrap` (`copy_surf = emit_surf[:,0]`). **Regression gate = the OFF path is
  bit-identical**: all **13 toy exact-enumeration gates stay green** (`tests/test_pairhmm_exact.py`).
- **Step 2 (latent)** — `theta ~ Dirichlet(action_alpha)` drawn per particle at init (`_theta_to_costs`),
  carried as `(theta, lp_copy, lp_sub, wdel_p, wins_p)` through the loop, resampled with the cloud.
- **Step 3 (read costs)** — `_caprop_scores`/kernel/`extend_bootstrap` take per-particle `wdel/wins` (kernel
  args, not closures) + the offset; the per-particle `a0` (leading-spurious init) is from `wins_p`. **13 gates
  re-run green** after this.
- **Step 4 (θ rejuvenation)** — `rejuv="gibbs"` with ON does the **Dirichlet-conjugate θ refresh**
  (`_action_counts` → positional `(n_copy,n_sub,n_ins,n_del)` → `θ|counts ~ Dir(α+counts)` →
  `_channel_carry_action` recomputes `log_alpha` consistent with the new θ; weight 0 under the
  deterministic-alignment likelihood — exact for the near-deterministic battery, positional-count
  approximation is the general-text caveat per §6). The ON-path alignment sweep is **deferred** (not θ-aware
  yet); θ-rejuv alone is what the battery needs.
- **Step 5 (close the loop)** — `calibration_word_action_smc.py` runs battery items through the ON SMC at
  `α=(3,1,1,1)` and reduces to the §4 preview's readings. **DOES NOT PASS at `rejuv="off"`** (out:
  `planning/calibration_word_action_smc_rejuvoff_out.txt`). Antidote→anecdote is great (q_smc=0.99 vs
  q_hier=0.89), but the **full filter is edit-happy** because with `rejuv="off"` each particle's θ is a
  *prior* draw (mean `p_ins=p_del=p_sub=0.17` — very cheap edits) and the SMC, unlike the 2-reading preview,
  explores hypotheses the cheap rates permit: **leading junk insertions** ("1?", "3,", "-", "1." from the
  prime), **DEL-to-05a kept "I want go home"** (deletion not restored), **INS-to-04a kept the spurious "to"**,
  **SUBN-02a over-edited "the boy"→"I did a little"**. See §6 (the open problem) below.

**§5.y OPEN — the §5.5 failure is the real diagnostic (deferred 2026-06-19, user said "record + pause").**
The offline preview validated the certification *mechanism* (a 2-reading L-vs-E contest); the deployed filter
needs real θ-**inference** (the posterior, not a prior draw). The hard part, surfaced here: **θ-rejuvenation
concentrates θ on each particle's CURRENT parse**, so a clean-looking "I want go home" pulls `p_del` low and
makes the genuine deletion *harder* to restore — a mode-collapse the Gibbs-from-current-parse θ move won't
escape (it reinforces whatever the parse already is). Candidate directions for the next session (not yet
tried): (a) a **tighter / less edit-happy prior** (copy-favoured, e.g. the `hi`-vs-floor tension — but the
battery can't constrain it, §4); (b) keep the **insertion rate global/low** (don't let `p_ins` be a wide
latent — leading-junk is the worst symptom; the old `ins_rate=0.02` was 2 nats dearer); (c) **alignment-aware
θ counts** (forward-backward expected counts, not the positional approximation) so θ tracks the marginalised
alignment not one parse; (d) interleave θ-rejuv with the **alignment sweep** (currently deferred for ON) so
the parse and θ co-adapt. Keep model-fit vs inference-error separate (`pythia-caprop-smc-band-fix.md`).

**Performance (verified, no regression):** word-action ON adds ~7% over OFF per step (LM forward dominates;
channel/θ ops are cheap), θ-rejuv +~0.3s (LM-free). The slow validation runs were **410m + dedup-off + P=256**
(my throwaway script didn't set the flags), not the word-action code: at 70m + dedup + P=128 the ON run is
~9.4s. Always pass `dedup=True` for iteration.

`poc_word_indel.py` and `config.py` were **not** functionally edited (the del/ins reinterpretation lives in
`pairhmm_smc.run`; the 4-way action prior is the live `action_alpha=(3,1,1,1)` default
`pythia_word_caprop.ACTION_ALPHA_DEFAULT`, NOT a `config.py` constant — the token-filter call-sites still read
the 3-way `config.ACTION_ALPHAS`, which got only a clarifying comment).

---

## 6. Open questions / risks (for the fresh agent)

- **Global vs per-word `p_copy`:** expected global suffices (mild word-count certification); confirm in §4. A
  per-word simplex is the fallback if a single sentence's clean words still over-suppress (unlikely).
- **Form normalization:** option (a) vs (b) in §2 — (a) for the sanity check; revisit if SMC numbers need (b).
- **λ (`lm_temp`) is dropped** (per the calibration pivot); the Dirichlet width is the regularizer now.
  Confirm `lm_temp=1.0` stays the default and isn't needed.
- **Rejuvenation alignment-counts** require a sampled/MAP alignment; fine for one-edit battery items, more
  care for general text.
- **Multi-token intended words:** the action is per intended *word* (which may be several BPE tokens); the
  char-DP form already scores multi-token surfaces (`genjax-port-bpe-substitution`). The action offset
  attaches at the word unit, not the token.
- **`ACTION_ALPHAS` is 3-way** `(copy,sub,insert)` today; extend to **4-way** with `delete`. The token-filter
  call-sites that read it (`particle_filter_unified.py:78`, `smc_substitution.py:184`) are a different path —
  don't break them, but they are the reference for the draw idiom.
- **`run_example_native.sh` is `git --skip-worktree`** (personal runner) — do not commit edits to it; flag
  production-path parameter changes to the user (`keep-run-example-script-current`).
