# Off-arm inference fix: the twist is missing from the proposal

**Date:** 2026-09-01 · **Branch:** `experiment-harness` · **Commit at investigation:** `dc84b5a`
**Supersedes nothing.** Extends `LEADING_DELETION_FINDINGS.md` and `LOOKAHEAD_CHARGE_PLAN.md`.

---

## 1. Where the experiment stands

**Phase-5 `main_off` is COMPLETE and verified.** 2337/2337 items ok, 0 error, 0 missing, under
`lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4`. Pulled to `results_nc/`,
collected to `experiments/outputs/`.

- Cost spent: 322 array tasks + 1 refill, 114.5 CPU-hours.
- The one Phase-5 casualty (huang2024 shard 20, TIMEOUT at its auto-sized `--time`) was refilled
  by job `21746380` in 1:37. Lesson: `SECONDS_PER_ITEM=120` is too thin where p90 sentence length
  is 15+ words (huang2024, tabor2004).
- Verification gates all pass: 124,695 word rows with **zero** non-finite surprisals;
  `p_copy+p_sub+p_ins = 1` everywhere; `Σ S_k + S_end = −logZ` to **2.8e-14** on all merged records.

**`main_bd` is NOT run** (only moses, 1 item). User decision 2026-09-01: hold off pending
inspection of the off results. If it later runs: `SENTENCES_PER_SHARD=2` everywhere, `MEM=32G`,
dataset by dataset — 1164 tasks exceeds the account's `MaxSubmit=500`. Estimated 1440 CPU-hours
at the measured 12.6× smoke ratio, up to 2300 at the 20× single-item ratio.

---

## 2. The problem

The off arm returns junk MAPs on a specific, identifiable class. Canonical case, smoke item 2:

```
observed:   Medics cleaned and bandaged the wound to prevent an inflection.
off  MAP:   This article medic cleaned and bandaged the wound to prevent an infection.
bd   MAP:   Medics cleaned and bandaged the wound to prevent an infection.   (p = 0.994)
```

**This is an inference failure, not a model preference.** Measured:

| | value |
|---|---|
| LM alone, clean parse | −75.07 |
| LM alone, off MAP | −77.76 (**2.69 nats worse**) |
| Channel, extra cost of off MAP | 2 deletions at −4.62 + `medic→Medics` sub (−4.5 to −9.0) |
| **Model's total preference for the clean parse** | **≈ 17–21 nats** |

The correct parse is present in the off cloud at rank 7 with **1.08%** of the weight; `bd` gives
the same string 99.43%. (Correcting the standing note in `align-off-loiter-artifact`: this is a
*weighting/support* failure, not absent support — check `hypotheses` before calling any instance
a support failure.)

---

## 3. Root cause (measured, verified in code)

### 3.1 The proposal never writes the literal reading down

Traced particle cloud at the first word, deployed config with `LOOKAHEAD=1`:

```
P=64,  key=0:   ""(whitespace) 89.7% of weight,  4 particles     ess = 3.9 / 64
                "This"          2.7%,           59 particles
P=256, key=0:   ""(whitespace) 90.7%,           37 particles     ess = 10.7 / 256
```

**"Medics" is proposed zero times at P = 64, 256, and 1024, across three keys.** The cloud is
already degenerate at step 0.

### 3.2 Why: the proposal scores immediate cost only

`_caprop_scores` (`pairhmm_smc.py:218`) scores each candidate `score = lm_temp*lm_part + dZ`:

| candidate at step 1 | LM term | action term (Dirichlet α) | total |
|---|---|---|---|
| `'\n'` posited as a deleted word | −0.50 | −4.62 (delete) | **−5.12** |
| `'Medics'` read literally | −13.63 | −0.02 (align) | **−13.65** |

The action prior **is** in the proposal via `dZ` and does hand the literal reading a **4.60-nat
advantage**. It is swamped by the LM's 13.13-nat preference for a cheap opener. Net **8.53 nats
against the literal**.

The deeper defect: a bridge candidate explains *none* of the observation — "Medics" is still
unexplained — yet is scored as if deferring that cost were free. The literal candidate pays
13.63 up front and gets no credit for discharging the debt.

### 3.3 The correction already exists but is applied in only one place

The lookahead charge (`la_C`, `psi`) is exactly this deferred-cost estimate.

- Resampling, `pairhmm_smc.py:886–910`: `lsel = log_w + psi` — **applied**.
- `_caprop_scores` — argument list contains no `la_C`. **Not applied.**

So the twist decides which particles *survive* but has no say in which candidates are *proposed*.
In a fully-adapted auxiliary particle filter the twist belongs in both. That asymmetry is the bug.

### 3.4 Scope — read this before estimating the payoff

| | value |
|---|---|
| Items with opening artifact (`del_before[0] > 0.5`) | **351 of 2336 = 15%** |
| Edit rate, items WITH the artifact | 50.4% |
| Edit rate, items WITHOUT it | 47.8% |
| Overall edited | 1126 of 2336 = 48% |
| Overall `p_literal == 0` | 731 of 2336 = 31% |
| Share of all posited-deletion mass sitting at unit 0 | 42% |

**Fixing this will not move the 48% overall edit rate.** It cleans up a 15% contamination that
matters because phantom leading words corrupt the LM context for every downstream word's
`surprisal_nc` — the experiment's main output. Over-editing at large is a separate, signal-side
problem.

Per-dataset artifact rate: ryskin2021 35%, tabor2004 27%, gibson2013 21%, chen2023 10%,
huang2024 5%, qian2023 5%, clark2026 3%.

---

## 4. Ruled out — do NOT retry

| lever | result | evidence |
|---|---|---|
| **Alternative primes** (`'\n'`, `''`, `'.\n'`) | **worse** | logZ −101.8 / −95.5 vs −88.2 deployed. Puts pythia in code/markup mode; MAPs become `'# Medics…'`, `'1. Medic…'`. |
| **α_del = 0.02** (raise deletion price) | **fixes artifact, destroys the science** | Medics p(target) 0.97–0.98, but candle "to" restoration goes **0/6 keys** vs **4/6** deployed. One parameter doing two jobs. |
| **`use_word_mask=True`** | **no effect** | p(target) 0.000 on 3/3 keys; `del_before[0]` rises to ~2.00. Removing `'\n'`/`'#'` just yields `'In a media…'`, `'This is medic…'`. The problem is not the vocabulary. |
| **More particles** | **no effect** | p(target) 0.000 at P = 64, 256, **1024**, three keys each. |
| **`gibbs+bd` at lookback 1 or 2** | **worse than deployed** | logZ −91.6 / −92.2 (k0), −96.0 / −96.2 (k1); 9–12× slower than off. |
| **Unigram-tilted proposal** | **superseded** | Would bolt an outside signal on to approximate a bias the model already specifies. §5 uses the model's own quantities, no new parameter. |
| **Bigger LM** | previously ruled out | see `calibration-improvement-antipatterns`. |

### Partial / unreliable

- **`band=1`**: **+5.5 nats** on the identical model (mean −84.2 vs −89.7), ~17% faster
  (12s → 10s warm), candle repair survives. Does **not** fix this class. A free inference
  improvement worth taking on its own merits, but needs battery validation and a check that no
  stimulus requires two adjacent missing words.
- **`gibbs` substitution-only**: key 0 → p 0.922, logZ −78.90 (best of any same-model config);
  key 1 → p 0.000, logZ −95.46. Wildly seed-dependent; matches the standing
  "gibbs-sub-only is a trap" finding. **Not a reliable cheap substitute for bd.**
- Note: full `gibbs+bd` at lookback 6 also **failed on key 1** locally (p 0.000, logZ −95.46).
  The cluster's 4-seed merge rescues it. Single-key local runs are not diagnostic in this regime.

---

## 5. Proposed fix

**Fold the existing lookahead twist into the candidate proposal scores.**

Location: `_caprop_scores`, `src/genjax_port/pairhmm_smc.py:218`. Today:

```python
def cand_dZ(col):
    return logsumexp(band_mask(_word_row_update(log_alpha, col, wdel, wins), t_new)) - Z
```

The per-candidate updated alpha row computed there is exactly what `psi` is built from at
resample time (`logsumexp(state[5] + la_C) - tot`, line 893). Scoring
`logsumexp(row + la_C)` instead of `logsumexp(row)` charges each candidate for what it leaves
unexplained.

**Cost: one vector add inside a function that already runs. No new LM forwards. No new parameter.**

Expected effect at step 1: the bridge candidate still owes 13.63 for the unexplained "Medics";
the copy candidate has paid it. That moves the bridge down ~13.6 nats relative to the copy,
turning the measured **8.5-nat deficit into roughly a 5-nat advantage** for the literal reading —
enough that it dominates the proposal instead of never appearing.

**Correctness requirement:** the proposal is currently fully adapted, so the incremental weight is
`logsumexp(scores)` independent of the draw — that is what keeps caprop low-variance. Tilting the
proposal requires the matching weight correction (the standard APF arrangement already used at
resampling: select ∝ `w·e^psi`, carry residual `w/e^psi`). Get this right or logZ stops being
unbiased. The existing gates must stay bit-identical with the flag off.

---

## 6. Decisions needed from the user

1. **Success criterion and validation set.** One item and two keys settle nothing here — see the
   sub-only result flipping 0.922 → 0.000 between keys. Proposal: the 87-item
   `calibration_battery_v0` at 4 seeds, pre-committed criteria: (a) artifact items
   (`del_before[0] > 0.5`) cleared, (b) genuine repairs retained — candle "to" restoration rate,
   (c) overall edit rate not worse, (d) logZ up on the same model.
2. **Flag name and default.** Recommend default-OFF behind a flag, same discipline as
   `lookahead` — the exact-enumeration gates depend on the certified path being bit-identical.
3. **Whether `band=1` rides along or is tested separately.** It is an independent +5.5-nat win;
   confounding the two would make the A/B unreadable. Recommend separately.
4. **The big one: does `main_off` get re-run if this lands?** The current 2337-item results carry
   the artifact on 15% of items, and `surprisal_nc` on those is contaminated beyond unit 0.
   Re-running costs ~114.5 CPU-hours. Alternatives: re-run only affected items, or ship with the
   contamination documented.

---

## 7. Reproduction pointers

- Probes written this session (scratch, not committed): step-1 trace via `pwc.run(..., trace=tr)`;
  pattern in `planning/leading_del_probe/trace_steps.py`.
- Existing probes: `planning/leading_del_probe/{mech_test,lm_paths,particles,trace_steps}.py`.
- Diagnostic column: unit-0 `del_before` in `words.csv.gz` is the artifact detector.
- Key measurement recipe: `pwc.lm_word_surprisals(sentence)` scores any candidate intended
  sentence under the LM with the deployed prime — this is how §2's 75.07 vs 77.76 was obtained,
  and it is the way to separate inference-limited from signal-limited on any disputed item.
- **Recommended next diagnostic before building anything:** score literal vs MAP directly on a
  sample of the 731 `p_literal == 0` items. Nobody knows the inference/signal split across that
  set; it determines whether §5 or a model-side change is where the effort belongs.
