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

### 3.4 Scope — how much of the off arm this reaches

The loitering incentive operates at **every** position, not just the opening: unit 0 holds 42% of
all posited-deletion mass, the other 58% is spread through the sentence. Scoping by unit 0 alone
badly understates the reach. Splitting all 2336 items by whether ANY unit carries substantial
posited-deletion mass (`del_before > 0.5`):

| | items | edited | edit rate | `p_literal == 0` |
|---|---|---|---|---|
| **affected** | 857 (36.7%) | 681 | **79.5%** | **58.7%** |
| not affected | 1479 (63.3%) | 445 | 30.1% | 15.4% |
| all | 2336 | 1126 | 48.2% | 31.3% |

- **60% of all edited items (681/1126) carry substantial posited deletions.**
- **69% of all `p_literal == 0` items (503/731) do.**

Per-dataset share affected: ryskin2021 64%, gibson2013 49%, tabor2004 49%, huang2024 45%,
clark2026 29%, chen2023 26%, qian2023 12%.

**Caveat — this is an upper bound on the fix's reach, not a predicted improvement.** Some posited
deletions are correct: genuine missing-word items (the candle "to" restoration) legitimately carry
`del_before > 0`, and ryskin2021's high share partly reflects deliberately corrupted stimuli where
deletions are the right answer. The fix must remove spurious deletions while preserving genuine
ones — that is exactly what §6 decision 1's criteria (a) and (b) are for.

For comparison, the narrower opening-only signature (`del_before[0] > 0.5`) is 351 items = 15%,
edit rate 50.4%. That figure describes the leading-deletion artifact specifically, NOT the reach
of this fix.

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
4. **The big one: does `main_off` get re-run if this lands?** 37% of the current 2337-item results
   carry substantial posited-deletion mass, and where those are spurious, `surprisal_nc` is
   contaminated for every downstream word (the phantom word changes the LM context, not just its
   own slot). Re-running costs ~114.5 CPU-hours, which is cheap next to `main_bd`. Given §3.4,
   re-running is probably the right call rather than the fallback — but decide it after the
   battery A/B, not before.

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

---

## 8. Execution log (2026-09-01)

### 8.1 The §7 diagnostic — the split is known now

Stratified sample of 84 of the 731 `p_literal == 0` items (12/dataset), each scored literal vs
MAP via `pwc.lm_word_surprisals` (deployed prime, EOS term included), with a conservative
channel lower bound of 4.5 nats per case-insensitive word-level edit op (every op — sub form
`K*d`, del, ins — costs the MAP at least that against the all-copy literal, so the classifier
under-counts inference failures, never over-counts).

- **61/84 (73%) are inference failures** — the model's TOTAL score prefers the literal parse the
  sampler never proposed. For 38/84 the **LM alone** prefers the literal, before channel costs.
- By the §3.4 affected flag: 58/75 of `del_before > 0.5` items are inference failures vs 3/9 of
  the rest. Per dataset: chen 10/12, clark 6/12, gibson 10/12, huang 10/12, qian 8/12,
  ryskin 7/12, tabor 10/12.
- The signal-limited residue is dominated by **genuine repairs** (umpire→empire, migraine→brain,
  enigeering→engineering) — items where the edit is correct and wanted.
- Verdict: the effort belongs in §5, not model-side. Scripts/CSV in the session scratchpad
  (`p0_diagnostic.py` / `.csv`); rerunnable from `experiments/outputs/` alone.

### 8.2 The §5 fix — BUILT (commit `4e61c50`)

`lookahead_proposal` (default OFF; requires `lookahead_lp`): `_caprop_scores` computes a
per-candidate twist `logsumexp(row + la_C) − logsumexp(row)` from the row `cand_dZ` already
builds (one extra vector add, no new LM forwards, no new parameter); the kernel samples from
`scores + twist` and corrects the incremental weight by `−twist[action]` — select ∝ e^twist,
carry e^−twist, the §5 correctness requirement. EOS twist = 0 (full consumption has no unpaid
units), matching the resample block's done-guard. Wired end-to-end: `pythia_word_caprop.run`,
`run_nc_batch.py --lookahead-proposal` (slug part `lap`), `submit_nc_batch.sh LA_PROPOSAL=1`,
`experiments/run.sh` CFG_VARS. `main.env` deliberately NOT changed (§6 decision 2: default OFF
pending this A/B).

Gates: 4 new exact-enumeration tests mirroring the lookahead ones — zero-charge bit-identical
(certifies the tilt arithmetic), logZ + posterior match exact enumeration with the twist ON
(the gate on the weight correction), align+gibbs end-to-end, input contract. Full suite
**121 passed** (117 + 4).

Smoke (local, P=64, keys 0–2, la vs la+lap):

| | key 0 | key 1 | key 2 |
|---|---|---|---|
| Medics p(target), la | 0.000 | 0.213 | 0.000 |
| Medics p(target), **la+lap** | **1.000** | **0.984** | **0.968** |
| Medics logZ, la → la+lap | −91.70 → −79.68 | −91.55 → −79.96 | −92.17 → −79.57 |

The ~12-nat logZ jump lands on the best any same-model config achieved (§4's gibbs-sub-only
one-off −78.90). The candle item is IDENTICAL between arms on these keys (no restoration
either way single-seed; logZ +0.6 to +1.6 under lap) — no regression signal; the 4-seed
battery merge is the pre-committed judge for criterion (b).

### 8.3 Battery A/B (§6 decision 1) — submitted

87-item battery, off arm, P=64, N_SEEDS=4, both arms fresh at commit `4e61c50`:
baseline `...__la__nseed4` job 21752588, fix `...__la__lap__nseed4` job 21752596 (13 shards
each, MEM=24G; 26/26 COMPLETED, 87/87 merged, 0 errors on both). Report:
`planning/lap_vs_la_diff.py` → `planning/calibration_lap_vs_la.csv`.

**All four pre-committed criteria pass:**

| criterion | la (deployed) | la + lap (fix) |
|---|---|---|
| (a) unit-0 `del_before > 0.5` artifact items | 4 | **0** |
| (a') any-unit `del_before > 0.5` (the §3.4 signature) | 13 | 9 |
| (b) genuine repairs retained (`expected == edit`, n=43, case-insens.) | 14/43 | **14/43** |
| (c) edited MAPs (`MAP != observed`) | 28/87 | **26/87** |
| (d) logZ, lap − la | — | **mean +1.04**, median +0.14; up 31 / down 14 / flat 42 |
| matches-expected exact / case-insensitive | 48 / 54 | **54 / 58** |
| MAP changed | — | 17 items: **8 newly-correct, 2 newly-wrong** |
| 4-seed logZ spread (mean) | 6.70 | **1.34** |

- The biggest logZ movers (+7 to +9.8 nats) are all artifact clears (SUBW-01a, DELFROM-01b,
  CTRL-04, DELFOR-01a, SUBW-04a). Gains are the junk-MAP class cleaned up: 'The Bakerite the
  children the cake.' → 'The baker iced the children the cake.', 'The tailor seed…' → 'sewed',
  'The chef seasoned the author.' → 'the soup.'. Losses: SUBW-02a 'medic'→'media' (an LM-favoured
  substitution) and LADDER-send-2 (spurious 'Clerk' capital + dropped 'to').
- **The spread collapse (6.70 → 1.34) is the most important line.** Same model, same P=64, same
  cost per item, but the four seeds now agree: the "P=64 heavy-tail collapse" flagged in the
  08-31 battery entry was mostly this proposal-support failure, not a particle-count problem.
- Fix cost: no measurable runtime change (per-shard elapsed and MaxRSS overlap across arms).

## 9. Recommendation on the remaining §6 decisions

- **Decision 2 (default):** turn it on — `LA_PROPOSAL=1` in `experiments/configs/main.env`, both
  arms (keeps the off-vs-bd contrast single-variable). The code default stays OFF so the
  exact-enumeration anchor is untouched. Not done yet — this is the user's call.
- **Decision 3 (`band=1`):** still untested and still separate; nothing here changes that.
- **Decision 4 (re-run `main_off`):** yes. The current 2337-item results carry the artifact on
  37% of items and 60% of edits (§3.4), the fix clears the canonical class outright, and the
  seed spread drops 5×, so `surprisal_nc` in the current outputs is contaminated in exactly the
  way §6 feared. Cost ≈ 115 CPU-hours (the previous run), ~1.5 h wall at the Phase-5 fan-out.
  `main_bd` has not been run, so nothing is wasted there; it inherits `LA_PROPOSAL=1` for free.
