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

## 9. The remaining §6 decisions — DECIDED (user, 2026-09-01)

- **Decision 2 (default): DONE.** `LA_PROPOSAL=1` in `experiments/configs/main.env`, both arms.
  The code default stays OFF so the exact-enumeration anchor is untouched.
- **Decision 3 (`band=1`):** still untested and still separate; nothing here changes that.
- **Decision 4 (re-run `main_off`): DONE — see §10.**

## 10. The Phase-5 `main_off` re-run under the fix (decision 4 executed, 2026-09-01)

Slug `...__la__lap__nseed4`, commit `fdbb354`, 322/322 shards COMPLETED, 2337/2337 stimulus rows
ok (= 2329 model inputs), 0 errors. Cost 120.6 task-hours vs the original 114.5 — runtime-free.
All Phase-5 verification gates pass (124,695 word rows all finite; `p_copy+p_sub+p_ins = 1`;
`Σ S_k + S_end = −logZ` to 1.4e-14). Against the superseded `...__la__nseed4` outputs
(`planning/lap_rerun_vs_phase5.py` → `planning/phase5_lap_vs_la_summary.csv`, 2328 joined inputs):

| | old (la) | new (la+lap) |
|---|---|---|
| unit-0 `del_before > 0.5` (the artifact) | 350 (15.0%) | **2 (0.1%)** |
| any-unit `del_before > 0.5` (§3.4 signature) | 853 (36.6%) | **115 (4.9%)** |
| `p_literal == 0` | 728 (31.3%) | **209 (9.0%)** |
| edited MAPs | 1122 (48.2%) | **694 (29.8%)** |
| logZ (new − old) | — | **mean +3.81**, median +1.24; 1424 up / 265 down |
| … on the 853 old-affected items | — | **mean +8.98** |
| median 4-seed logZ spread | 7.09 | **0.65** |
| MAP changed | — | 999: 544 → literal, 339 edit → different edit, 116 → edited |
| gibson2013 edited, implausible vs plausible | 47% vs 35% | **33% vs 10%** |

The §3.4 scoping held up: the artifact class collapses, roughly a third of former "edits" were
the phantom-deletion artifact, and the remaining edits discriminate implausible from plausible
inputs far more sharply. The old outputs stay on disk for reference but are superseded.

**Still open:** `main_bd` (never run beyond moses; inherits `LA_PROPOSAL=1` from main.env;
sizing per the 08-31 12.6–20× estimate). The `band=1` question is CLOSED — see §11.

## 11. Decision 3 resolved: band=1 is RULED OUT for the experiment (2026-09-01)

Static capability probe `planning/band_requirement_check.py`: for every (stimulus, intended
repair) row in `experiments/stimuli/*.repairs.csv` plus the battery, the required band = the
maximum word-alignment drift between observed and intended (model unit segmentation, difflib
block boundaries). Result, 1541 repair rows:

| dataset | needs band ≥ 2 | why |
|---|---|---|
| chen2023 | **120/240** (all active↔passive rows) | "kicked" ↔ "was kicked by": ±2 words |
| tabor2004 | **64/64** | reduced relative repaired by inserting "who was" |
| huang2024 | **25/72** (the MVRR ambiguous items) | same "who was" insertion |
| battery_v0, clark2026, gibson2013, qian2023, ryskin2021 | 0 | all repairs are ≤ 1 drift |

**209 stimuli have NO band-1-representable intended repair** (none has an alternative that
fits). A length change of two words alone forces terminal drift 2, so no alignment placement
can save these. Consequences: (1) `BAND=2` stays — band=1 would silently amputate the very
repair readings chen2023/tabor2004/huang2024 exist to study; (2) the §4 "+5.5 nats, ~17%
faster" observation was smoke-local, measured in the broken-proposal regime, and is now moot;
(3) the calibration battery is 100% band-1-safe, so no battery A/B could ever have caught
this — dataset-level reachability probes, not battery runs, are the right gate for support
questions. Offender list: `planning/band_requirement_offenders.csv`.
