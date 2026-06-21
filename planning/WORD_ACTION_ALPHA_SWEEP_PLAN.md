# Word-action α sweep plan (copy-concentration re-tune)

**Status:** drafted 2026-06-19, **not yet run.** Purpose: find the action-prior Dirichlet α that best aligns the
word-action channel with our intuition — *plausible sentences read literally, implausible sentences corrected to
their minimal-pair twin* — by sweeping the **copy concentration** (the user observed `100,1,1,1` already beats the
current `3,1,1,1` default). This is the α re-tune that the consolidation plan's §1b default-flip is gated on
(see [[word-action-channel-status]], [[code-consolidation-done]]).

## 0. Hypothesis
α = `(α_copy, 1, 1, 1)` with `sub=ins=del=1`. At rejuv=`off` the per-particle θ is a **prior draw**, so the prior
mean `p_copy = α_copy/(α_copy+3)` directly sets the edit rates: bigger `α_copy` ⇒ more copy-favoured ⇒ less
over-editing. We expect, as `α_copy` rises: should-KEEP literal retention `L` ↑ (good), should-EDIT correction
rate `E` ↓ (eventually too low). **Pick the knee**: the highest `α_copy` that still corrects the implausible items.

## 1. What we reuse (no new evaluation logic)
- **Runner:** `src/genjax_port/calibration_word_action_smc.py`. Already supports `NC_ALPHA`, `P`, `SEED`, an item
  list or `ALL`, `dedup=True`, `NC_REJUV`, and **caps the sentence-initial letter** (`CAP`, `NC_NOCAP=1` to
  disable). Writes one fixed-width row per item (`item exp metric q_ref L E junk  obs -> intended`) + a SUMMARY.
  - Metric (already exactly our intuition): EDIT items `q_smc = E/(E+L)` (pass `>0.5`); KEEP items `kept = L`
    (pass `>0.5`); `junk = 1−L−E` (spurious-insertion / over-edit mass — the leading-opener diagnostic).
- **Analyzer (the lens that matters):** `src/genjax_port/calibration_battery_analyze.py`. Side-by-side over result
  files; reports **should-EDIT correction rate `E`**, **should-KEEP literal retention `L`** (and spurious `1−L`),
  and the **within-pair "tracks manipulation" rate** = fraction of matched pairs with `E_implausible > (1−L_plausible)`,
  plus the by-family breakdown and junk count. This is the primary decision surface.
- **Battery:** `planning/calibration_battery_v0_gated.csv` (the **70m** gate — same 87 items as the 410m variant,
  only the gate columns differ; use the 70m gate for a 70m sweep). Matched pairs by `pair_id`: `a` = implausible →
  `edit`, `b` = plausible → `keep`. This is the **synthetic** calibration substrate, designed blind — **not** the
  locked human hold-out in `data/` ([[human-data-reserved-holdout]]); do not touch `data/`.

## 2. Fixed sweep config (fast)
`P=128`, `NC_LM=EleutherAI/pythia-70m`, `seed=0`, `dedup=True`, `band=2`, `rejuv=off` (isolates the prior-α effect),
all other `run()` defaults (`wdel=-8`, `ins_rate=0.02`, `lm_temp=1.0`). `SUB_FORM_LP` (= log(1/26)) held fixed —
the α sweep is orthogonal to the substitution-form cost.

**α grid (4 columns):** `3,1,1,1` (current default = baseline) · `10,1,1,1` · `50,1,1,1` · `100,1,1,1`.
Prior-mean p_copy = 0.50 / 0.77 / 0.94 / 0.97 respectively.

## 3. Balanced subsample (20 matched pairs / 40 items = 20 edit + 20 keep, all 11 families)
Selection rule (reproducible): from `calibration_battery_v0_gated.csv`, keep `pair_id`s that have **both** an
`edit` and a `keep` member and both `gate_pass=PASS`; take **up to 2 lowest-numbered pairs per family**; emit both
members. Yields exactly these 40 item_ids (write to `planning/wa_alpha_subsample.txt`, one space-separated line):

```
SUBW-01a SUBW-01b SUBW-02a SUBW-02b SUBN-01a SUBN-01b SUBN-02a SUBN-02b DELTO-01a DELTO-01b DELTO-02a DELTO-02b
DELFOR-01a DELFOR-01b DELFOR-02a DELFOR-02b DELFROM-01a DELFROM-01b DELFROM-02a DELFROM-02b INS-01a INS-01b
INS-02a INS-02b LADDER-give-2 LADDER-give-3 LADDER-send-2 LADDER-send-3 DEL-of-01a DEL-of-01b DEL-a-01a DEL-a-01b
DEL-a-02a DEL-a-02b DEL-the-01a DEL-the-01b INS-to-01a INS-to-01b INS-to-02a INS-to-02b
```
Families: SUBW·SUBN·DEL_TO·DEL_FOR·DEL_FROM·INS_DUP·LADDER·DEL_OF·DEL_A·DEL_THE·INS_TO (DEL_OF / DEL_THE have only
1 clean pair each, hence 20 not 22 pairs). To regenerate, see the selector snippet in §6.

## 4. Two prep changes to the runner (the only code edits)
**(a) Final-period normalization (the user requirement: "capitalized AND a final period").** The runner already
caps the initial letter but does not add a terminal period (83/87 battery items lack one; 49/87 are
lowercase-initial). Replace the `_cap`-only step in `evaluate()` with a `_wellform()` that does both. Safe because
`_norm()` strips `[^a-z0-9 ]` before L/E matching, so the period only **well-forms the LM input** (it does not
change which decoded sentences count as literal/correction):
```python
def _wellform(s):
    s = s.strip()
    if s and s[0].islower():      s = s[0].upper() + s[1:]   # capitalize initial
    if s and s[-1] not in ".!?":  s = s + "."                # ensure terminal period
    return s
# in evaluate():  if CAP: observed, intended = _wellform(observed), _wellform(intended)
```
**(b) Point the runner at the 70m battery.** Change `CSV` to `planning/calibration_battery_v0_gated.csv` (or add an
`NC_CSV` env override). Item membership is identical to the 410m variant; only the gate columns differ.

## 5. Run it (redirect to files — never pipe, see [[never-pipe-expensive-output]])
```bash
SUB=$(cat planning/wa_alpha_subsample.txt)
for A in 3,1,1,1 10,1,1,1 50,1,1,1 100,1,1,1; do
  NC_LM=EleutherAI/pythia-70m NC_ALPHA=$A NC_REJUV=off NC_VERBOSE=0 PYTHONPATH=src \
    conda run -n ncgenjax python -u -m genjax_port.calibration_word_action_smc 128 0 $SUB \
    > planning/wa_alpha_sweep_${A//,/_}.txt 2>&1
done
# matched-pair lens, all four side by side:
PYTHONPATH=src conda run -n ncgenjax python -m genjax_port.calibration_battery_analyze \
  planning/wa_alpha_sweep_3_1_1_1.txt planning/wa_alpha_sweep_10_1_1_1.txt \
  planning/wa_alpha_sweep_50_1_1_1.txt planning/wa_alpha_sweep_100_1_1_1.txt \
  > planning/wa_alpha_sweep_analysis.txt 2>&1
```
**Runtime estimate:** ~1–2 min per α (40 items, P=128/70m, dedup on, rejuv off) + a one-time model load; ~**8–12 min**
for the 4-α sweep + analysis. The runner already prints per-item elapsed timing.

## 6. Decision criterion
From `wa_alpha_sweep_analysis.txt`, choose the α that best balances (in priority order):
1. **should-KEEP literal retention `L` high** — plausible sentences read literally (the user's primary concern;
   the over-editing we're curing). Expect `L` to climb monotonically with `α_copy`.
2. **within-pair "tracks manipulation" % high** — the model edits the implausible member more than it spuriously
   edits the plausible twin (`E_a > 1−L_b`).
3. **should-EDIT correction rate `E` not collapsed** — genuine slips still corrected. Expect `E` to fall with
   `α_copy`; the **knee** is the largest `α_copy` before `E` craters.
4. **low `junk`** (leading-opener / over-edit mass).

If `100,1,1,1` is still improving on (1)/(2) without cratering (3), **extend the grid** (e.g. `200,1,1,1`,
`500,1,1,1`) in a second pass. Re-generate the subsample / confirm with `seed=1` if the top two αs are within noise.

## 7. After the prior sweep: confirm at the deployment setting
The sweep above is `rejuv=off` (prior-only, the clean α knob). The deployed filter runs `rejuv=gibbs` (the θ
**posterior**), which has the known deletion mode-collapse caveat ([[word-action-channel-status]] §5.5). Once an α
is chosen, run **one** confirmation pass at the chosen α with `NC_REJUV=gibbs` on the same subsample and re-analyze,
to check the posterior-θ path doesn't reintroduce over-editing. Only then promote the α to the deployment default
(`ACTION_ALPHA_DEFAULT` in `pythia_word_caprop.py` + the `run_example_native.sh` default + the `--selftest`
expectations) — i.e. complete consolidation-plan §1b.

## 8. Deliverables
`planning/wa_alpha_subsample.txt`, four `planning/wa_alpha_sweep_<α>.txt` result files,
`planning/wa_alpha_sweep_analysis.txt`, and a one-paragraph conclusion naming the chosen α (and whether the grid
needs extending). No change to `data/`.
```python
# §3 subsample regenerator (writes planning/wa_alpha_subsample.txt)
import csv
from collections import defaultdict
rows=list(csv.DictReader(open("planning/calibration_battery_v0_gated.csv")))
META={r["item_id"]:r for r in rows}; bypair=defaultdict(dict); order=[]
for r in rows:
    if r["pair_id"] not in bypair: order.append(r["pair_id"])
    bypair[r["pair_id"]][r["expected"]]=r["item_id"]
ok=lambda p: all(k in bypair[p] for k in ("edit","keep")) and all(
    META[bypair[p][k]]["gate_pass"]=="PASS" for k in ("edit","keep"))
seen=defaultdict(int); items=[]
for p in order:
    if not ok(p): continue
    f=META[bypair[p]["edit"]]["family"]
    if seen[f]<2:
        seen[f]+=1; items += [bypair[p]["edit"], bypair[p]["keep"]]
open("planning/wa_alpha_subsample.txt","w").write(" ".join(items)+"\n")
```

---

# Progress log / RESULTS (2026-06-21) — rejuv=off sweep DONE

**Status: the §2 rejuv=off α grid is RUN and analyzed. α=100,1,1,1 is the clear winner; the user's
hypothesis is confirmed. NOT yet done: the §6 grid extension (200/500) and the §7 rejuv=gibbs
confirmation — both deferred to the next session at the user's request ("no new long runs tonight;
tomorrow I'll want both alpha=200/500 AND with rejuv").**

## What was done
- **§4 runner edits made** in `src/genjax_port/calibration_word_action_smc.py` (UNCOMMITTED working-tree
  change, branch `word-action-rejuv`): (a) `_cap()` → `_wellform()` (capitalize initial **and** add a
  terminal period; safe because `_norm` strips `[^a-z0-9 ]` so L/E matching is unaffected); (b) `CSV` now
  defaults to the **70m** gate `planning/calibration_battery_v0_gated.csv` with an `NC_CSV` override.
- **§3 subsample generated** → `planning/wa_alpha_subsample.txt` (the exact 40 ids in §3; regenerator reproduces them).
- **Migration sanity check (the `--word_action` → `--channel word_action` Phase-3 rename):** full live
  regression suite **24/24 green** (incl. `test_channel_selector_pure_rename` + `test_channel_selector_validation`).
  The runner uses the Python API `W.run(action_alpha=…, channel=None)`, which infers `channel="word_action"`
  — insulated from the CLI rename. CLI is `--channel {word_action,char_copy}` + `--action_alpha`; no `--word_action` stragglers.
- **§2 sweep RUN** (P=128, 70m, seed=0, dedup=True, band=2, rejuv=off): four files
  `planning/wa_alpha_sweep_{3,10,50,100}_1_1_1.txt` + side-by-side `planning/wa_alpha_sweep_analysis.txt`.

## Headline numbers (20 edit + 20 keep)
| α_copy | prior p_copy | E (should-EDIT corr.) | L (should-KEEP literal) | tracks-manip. (E_a>1−L_b) | junk>0.5 |
|--------|-------------|----------------------|------------------------|---------------------------|----------|
| 3 (baseline) | 0.50 | 0.09 | 0.75 | 15% | 10/40 |
| 10 | 0.77 | 0.20 | 0.80 | 20% | 8/40 |
| 50 | 0.94 | 0.18 | 0.73 | 25% | 10/40 |
| **100** | 0.97 | **0.25** | 0.73 | **35%** | **7/40** |

## The mechanism (INVERTS the §0 hypothesis — good news)
§0 predicted concentration would *suppress* editing (pick the knee before E craters). Instead, raising
α_copy **raises** E. At the edit-happy α=3 the dominant failure is not under-editing but the model bleeding
mass into spurious-insertion / over-edit **junk**, which steals probability from BOTH the literal reading
and the genuine correction. Concentrating the prior on copy kills that junk channel, freeing the mass to
flow to the real correction wherever the LM supports it. Decisive per-item evidence (item-level rows in the
sweep files): SUBN-01a "recieve"→"receive" **junk 1.00 @α=3 → E 0.99 @α=100**; DEL-of-01a "one the best"→"one
of the best" **E 0.00/junk 0.49 @α=3 → E 0.98 @α=100**; SUBW-01a "antidote"→"anecdote" junk 0.85→0.03 (E 0.67);
INS-02a "on on"→"on" E 0.00→0.75. Keep-retention holds flat (~0.73–0.80) — no over-editing of the plausible twins.

α=100 is best on the grid on E, tracks-manipulation, AND junk, with L flat. **Both E and tracks-manipulation
are still RISING at α=100 → the knee is at or above 100** (hence the §6 extension is well-motivated).

## Two caveats — neither is fixable by the prior α (do NOT chase them with α)
1. **70m semantic/structural ceiling (E≈0.00 at EVERY α):** DEL_TO, INS_TO, DEL_FOR, DEL_FROM (dative/argument
   cases), and **content-word** doublings ("handed handed"→"handed", surp too high under frequency-aware
   insertion — the documented INS_DUP × ins-cost tension). These need **410m** and/or the duplicate-aware
   channel, not a different prior. (Function-word doublings like "on on" DO work at α=100.)
2. **Particle-noise wobbles at P=128/seed=0 (NOT α trends):** LADDER-send-3 keep is L=1.00 @α=3/10/50 then
   cliffs to 0.00 *only* @α=100; SUBN-02a craters *only* @α=50. Sharp non-monotonic single-point collapses =
   impoverishment the `rejuv=gibbs` sweep is designed to cure → re-test with rejuv + a 2nd seed (below).
   SEPARATE α-invariant bug: **DEL-the-01b "we went to the store" is 100% junk at ALL α** — a 70m
   over-insertion on a short clean sentence (3s runs); unrelated to the prior, worth its own look.

## RESUME TOMORROW (the two runs the user wants), in priority order
Env reminders: ncgenjax conda env; **the Bash tool shell is zsh** — bare `$SUB` does NOT word-split, use
`${=SUB}` (this bit us once). `conda run` BUFFERS stdout, so each α's result file only populates when that
α's process exits (no incremental progress; ~12 min/α at this budget). Always redirect to a file, never pipe.
```bash
cd /Users/thomasclark/mit/noisy_channel_model
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh; SUB=$(cat planning/wa_alpha_subsample.txt)

# (A) §6 GRID EXTENSION (rejuv=off, ~22 min for 2 α): does E keep rising / where does it crater?
for A in 200,1,1,1 500,1,1,1; do
  NC_LM=EleutherAI/pythia-70m NC_ALPHA=$A NC_REJUV=off NC_VERBOSE=0 PYTHONPATH=src \
    conda run -n ncgenjax python -u -m genjax_port.calibration_word_action_smc 128 0 ${=SUB} \
    > planning/wa_alpha_sweep_${A//,/_}.txt 2>&1; done

# (B) §7 rejuv=gibbs CONFIRMATION at the chosen α (the DEPLOYMENT setting; slower, a few× per item).
#     Run at α=100 (and at the §6 winner if 200/500 wins). Expect it to CURE the LADDER/SUBN-02a noise
#     collapses and to test that the θ-posterior path doesn't reintroduce over-editing (the §5.5 caveat).
for A in 100,1,1,1; do
  NC_LM=EleutherAI/pythia-70m NC_ALPHA=$A NC_REJUV=gibbs NC_VERBOSE=0 PYTHONPATH=src \
    conda run -n ncgenjax python -u -m genjax_port.calibration_word_action_smc 128 0 ${=SUB} \
    > planning/wa_alpha_sweep_gibbs_${A//,/_}.txt 2>&1; done

# Re-analyze everything side by side (analyzer meta uses pair_id/family, identical across 70m/410m gates):
PYTHONPATH=src conda run -n ncgenjax python -m genjax_port.calibration_battery_analyze \
  planning/wa_alpha_sweep_3_1_1_1.txt planning/wa_alpha_sweep_10_1_1_1.txt \
  planning/wa_alpha_sweep_50_1_1_1.txt planning/wa_alpha_sweep_100_1_1_1.txt \
  planning/wa_alpha_sweep_200_1_1_1.txt planning/wa_alpha_sweep_500_1_1_1.txt \
  planning/wa_alpha_sweep_gibbs_100_1_1_1.txt > planning/wa_alpha_sweep_analysis.txt 2>&1
```
Optional 3rd check: 2nd seed (`128 1 …`) at α=100 to confirm the keep wobbles are noise. **Only after the
gibbs confirmation looks clean** promote the chosen α to the deployment default (consolidation §1b): flip
`ACTION_ALPHA_DEFAULT` in `pythia_word_caprop.py`, the `run_example_native.sh` default, and the `--selftest`
expectations. The synthetic battery here is NOT the reserved human hold-out (`data/`) — keep it sealed.

---

# RESULTS (2026-06-21, session 2) — §6 grid extension + §7 gibbs confirmation DONE

**Status: both RESUME runs are DONE and analyzed. The knee is α=200,1,1,1 (rejuv=off), and the
gibbs (θ-posterior, deployment) path is confirmed safe — it does NOT reintroduce over-editing.
Final promotion of α=200 is gated on the one remaining confirmation, gibbs@α=200 (RUNNING).**

## What was run (P=128, 70m, seed=0, dedup=True, band=2)
- §6 grid extension (rejuv=off): `planning/wa_alpha_sweep_{200,500}_1_1_1.txt` (~7 min each).
- §7 gibbs confirmation (rejuv=gibbs, deployment setting): `planning/wa_alpha_sweep_gibbs_100_1_1_1.txt`
  (~16 min). gibbs@α=200 launched after (the winner — see below); file
  `planning/wa_alpha_sweep_gibbs_200_1_1_1.txt`.
- Re-analyzed all 7 side by side → `planning/wa_alpha_sweep_analysis.txt`.

## Full headline numbers (20 edit + 20 keep)
| α_copy | prior p_copy | E (corr.) | L (retention) | tracks-manip | junk>0.5 | mean gap |
|--------|-------------|-----------|---------------|--------------|----------|----------|
| 3      | 0.50  | 0.09 | 0.75 | 15% | 10/40 | −0.16 |
| 10     | 0.77  | 0.20 | 0.80 | 20% | 8/40  | +0.00 |
| 50     | 0.94  | 0.18 | 0.73 | 25% | 10/40 | −0.08 |
| 100    | 0.97  | **0.25** | 0.73 | 35% | 7/40 | −0.02 |
| **200**| 0.985 | 0.22 | **0.91** | **40%** | **0/40** | **+0.13** |
| 500    | 0.994 | 0.08 | 0.86 | 30% | 4/40 | −0.06 |
| gibbs@100 | post. | 0.18 | 0.88 | 40% | 6/40 | +0.06 |

## The knee is α=200 (§6 conclusion)
α=200 wins on **all four** §6 criteria: highest retention L=0.91 (the user's primary concern — the
over-editing we're curing), **zero** junk (0/40, down from 7–10), best within-pair tracks-manipulation
40%, best gap +0.13 — while keeping E healthy at 0.22 (vs α=100's 0.25). At **α=500 the E craters to
0.08** (SUBW retention 0.50, SUBW E 0.16): over-concentration finally suppresses genuine correction,
so the §0-predicted "knee then crater" does exist — just far higher than §0 expected (knee≈200, not ≤10).
The E peak is broad over α≈100–200; α=200 is chosen because it dominates on L/junk/tracks at near-peak E.

## §7 gibbs confirmation is CLEAN — the θ-posterior path does not over-edit
gibbs@α=100 vs rejuv=off@α=100: E 0.25→0.18, **L 0.73→0.88**, tracks 35%→40%, gap −0.02→+0.06,
junk 7→6. The deployment (θ-posterior) path **raises** retention rather than reintroducing the §5.5
over-editing it was feared to — the conjugate refresh self-concentrates p_copy on the clean parse, so
gibbs@100 ≈ rejuv=off@200 in profile. It also **unlocks deletion corrections the prior-only path never
made**: DEL_FOR E 0.00→0.34, DEL_OF→0.49 (the sweep's word-restoration move + refresh). It partly cures
the earlier P=128 keep wobbles too: LADDER L 0.10→0.50, SUBN L→1.00. The §5.5 deletion mode-collapse
caveat does NOT bite on this battery.

## DECISION (gibbs@200 DONE — the deployment config wins outright)
**Chosen α = 200,1,1,1.** gibbs@α=200 (the final deployment-setting confirmation) is the BEST config on
the whole grid: E=0.21 (non-collapsed — NOT the α=500 crater), **L=0.99 (keep 20/20)**, tracks-manip
**55%**, gap **+0.20**, junk **0/40**. The §7 fear — that the conjugate refresh at the higher prior would
over-suppress E the way α=500 did at rejuv=off — did NOT materialize. SUBN 0.94 / SUBW 0.84 (sub
correction climbed); deletion-restoration is lower than gibbs@100 (DEL_FOR 0.34→0.01, DEL_OF 0.49→0.34:
higher p_copy ⇒ less word-restoration), a mild tension but net dominant. Caveat on 55%: with L_keep=0.99
the within-pair bar 1−L_b≈0.01 is trivially low, so part of 40→55% is near-perfect twin-retention, not
raw correction — itself the good news (zero over-editing).

**PROMOTED α=200 (consolidation §1b DONE — user chose full promotion 2026-06-21).** Edits in
`src/genjax_port/pythia_word_caprop.py`: `ACTION_ALPHA_DEFAULT` (3,1,1,1)→(200,1,1,1) w/ calibration
provenance comment; `--channel` CLI default char_copy→**word_action** (word_action is now the default
model); `main()`/`--selftest` rebuilt to smoke the word_action default (SUB+KEEP) alongside the char_copy
anchor (DEL/SUB/KEEP). **Verified: selftest 5/5 OK** (word_action SUB→"the cat sat on the mat", KEEP held;
char_copy DEL restores "to"); **full live suite 22/22 green** (all exact-enum cert anchors +
channel-selector + wa-rejuv + pythia smoke). `model_current.tex` reference values updated to (200,1,1,1) +
word_action default (compiles clean). `run_example_native.sh` is skip-worktree: already defaults to
word_action (WORD_ACTION=1) so it auto-picks-up ACTION_ALPHA_DEFAULT=200 — only its stale `(3,1,1,1)`
comments need a local touch (flagged to user, NOT committed). All edits UNCOMMITTED on branch
`word-action-rejuv` (user hasn't asked to commit).

## Evidence depth (added this session): slp_gain ⟷ E and the two ceilings
Joined the gated CSV's `slp_gain` (= 70m log P(intended)−log P(observed), the LM's own preference for the
grammatical sentence) against per-item E@α=200. Pearson r(slp_gain,E)=**0.55**. The 14 E≈0 pairs split
into TWO causes, not one LM ceiling: (1) ~5 genuine **LM-indifference** items (slp_gain<2 nats: LADDER,
DELTO-01, DEL-a-02 — mean E 0.01); (2) several **inference/channel-limited** items where the LM DOES
prefer grammatical (DELFROM-01 6.4 nats, INS-01 9.6 nats, DELTO-02 3.5 — all E=0) but the fix is a
word-restoration/deletion the rejuv=off channel doesn't execute. Class (2) is exactly what gibbs
rescues (DEL_FROM/DEL_FOR/DEL_OF 0→0.3–0.5 under rejuv) — NOT an LM limit. So the earlier blanket "70m
semantic ceiling" over-attributed to the LM: ~5 are LM-bound (need 410m), the rest are channel/rejuv-bound.
The persistent LM-bound families (DEL_TO datives, true content-word INS_DUP) still need 410m, not α.
