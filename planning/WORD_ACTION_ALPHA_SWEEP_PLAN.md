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
