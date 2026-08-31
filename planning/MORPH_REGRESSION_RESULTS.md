# Inflectional edit class — battery regression check (2026-08-30)

**Verdict: no detectable regression.** The class can stay on by default. `MORPH_LP` remains
uncalibrated; this run could not calibrate it and did not try.

## What was run

The 87-item calibration battery, `rejuv=off`, `P=64`, `channel=align`, as a **paired A/B**:
`NC_MORPH=0` (the channel as it was before the class existed) against `NC_MORPH=1`, on seeds 0
and 1. Four arms, ~12.5 min each. Logs in `morph_regression/`; analysis by
`morph_regression_compare.py`.

Paired because the eval is noisy: 45 of 87 items give a different MAP across 5 seeds
(`calibration_seedcompare.csv`), and the binomial SE on 87 items is ~4.6 items. Two seeds paired
beats more seeds unpaired — the same items are unstable in both arms and largely cancel.

## Results

| arm | pass | edit | keep | mean junk |
|---|---|---|---|---|
| morph=0 seed=0 | 45/87 | 9/43 | 36/44 | 0.255 |
| morph=0 seed=1 | 47/87 | 10/43 | 37/44 | 0.192 |
| morph=1 seed=0 | 45/87 | 10/43 | 35/44 | 0.255 |
| morph=1 seed=1 | 45/87 | 9/43 | 36/44 | 0.207 |

Paired net: **0** items on seed 0, **−2** on seed 1. Both inside the ~4.6-item noise floor.
**No item flips the same way on both seeds** — every flip is seed-dependent.

**The sharp check passes.** Seven items have no morphological alternant on any word, so both
hooks are provable no-ops; all seven are **bit-identical** across arms on both seeds. That
conclusion owes nothing to seed variance: the class does not leak.

**The over-editing families are clean.** `CTRL` junk 0.000 -> 0.000, `INS_DUP` 0.301 -> 0.299.
These are where a cheaper edit route would show up first, and they did not move.

## The one flag, and why it is not a regression

`DELTO` junk rose 0.112 -> 0.181 in aggregate. Per item it swings **both ways** — DELTO-01a
−0.61, DELTO-04a −0.39 (seed 0) then +0.97 (seed 1), DELTO-04b +0.94 on seed 1 while identical
on seed 0. It is dominated by two items on one seed.

The mechanism check settles it. DELTO-04a/b's only alternants are `judges` and `winners`, both
**one character** from the observed word — so SymSpell already retrieved them and the class adds
no candidate at all. All it changes is the emission: `logaddexp(K*1, MORPH_LP)` instead of `K*1`,
which at `K = MORPH_LP = -4.5` is an improvement of `ln 2` = **0.69 nats**. A 0.69-nat nudge
flipping a 0.98 keep to 0.03 means the item was already on a knife edge, which is exactly what
the 52% MAP instability describes.

## One real interaction, recorded

The class is **not purely additive**. `_candidate_words` ranks alternants just after the COPY, so
under the `Ke=12` cap they can displace character neighbours: **41 displacements across the whole
battery**. The displaced candidates are overwhelmingly tail junk, and at least one trade is
strictly good:

```
'was'   displaced ['wa']     for ['were']
'rich'  displaced ['trich']  for ['riches']
'wine'  displaced ['wing']   for ['wines']
'story' displaced ['start']  for ['stories']
```

Small and mostly benign, but it means a future `Ke` change interacts with the class.

## What this run cannot tell us

The battery has **no agreement items and no punctuation items**, so:

- the class was checked for collateral damage only — it could not be validated here;
- `MORPH_LP` could not be calibrated, and remains at its neutral default (one alternation costs
  one character edit);
- the comma pool addition is untested, because `rejuv=off` never consults the insertion pool.
  Testing it needs a `gibbs+bd` arm over comma items that do not exist yet.

Closing those needs new battery items, written to the blind-construction discipline in
`CALIBRATION_BATTERY_DRAFT.md` — by someone who has not read `data/qian2023/` or
`data/huang2024/`, which rules out the author of this note.
