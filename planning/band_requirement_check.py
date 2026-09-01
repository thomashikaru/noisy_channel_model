#!/usr/bin/env python
"""Which intended repairs need which band? (planning/OFF_ARM_INFERENCE_FIX.md sec 9, the
band=1 question.)

The forward DP constrains the alignment drift |observed units consumed - intended words
emitted| <= band at every step, so a repair is representable at band b only if the word-level
alignment between the observed sentence and the intended sentence never drifts more than b.
This probe computes that required band for EVERY (stimulus, intended repair) row in
experiments/stimuli/*.repairs.csv and for the 87-item calibration battery, using the model's
own unit segmentation (pythia_word_caprop._obs_word_units), and reports the distribution plus
every row that needs band >= 2. A static capability check, not a model run -- the same spirit
as experiments/reachability.py.

Drift is measured at difflib block boundaries (within an unequal replace block the ops can be
interleaved, so the extremum is at a boundary). Case differences do not move the alignment.

    conda run -n ncgenjax python planning/band_requirement_check.py
"""
import csv
import collections
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from genjax_port import pythia_word_caprop as pwc  # noqa: E402

STIMULI = REPO / "experiments" / "stimuli"


def units(text):
    return [u.lower() for u in pwc._obs_word_units(text.strip())]


def required_band(observed, intended):
    a, b = units(observed), units(intended)
    drift = 0
    worst = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        drift += (i2 - i1) - (j2 - j1)
        worst = max(worst, abs(drift))
    return worst


def main():
    rows = []
    for f in sorted(STIMULI.glob("*.repairs.csv")):
        ds = f.name.split(".")[0]
        stim = {r["stim_uid"]: r for r in csv.DictReader((STIMULI / f"{ds}.stimuli.csv").open())}
        for e in csv.DictReader(f.open()):
            if e["edit_type"] == "none" or not e["intended_text"].strip():
                continue
            obs = stim[e["stim_uid"]]["model_input"]
            rows.append(dict(dataset=ds, stim_uid=e["stim_uid"], edit_type=e["edit_type"],
                             observed=obs, intended=e["intended_text"],
                             band=required_band(obs, e["intended_text"])))
    for r in csv.DictReader(open(REPO / "planning" / "calibration_battery_v0.csv")):
        if r.get("expected") == "edit" and r["intended"].strip():
            rows.append(dict(dataset="battery_v0", stim_uid=r["item_id"], edit_type="",
                             observed=r["observed"], intended=r["intended"],
                             band=required_band(r["observed"], r["intended"])))

    print(f"{len(rows)} (stimulus, intended repair) rows with a real edit\n")
    print(f"{'dataset':>12}  band=1-safe   needs band>=2   (repair rows by required band)")
    offenders = []
    by_stim_max = collections.defaultdict(list)
    for ds in sorted({r['dataset'] for r in rows}):
        g = [r for r in rows if r["dataset"] == ds]
        dist = collections.Counter(r["band"] for r in g)
        bad = [r for r in g if r["band"] >= 2]
        offenders += bad
        print(f"{ds:>12}  {len(g)-len(bad):>6}/{len(g):<6} {len(bad):>8}        {dict(sorted(dist.items()))}")
        for r in g:
            by_stim_max[(ds, r["stim_uid"])].append(r["band"])

    all_need2 = [(k, v) for k, v in by_stim_max.items() if min(v) >= 2]
    some_need2 = [(k, v) for k, v in by_stim_max.items() if max(v) >= 2 and min(v) < 2]
    print(f"\nstimuli where ALL intended repairs need band>=2: {len(all_need2)}"
          f"   (SOME but not all: {len(some_need2)})")
    if offenders:
        print("\nrows needing band>=2 (band=1 would make these repairs unrepresentable):")
        for r in offenders[:15]:
            print(f"  [{r['dataset']}] band={r['band']} {r['stim_uid']}")
            print(f"      obs: {r['observed'][:84]}")
            print(f"      int: {r['intended'][:84]}")
        if len(offenders) > 15:
            print(f"  ... and {len(offenders) - 15} more")
        out = REPO / "planning" / "band_requirement_offenders.csv"
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(offenders[0].keys()))
            w.writeheader(); w.writerows(offenders)
        print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
