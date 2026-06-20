"""Analyze word-action battery runs through the lens that MATTERS (per the user, 2026-06-19): not
"how many items cleared a fixed threshold" but "which setting aligns the model with our intuitions" --
i.e. does the model concentrate its CORRECTION mass on the implausible (should-edit) member of each
matched pair and leave the plausible (should-keep) twin mostly literal, with a sensible GRADIENT
between them? Thresholds (>0.9 literal / >0.5 corrected) are reference points, not pass/fail gates;
item-to-item scatter is expected.

The battery is matched pairs (``pair_id``): the ``edit`` member is implausible-in-context (a slip the
reader should infer and correct), the ``keep`` member is the plausible control (read literally). The
key statistic is the WITHIN-PAIR correction-mass separation ``E_edit - E_keep`` -- does the model track
the plausibility manipulation? -- reported alongside absolute levels (so we can see whether a setting
is correcting everything, nothing, or the right things) and the leading-opener junk it should ignore.

Usage:  PYTHONPATH=src conda run -n ncgenjax python -m genjax_port.calibration_battery_analyze \
            planning/wa_battery_gibbs_27_1_1_1.txt planning/wa_battery_gibbs_54_1_1_1.txt
Each arg is a result file written by calibration_word_action_smc (one fixed-width row per item:
``item exp metric q_ref L E junk  ...``). Multiple files are shown side by side so the concentration
gradient is visible.
"""
import csv
import re
import sys
from collections import defaultdict

CSV = "planning/calibration_battery_v0_gated_410m.csv"
# item_id exp metric q_ref L E junk  (the first 7 whitespace fields; trailing obs->intended ignored)
ROW = re.compile(r"^(\S+)\s+(edit|keep)\s+(nan|-?[\d.]+)\s+(nan|-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s")


def _f(x):
    return float("nan") if x == "nan" else float(x)


def parse(path):
    """path -> {item_id: dict(exp, metric, L, E, junk)} (E = correction mass, the axis we compare)."""
    out = {}
    for line in open(path):
        m = ROW.match(line)
        if m:
            iid, exp, metric, _qref, L, E, junk = m.groups()
            out[iid] = {"exp": exp, "metric": _f(metric), "L": float(L), "E": float(E), "junk": float(junk)}
    return out


def analyze(meta, res, label):
    items = [iid for iid in res if iid in meta]
    edit = [iid for iid in items if res[iid]["exp"] == "edit"]
    keep = [iid for iid in items if res[iid]["exp"] == "keep"]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    # The two SEPARATE goods (not to be subtracted): correction mass E on should-edit items (model infers
    # the slip), and literal mass L on should-keep items (model leaves plausible text alone). For keeps
    # intended==observed so E==L there; the spurious-edit/drift rate on a keep is 1-L.
    corr_edit = mean([res[i]["E"] for i in edit])              # should-EDIT: correction rate (high good)
    lit_keep = mean([res[i]["L"] for i in keep])               # should-KEEP: literal retention (high good)
    spur_keep = mean([1 - res[i]["L"] for i in keep])          # should-KEEP: spurious-edit/drift (low good)

    # Matched-pair contrast (does the model TRACK the plausibility manipulation): within a pair, the
    # correction mass on the implausible member E_a vs the spurious-edit rate on the plausible twin
    # (1 - L_b). Aligns with intuition iff E_a > (1 - L_b): edits the implausible far more than it
    # wrongly edits the plausible.
    by_pair = defaultdict(dict)
    for iid in items:
        by_pair[meta[iid]["pair_id"]][res[iid]["exp"]] = res[iid]
    pairs = [(d["edit"]["E"], 1 - d["keep"]["L"]) for d in by_pair.values() if "edit" in d and "keep" in d]
    gaps = [ea - eb for ea, eb in pairs]
    pos = sum(g > 0 for g in gaps)

    print(f"\n### {label}   ({len(items)} items: {len(edit)} edit / {len(keep)} keep)")
    print(f"  should-EDIT  correction rate E   mean {corr_edit:.2f}   (high = infers real slips)")
    print(f"  should-KEEP  literal retention L mean {lit_keep:.2f}   spurious-edit (1-L) {spur_keep:.2f}"
          f"   (high L / low 1-L = leaves clean text alone)")
    if pairs:
        print(f"  matched pairs ({len(pairs)}): E_implausible {mean([p[0] for p in pairs]):.2f} vs "
              f"spurious_plausible {mean([p[1] for p in pairs]):.2f};  tracks manipulation (E_a>1-L_b) in "
              f"{pos}/{len(pairs)} ({100*pos/len(pairs):.0f}%);  mean gap {mean(gaps):+.2f}")
    jhi = sum(res[i]["junk"] > 0.5 for i in items)
    print(f"  leading-opener/over-edit junk >0.5: {jhi}/{len(items)} (separate prime artifact, not the rate)")

    fams = sorted({meta[i]["family"] for i in items})
    print("  by family:  " + "  ".join(
        f"{fam}[E_edit {mean([res[i]['E'] for i in edit if meta[i]['family']==fam]):.2f}/"
        f"L_keep {mean([res[i]['L'] for i in keep if meta[i]['family']==fam]):.2f}]" for fam in fams))


def main():
    meta = {r["item_id"]: r for r in csv.DictReader(open(CSV))}
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: ... calibration_battery_analyze RESULT_FILE [RESULT_FILE ...]")
    for p in paths:
        analyze(meta, parse(p), p.split("/")[-1])


if __name__ == "__main__":
    main()
