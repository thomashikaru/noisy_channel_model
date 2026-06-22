"""Summarize align-optimization SMC result files for the error-correction calibration goal.

Parses the per-item summary lines emitted by calibration_word_action_smc (format:
    ITEM  edit|keep  metric  q_ref  L  E  junk  'obs' -> 'intended' ...)
and reports, per file, the two quantities the GOAL maximizes:
    E_edit = mean correction mass (col E) over EDIT items   -> mass on the intended fix (implausible)
    L_keep = mean literal mass    (col L) over KEEP items   -> mass on the literal string (plausible)
plus junk and a single combined score (mean of E_edit and L_keep) for quick ranking.

Usage:  python planning/align_opt_summ.py LABEL=file.txt [LABEL2=file2.txt ...]
        (LABEL is optional; bare path uses the basename.)
"""
import re
import sys

LINE = re.compile(r"^(\S+)\s+(edit|keep)\s+(\S+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s")


def parse(path):
    edit, keep = {}, {}                      # item -> (L, E, junk)
    for ln in open(path):
        m = LINE.match(ln)
        if not m:
            continue
        item, exp, _metric, _qref, L, E, junk = m.groups()
        (edit if exp == "edit" else keep)[item] = (float(L), float(E), float(junk))
    return edit, keep


def agg(d, idx):
    return sum(v[idx] for v in d.values()) / len(d) if d else float("nan")


def main():
    runs = []
    for a in sys.argv[1:]:
        label, _, path = a.partition("=")
        if not path:
            path, label = label, label.rsplit("/", 1)[-1]
        runs.append((label, path, *parse(path)))

    print(f"{'config':28s} {'E_edit':>7s} {'L_keep':>7s} {'combined':>9s} "
          f"{'junkE':>6s} {'junkK':>6s} {'nE':>3s} {'nK':>3s}")
    print("-" * 86)
    for label, path, edit, keep in runs:
        E = agg(edit, 1)
        L = agg(keep, 0)
        comb = (E + L) / 2
        jE = sum(1 for v in edit.values() if v[2] > 0.5)
        jK = sum(1 for v in keep.values() if v[2] > 0.5)
        print(f"{label:28s} {E:7.3f} {L:7.3f} {comb:9.3f} "
              f"{jE:4d}/{len(edit):<1d} {jK:4d}/{len(keep):<1d} {len(edit):3d} {len(keep):3d}")

    # Per-item E (edit) and L (keep) deltas vs the first run, to see WHICH items move.
    if len(runs) >= 2:
        base = runs[0]
        for kind, idx, col in (("EDIT  E(correction mass)", 1, 2), ("KEEP  L(literal mass)", 0, 3)):
            print(f"\n  per-item {kind}, vs {base[0]}:")
            ref = base[col]
            items = sorted(ref)
            hdr = "    " + f"{'item':14s}" + "".join(f"{r[0][:9]:>10s}" for r in runs)
            print(hdr)
            for it in items:
                cells = ""
                for r in runs:
                    d = r[col]
                    cells += f"{d[it][idx]:10.2f}" if it in d else f"{'-':>10s}"
                print(f"    {it:14s}{cells}")


if __name__ == "__main__":
    main()
