#!/usr/bin/env python3
"""Compare battery arms run with and without the inflectional edit class.

    python planning/morph_regression_compare.py <dir-with-morph{0,1}_seed{N}.txt>

The question is narrow: the class adds cheap edit routes to a channel whose K and alpha were
calibrated without them, so does it make the model edit-happy? The battery has no agreement
items, so the class can only ever cost here -- it cannot be validated by this run, only fail to
be condemned by it. INS_DUP (spurious-word removal) and CTRL (no-edit controls) are the families
that would show over-editing first.

The comparison is PAIRED: the same seeds run both arms, so an item that is unstable across seeds
is unstable in both and largely cancels. That matters because 45 of the 87 items give a different
MAP across 5 seeds (planning/calibration_seedcompare.csv) -- the per-item signal is close to a
coin flip, and only the aggregate and the paired difference mean anything.
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

# "SUBW-01a     edit    0.96   0.89  0.04 0.92 0.04  'obs' -> 'int'  (q_smc; 19s)"
ROW = re.compile(
    r"^(?P<item>[A-Z][A-Z_]*-\d+[ab]?)\s+(?P<exp>edit|keep)\s+"
    r"(?P<metric>[-\d.]+|nan)\s+(?P<qref>[-\d.]+|nan)\s+"
    r"(?P<L>[-\d.]+)\s+(?P<E>[-\d.]+)\s+(?P<junk>[-\d.]+)\s")


def parse(path: pathlib.Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        m = ROW.match(line)
        if m:
            d = m.groupdict()
            out[d["item"]] = {k: (float(v) if v not in (None,) and k not in ("item", "exp")
                                  else v) for k, v in d.items()}
    return out


def family(item: str) -> str:
    return item.split("-")[0]


def passed(rec: dict) -> bool:
    """The runner's own criterion: an edit item passes when the correction wins (metric > .5)."""
    metric = rec["metric"]
    return metric == metric and metric > 0.5


def main(dirname: str) -> None:
    d = pathlib.Path(dirname)
    seeds = sorted({int(p.stem.split("seed")[1]) for p in d.glob("morph*_seed*.txt")})
    arms = {(m, s): parse(d / f"morph{m}_seed{s}.txt") for m in (0, 1) for s in seeds
            if (d / f"morph{m}_seed{s}.txt").exists()}
    missing = [k for k in [(m, s) for m in (0, 1) for s in seeds] if k not in arms]
    if missing:
        print(f"WARNING: missing arms {missing}\n")

    items = sorted(set().union(*[set(a) for a in arms.values()]))
    print(f"{len(items)} items x {len(seeds)} seeds x 2 arms (morph off / on), rejuv=off, P=64\n")

    # --- aggregate pass counts, per arm ---------------------------------------------------
    print(f"{'arm':16s} {'pass':>6s} {'edit pass':>10s} {'keep pass':>10s} {'mean junk':>10s}")
    for (m, s), recs in sorted(arms.items()):
        ok = sum(passed(r) for r in recs.values())
        e = [r for r in recs.values() if r["exp"] == "edit"]
        k = [r for r in recs.values() if r["exp"] == "keep"]
        junk = [r["junk"] for r in recs.values() if r["junk"] == r["junk"]]
        print(f"morph={m} seed={s}   {ok:3d}/{len(recs):<3d} {sum(map(passed, e)):4d}/{len(e):<4d}  "
              f"{sum(map(passed, k)):4d}/{len(k):<4d}  {sum(junk) / max(len(junk), 1):10.3f}")

    # --- paired difference, per seed ------------------------------------------------------
    print("\nPAIRED per-item changes (same seed, morph off -> on):")
    all_flips = collections.Counter()
    for s in seeds:
        off, on = arms.get((0, s)), arms.get((1, s))
        if not (off and on):
            continue
        gained = [i for i in items if i in off and i in on and not passed(off[i]) and passed(on[i])]
        lost = [i for i in items if i in off and i in on and passed(off[i]) and not passed(on[i])]
        print(f"  seed {s}: +{len(gained)} gained, -{len(lost)} lost   net {len(gained) - len(lost):+d}")
        for i in gained:
            all_flips[("gained", i)] += 1
        for i in lost:
            all_flips[("lost", i)] += 1

    # --- what moved consistently across seeds ---------------------------------------------
    consistent = {k: v for k, v in all_flips.items() if v == len(seeds)}
    print(f"\nItems that flip the SAME WAY on every seed ({len(consistent)}):")
    for (kind, item), _ in sorted(consistent.items()):
        s = seeds[0]
        print(f"  {kind:7s} {item:12s} ({family(item)})  "
              f"metric {arms[(0, s)][item]['metric']:.2f} -> {arms[(1, s)][item]['metric']:.2f}")
    if not consistent:
        print("  (none -- every flip is seed-dependent, i.e. inside the noise)")

    # --- the families that matter for over-editing ----------------------------------------
    print("\nOver-editing watch (junk mass, the families that would show it first):")
    print(f"{'family':10s} {'n':>3s}  {'junk off':>9s} {'junk on':>9s} {'delta':>8s}")
    fams = collections.defaultdict(lambda: {0: [], 1: []})
    for (m, _s), recs in arms.items():
        for i, r in recs.items():
            if r["junk"] == r["junk"]:
                fams[family(i)][m].append(r["junk"])
    for fam in sorted(fams):
        a, b = fams[fam][0], fams[fam][1]
        if not (a and b):
            continue
        ja, jb = sum(a) / len(a), sum(b) / len(b)
        flag = "  <-- WORSE" if jb - ja > 0.05 else ""
        print(f"{fam:10s} {len(a) // max(len(seeds), 1):3d}  {ja:9.3f} {jb:9.3f} {jb - ja:+8.3f}{flag}")

    print("\nNoise floor: 45/87 items give a different MAP across 5 seeds; the binomial SE on 87\n"
          "items is ~4.6 items. Treat a net difference smaller than that as no difference.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
