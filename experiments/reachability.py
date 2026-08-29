#!/usr/bin/env python3
"""Can the deployed channel even propose each dataset's intended repair?

    conda run -n ncgenjax python experiments/reachability.py

A capability probe, not a model run. For every stimulus whose intended counterpart is one word
substitution away, it asks whether the production candidate generator actually surfaces the
intended word at the deployed ``max_dist=2`` -- the same question ``calibration_gate.reachable``
(gate G2) asks of the calibration battery, applied to the experiment stimuli.

Why this runs BEFORE the cluster: an unreachable repair is not a result. If the channel cannot
propose the intended word, the posterior cannot put mass on it, and a row that fails for that
reason says nothing about whether the model's inferences match people's. Knowing which rows are
in that position is what separates "the model disagrees" from "the model was never asked".
Reading it off a 3-second probe beats reading it off a two-day run.

Writes ``stimuli/reachability.json``. Needs jax (hence the ncgenjax env), which is why it is a
separate step: ``build_stimuli.py`` stays stdlib-only.
"""

from __future__ import annotations

import collections
import csv
import difflib
import inspect
import json
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS))

from genjax_port import calibration_gate, pythia_word_caprop  # noqa: E402


def deployed_max_dist() -> int:
    """The max_dist the model actually runs with, read off its own signature.

    Taken from ``pythia_word_caprop.run`` rather than written down here, so this probe cannot
    quietly disagree with the channel it is probing.  ``configs/main.env`` sets MAX_DIST to the
    same value; if that ever diverges, the config wins for the run and this number is wrong.
    """
    return inspect.signature(pythia_word_caprop.run).parameters["max_dist"].default

STIMULI = EXPERIMENTS / "stimuli"


def _body(word: str) -> str:
    return "".join(c for c in word if c.isalpha()).lower()


def classify(edit_from: str, edit_to: str, max_dist: int) -> tuple[str, int | None]:
    """Return (verdict, char_distance) for one word-level repair."""
    if len(edit_from.split()) != 1 or len(edit_to.split()) != 1:
        return ("multiword", None)          # a voice alternation, not a word substitution
    a, b = _body(edit_from), _body(edit_to)
    if a == b:
        return ("punctuation_only", 0)      # huang2024's comma: not a word substitution at all
    ok, dist, why = calibration_gate.reachable(a, b, max_dist=max_dist)
    return ("reachable" if ok else why, dist)


def probe_dataset(path: Path, max_dist: int) -> dict:
    rows = list(csv.DictReader(path.open()))
    per_condition: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    unreachable: list[dict] = []
    for r in rows:
        if r["edit_type"] != "sub":
            continue
        verdict, dist = classify(r["edit_from"], r["edit_to"], max_dist)
        key = f"{r['subset']}/{r['condition']}" if r["subset"] else r["condition"]
        per_condition[key][verdict] += 1
        if verdict != "reachable":
            unreachable.append({"stim_uid": r["stim_uid"], "verdict": verdict,
                                "from": r["edit_from"], "to": r["edit_to"], "char_dist": dist})
    totals = collections.Counter()
    for counts in per_condition.values():
        totals.update(counts)
    return {
        "n_sub_rows": sum(totals.values()),
        "n_reachable": totals["reachable"],
        "verdicts": dict(sorted(totals.items())),
        "per_condition": {k: dict(sorted(v.items())) for k, v in sorted(per_condition.items())},
        "unreachable": unreachable,
    }


def qian_alternative_route(max_dist: int) -> dict:
    """qian2023 admits two repairs; report both, because they are not equally reachable.

    An ungrammatical row like "The gifts for the kid is hidden under the bed." can be repaired
    two ways, and the stimuli encode only the first:

      A  fix the VERB   -> "The gifts for the kid are hidden..."   (is -> are, what intended_uid says)
      B  fix the NOUN   -> "The gift  for the kid is  hidden..."   (gifts -> gift)

    Both are legitimate noisy-channel readings. They differ sharply in whether the substitution
    channel can propose them at all, so which one the analysis treats as "the" intended sentence
    decides what a qian2023 result means.
    """
    rows = list(csv.DictReader((STIMULI / "qian2023.stimuli.csv").open()))
    by = {(r["item_id"], r["condition"]): r for r in rows}
    out = collections.Counter()
    for r in rows:
        cond = r["condition"]
        if cond[0] == cond[2]:                       # already grammatical
            continue
        out["ungrammatical_rows"] += 1
        if classify(r["edit_from"], r["edit_to"], max_dist)[0] == "reachable":
            out["route_A_verb_reachable"] += 1
        alt = by[(r["item_id"], cond[2] + cond[1] + cond[2])]
        ow, iw = r["model_input"].split(), alt["model_input"].split()
        ops = [o for o in difflib.SequenceMatcher(a=ow, b=iw, autojunk=False).get_opcodes()
               if o[0] != "equal"]
        if len(ops) == 1 and ops[0][0] == "replace" and ops[0][2] - ops[0][1] == 1:
            frm, to = ow[ops[0][1]], iw[ops[0][3]]
            if classify(frm, to, max_dist)[0] == "reachable":
                out["route_B_noun_reachable"] += 1
    return dict(out)


def main() -> None:
    max_dist = deployed_max_dist()
    datasets = sorted(p.name.split(".")[0] for p in STIMULI.glob("*.stimuli.csv"))
    datasets = [d for d in datasets if d not in ("smoke", "probe")]

    report = {"max_dist": max_dist, "lm_candidate_source": "noise_word.word_sub_candidates "
                                                          "(SymSpell + wordfreq multi-token)",
              "datasets": {}}
    for name in datasets:
        res = probe_dataset(STIMULI / f"{name}.stimuli.csv", max_dist)
        report["datasets"][name] = res
        n, ok = res["n_sub_rows"], res["n_reachable"]
        if not n:
            print(f"{name:12s} no single-substitution repairs")
            continue
        print(f"{name:12s} {ok:4d}/{n:<4d} reachable ({100 * ok / n:3.0f}%)   {res['verdicts']}")
        for cond, counts in res["per_condition"].items():
            tot = sum(counts.values())
            print(f"             {cond:24s} {counts.get('reachable', 0):3d}/{tot:<3d}  "
                  f"{ {k: v for k, v in counts.items() if k != 'reachable'} or ''}")

    report["qian2023_repair_routes"] = qian_alternative_route(max_dist)
    print(f"\nqian2023 repair routes: {report['qian2023_repair_routes']}")

    out = STIMULI / "reachability.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
