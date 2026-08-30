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

from genjax_port import calibration_gate, morphology, pythia_word_caprop  # noqa: E402


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


def _punctuation_delta(edit_from: str, edit_to: str) -> set[str]:
    """The punctuation characters one side has and the other lacks."""
    pa = {c for c in edit_from if not c.isalnum()}
    pb = {c for c in edit_to if not c.isalnum()}
    return pa ^ pb


def classify(edit_from: str, edit_to: str, max_dist: int) -> tuple[str, int | None]:
    """Return (verdict, char_distance) for one word-level repair.

    Three routes count as reachable, because the channel now has three:

    * ``reachable`` -- the character substitution channel proposes it (SymSpell within max_dist).
    * ``reachable_morph`` -- the words stand in the inflectional-alternation relation, so the
      morphological edit class proposes it at a flat rate regardless of character distance.
    * ``reachable_insertion`` -- the two differ only by a punctuation unit that is in the indel
      move's insertion pool. These look like substitutions at the whitespace-token level
      ("changed" -> "changed,") but the model segments punctuation as its own unit, so restoring
      one is a word INSERTION, not a substitution.
    """
    if len(edit_from.split()) != 1 or len(edit_to.split()) != 1:
        return ("multiword", None)          # a voice alternation, not a word substitution
    a, b = _body(edit_from), _body(edit_to)
    if a == b:
        delta = _punctuation_delta(edit_from, edit_to)
        if delta and delta <= set(pythia_word_caprop.FUNCWORDS):
            return ("reachable_insertion", 0)
        return ("punctuation_only", 0)
    if morphology.alternates(a, b):
        return ("reachable_morph", None)
    ok, dist, why = calibration_gate.reachable(a, b, max_dist=max_dist)
    return ("reachable" if ok else why, dist)


def probe_dataset(name: str, max_dist: int) -> dict:
    """One verdict per (stimulus, repair) -- a stimulus with two repairs gets two."""
    stim = {r["stim_uid"]: r for r in csv.DictReader((STIMULI / f"{name}.stimuli.csv").open())}
    repairs = list(csv.DictReader((STIMULI / f"{name}.repairs.csv").open()))
    per_condition: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    unreachable: list[dict] = []
    reachable_by_stim: dict[str, bool] = {}
    for e in repairs:
        if e["edit_type"] != "sub":
            continue
        r = stim[e["stim_uid"]]
        verdict, dist = classify(e["edit_from"], e["edit_to"], max_dist)
        key = f"{r['subset']}/{r['condition']}" if r["subset"] else r["condition"]
        per_condition[key][verdict] += 1
        good = verdict.startswith("reachable")
        reachable_by_stim[e["stim_uid"]] = reachable_by_stim.get(e["stim_uid"], False) or good
        if not good:
            unreachable.append({"stim_uid": e["stim_uid"], "intended_uid": e["intended_uid"],
                                "verdict": verdict, "from": e["edit_from"], "to": e["edit_to"],
                                "char_dist": dist})
    totals = collections.Counter()
    for counts in per_condition.values():
        totals.update(counts)
    n_reachable = sum(v for k, v in totals.items() if k.startswith("reachable"))
    return {
        "n_sub_rows": sum(totals.values()),
        "n_reachable": n_reachable,
        "n_stimuli_with_any_reachable_repair": sum(reachable_by_stim.values()),
        "n_stimuli_with_a_sub_repair": len(reachable_by_stim),
        "verdicts": dict(sorted(totals.items())),
        "per_condition": {k: dict(sorted(v.items())) for k, v in sorted(per_condition.items())},
        "unreachable": unreachable,
    }


def qian_alternative_route(max_dist: int) -> dict:
    """qian2023 admits two repairs; report both, because they are not equally reachable.

    An ungrammatical row like "The gifts for the kid is hidden under the bed." can be repaired
    two ways, and the stimuli encode only the first:

      A  fix the VERB   -> "The gifts for the kid are hidden..."   (is -> are)
      B  fix the NOUN   -> "The gift  for the kid is  hidden..."   (gifts -> gift)

    Both are legitimate noisy-channel readings, and the stimuli carry both with no primacy --
    the sentence really is ambiguous about which word is wrong. They differ sharply in whether
    the substitution channel can propose them at all, which is a fact about the model, not a
    reason to treat one as correct.
    """
    stim = {r["stim_uid"]: r for r in csv.DictReader((STIMULI / "qian2023.stimuli.csv").open())}
    out = collections.Counter()
    seen: set[str] = set()
    for e in csv.DictReader((STIMULI / "qian2023.repairs.csv").open()):
        cond = stim[e["stim_uid"]]["condition"]
        if cond[0] == cond[2]:                       # already grammatical: one repair, itself
            continue
        if e["stim_uid"] not in seen:
            seen.add(e["stim_uid"]); out["ungrammatical_stimuli"] += 1
        target_cond = e["intended_uid"].rsplit("/", 1)[1]
        route = "A_verb" if target_cond == cond[0] + cond[1] + cond[0] else "B_noun"
        ok = classify(e["edit_from"], e["edit_to"], max_dist)[0].startswith("reachable")
        out[f"route_{route}_reachable"] += ok
        out[f"route_{route}_total"] += 1
    return dict(out)


def main() -> None:
    max_dist = deployed_max_dist()
    datasets = sorted(p.name.split(".")[0] for p in STIMULI.glob("*.stimuli.csv"))
    datasets = [d for d in datasets if d not in ("smoke", "probe")]

    report = {"max_dist": max_dist, "lm_candidate_source": "noise_word.word_sub_candidates "
                                                          "(SymSpell + wordfreq multi-token)",
              "datasets": {}}
    for name in datasets:
        res = probe_dataset(name, max_dist)
        report["datasets"][name] = res
        n, ok = res["n_sub_rows"], res["n_reachable"]
        if not n:
            print(f"{name:12s} no single-substitution repairs")
            continue
        any_ok, n_stim = res["n_stimuli_with_any_reachable_repair"], res["n_stimuli_with_a_sub_repair"]
        extra = (f"   |  {any_ok}/{n_stim} stimuli have >=1 reachable repair"
                 if n_stim != n else "")
        print(f"{name:12s} {ok:4d}/{n:<4d} repairs reachable ({100 * ok / n:3.0f}%)"
              f"   {res['verdicts']}{extra}")
        for cond, counts in res["per_condition"].items():
            tot = sum(counts.values())
            ok_c = sum(v for k, v in counts.items() if k.startswith("reachable"))
            print(f"             {cond:24s} {ok_c:3d}/{tot:<3d}  "
                  f"{ {k: v for k, v in counts.items()} }")

    report["qian2023_repair_routes"] = qian_alternative_route(max_dist)
    print(f"\nqian2023 repair routes: {report['qian2023_repair_routes']}")

    out = STIMULI / "reachability.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
