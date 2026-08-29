"""clark2026 -- noisy-channel garden paths (this project's own materials; formerly "ncgp2").

``data/clark2026/materials.csv``: 36 items x 10 labels = 360 rows.  Each item has two critical
verbs that are one keystroke apart (``kicked`` / ``licked``), two predicates that make one verb
plausible and the other not, a control verb, a control predicate, and a typo'd version of each
critical verb::

    1.1       The boy kicked the big round ball into the net.        plausible
    1.2       The boy kicked the big round lollipop with delight.    implausible -> 2.2
    2.1       The boy licked the big round ball into the net.        implausible -> 1.1
    2.2       The boy licked the big round lollipop with delight.    plausible
    1.Control The boy kicked the big round breath after the run.     implausible, no counterpart
    2.Control The boy licked the big round breath after the run.     implausible, no counterpart
    Control.1 The boy read the big round ball into the net.          implausible, no counterpart
    Control.2 The boy read the big round lollipop with delight.      implausible, no counterpart
    Typo1.1   The boy kjcked the big round ball into the net.        -> 1.1
    Typo2.2   The boy ljcked the big round lollipop with delight.    -> 2.2

**Counterparts.**  The typo rows restore their verb.  ``2.1`` and ``1.2`` are the noisy-channel
garden paths: the sentence is implausible as written but one keystroke from a plausible reading
with the other critical verb.  ``1.2 -> 2.2`` is the mirror image of the ``2.1 -> 1.1`` case the
harness plan names as example 6, and is included on that symmetry; only the four control rows,
whose implausibility has no one-edit repair, are left without a counterpart.

**Hold-out.**  This is the one study in the set whose human data sits in the same directory.
``exp_data_merged.csv`` and ``lists/`` are named in ``common.HOLDOUT_PATHS`` and this converter
never touches them; ``raw_materials.csv`` (the per-item word lists the stimuli were built from)
is read only for ``meta``.
"""

from __future__ import annotations

from .common import StimRow, critical_index, normalize, read_source_csv, standardize, uid

#: label -> the label whose sentence a noisy-channel reader would recover ("" = none defined)
INTENDED_LABEL = {
    "Typo1.1": "1.1",
    "Typo2.2": "2.2",
    "2.1": "1.1",
    "1.2": "2.2",
    "1.1": "1.1",
    "2.2": "2.2",
    "1.Control": "",
    "2.Control": "",
    "Control.1": "",
    "Control.2": "",
}


def convert():
    raw = read_source_csv("data/clark2026/raw_materials.csv")
    rows = read_source_csv("data/clark2026/materials.csv")
    for r in rows:
        label = r["Label"]
        assert label in INTENDED_LABEL, f"unexpected clark2026 label {label!r}"
        target = INTENDED_LABEL[label]
        model_input = standardize(r["sentence"])
        source = raw[int(r["Item"])] if int(r["Item"]) < len(raw) else {}
        yield StimRow(
            dataset="clark2026",
            item_id=r["Item"],
            condition=label,
            sentence_orig=r["sentence"],
            sentence_norm=normalize(r["sentence"]),
            model_input=model_input,
            plausibility=r["Plausibility"],
            contrast=("typo" if label.startswith("Typo") else "word_form" if target else ""),
            intended_uid=uid("clark2026", r["Item"], target) if target else "",
            critical_word_idx=_critical_index(model_input, r["CriticalWord"]),
            meta={
                "critical_word": r["CriticalWord"],
                "critical_cond": r["CriticalCond"],
                "intervening_cond": r["InterveningCond"],
                "intervening": r["Intervening"],
                "preamble": r["Preamble"],
                "predicate": r["Predicate"],
                "id": r["ID"],
                "source_critical": {k: source.get(k, "") for k in
                                    ("Critical1", "Critical2", "Control", "Typo1", "Typo2")},
            },
        )


def _critical_index(model_input: str, critical_word: str) -> int:
    """Position of the item's critical verb, which always follows the preamble.

    Searched by identity rather than by a stored offset because the preamble length varies by
    item ("The boy" vs "In the").  Asserted rather than allowed to return None: every row's
    ``CriticalWord`` is present in its own sentence by construction.
    """
    idx = critical_index(model_input, critical_word)
    assert idx is not None, f"clark2026 critical word {critical_word!r} not in {model_input!r}"
    return idx
