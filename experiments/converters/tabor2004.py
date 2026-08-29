"""tabor2004 -- local coherence in English reduced relative clauses.

The original Tabor, Galantucci & Richardson (2004) appendix is not openly available.  These are
the items from Paape, Smith & Vasishth (2025), *Do local coherence effects exist in English
reduced relative clauses?* (JML 140), their direct English replication -- OSF ``f8qwh``,
``Materials/items.csv``, kept locally as ``data/tabor2004/items.csv`` with its provenance and
sha256 in ``data/tabor2004/SOURCE.md``.  The dataset is labelled ``tabor2004`` for the
phenomenon; cite the 2025 adaptation.

Semicolon-separated, 128 rows = 2 experiments x 16 items x 2 ``reduced_rel`` x 2 ``coherence``::

    nonreduced global  The coach smiled at the player who was thrown a frisbee by the opposing team.
    reduced    global  The coach smiled at the player thrown a frisbee by the opposing team.
    nonreduced local   The coach smiled at the player who was tossed a frisbee by the opposing team.
    reduced    local   The coach smiled at the player tossed a frisbee by the opposing team.

``coherence`` names whether the participle also reads as a plain past-tense verb, which is what
makes "the player tossed a frisbee" locally coherent.  The intended counterpart of each row is
its ``nonreduced`` sibling at the same coherence level, so the reduced rows come out as a single
``ins`` restoring the relativizer.  ``item_nr`` is already experiment-qualified (``e1_1``,
``e2_1``), so it is unique across the 32 items; ``item_type`` is kept in ``meta`` anyway.

``critical_word_idx`` is the relative-clause participle.  The file has no column for it, so it
is derived from the reduced/nonreduced pair: the two versions differ by exactly one insertion of
a relativizer plus a copula, and the participle is the token right after it.  Which relativizer
varies across the 64 pairs -- "who was" (41), "that was" (14), "who were" (3), "which was" (2),
"who is" (2), "that were" (2) -- so it is read off the diff rather than assumed, and the
one-insertion shape is asserted on every pair.

Fillers (OSF ``wr28g``) are excluded by decision and were never fetched.
"""

from __future__ import annotations

from .common import (StimRow, classify_edit, normalize, read_source_csv, standardize,
                     strip_punct, uid)


def convert():
    rows = read_source_csv("data/tabor2004/items.csv", delimiter=";")
    by_cell = {(r["item_nr"], r["coherence"], r["reduced_rel"]): r["sentence"] for r in rows}
    for r in rows:
        condition = f"{r['reduced_rel']}_{r['coherence']}"
        model_input = standardize(r["sentence"])
        cell = (r["item_nr"], r["coherence"])
        relativizer, red_idx = _relativizer(standardize(by_cell[(*cell, "reduced")]),
                                            standardize(by_cell[(*cell, "nonreduced")]), cell)
        # The participle sits right where the relativizer would go, so in the nonreduced version
        # it is that many tokens further along.
        idx = red_idx if r["reduced_rel"] == "reduced" else red_idx + len(relativizer.split())
        toks = model_input.split()
        participle = strip_punct(toks[idx]).lower()
        yield StimRow(
            dataset="tabor2004",
            item_id=r["item_nr"],
            condition=condition,
            sentence_orig=r["sentence"],
            sentence_norm=normalize(r["sentence"]),
            model_input=model_input,
            contrast="relativizer",
            intended_uid=uid("tabor2004", r["item_nr"], f"nonreduced_{r['coherence']}"),
            critical_word_idx=idx,
            comprehension_q=r["question"],
            correct_answer=r["corr_ans"],
            meta={
                "experiment": r["item_type"],
                "reduced_rel": r["reduced_rel"],
                "coherence": r["coherence"],
                "participle": participle,
                "relativizer": relativizer,
                "src_orc": r["src_orc"],
                "answer_options": [r["ans1"], r["ans2"], r["ans3"]],
            },
        )


def _relativizer(reduced: str, nonreduced: str, cell) -> tuple[str, int]:
    """The words the reduced version drops, and the reduced-token index they would go at."""
    edit = classify_edit(reduced, nonreduced)
    assert edit.type == "ins" and len(edit.to.split()) == 2, (
        f"tabor2004 {cell}: the reduced and nonreduced versions should differ by exactly one "
        f"inserted relativizer + copula, but the diff is {edit.type}/{edit.ops} -> {edit.to!r}\n"
        f"  reduced    = {reduced!r}\n  nonreduced = {nonreduced!r}")
    return edit.to, edit.obs_gap
