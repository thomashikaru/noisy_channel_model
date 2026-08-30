"""gibson2013 -- Gibson, Bergen & Piantadosi (2013), PNAS 110(20).

Three subsets, each ``data/gibson2013/<subset>/materials.csv`` with columns
``Item,Type,Structure,Plausibility,Sentence,Question,Answer``: 20 items x 2 structures x
2 plausibility levels = 80 rows per subset, 240 in total.

**Intended counterpart.**  The design's point is that an implausible sentence sits ONE word
edit away from a plausible sentence of the *other* structure, and that the two directions of
that edit are not equally likely:

    DO_implausible  "The mother gave the candle the daughter."      -- insert "to"  -->
    PO_plausible    "The mother gave the candle to the daughter."

    PO_implausible  "The mother gave the daughter to the candle."   -- delete "to"  -->
    DO_plausible    "The mother gave the daughter the candle."

So ``intended_uid`` points at the plausible row of the opposite structure, which is what makes
``edit_type`` come out as a clean ``ins`` / ``del`` and what the harness's first two example
phenomena refer to.  The same-structure plausible sibling -- a role swap, not a one-word edit --
is kept in ``meta["same_structure_plausible_uid"]`` for anyone who wants that contrast instead.
Plausible rows are their own counterpart (``edit_type == "none"``).

**Question answers.**  ``Answer`` is the answer under the sentence as literally written, and
which participant the question asks about is counterbalanced across items: exactly half the
items in each Structure x Plausibility cell answer "Yes" and half "No".  It therefore cannot be
derived from the condition, and is carried through verbatim.

**transitive_intransitive** manipulates Preposition vs Transitive instead of PO vs DO; the same
cross-structure rule applies ("benefited from the businessman" -- delete "from" --> "benefited
the businessman").
"""

from __future__ import annotations

from .common import StimRow, normalize, read_source_csv, standardize, uid

SUBSETS = ("dopo_to", "dopo_for", "transitive_intransitive")

#: The structure a subset's minimal edit crosses to.
_OTHER_STRUCTURE = {"DO": "PO", "PO": "DO", "Preposition": "Transitive", "Transitive": "Preposition"}

#: The design axis each subset varies (see common.CONTRASTS).
_CONTRAST = {"dopo_to": "dative", "dopo_for": "dative",
             "transitive_intransitive": "transitivity"}


def convert():
    for subset in SUBSETS:
        yield from _convert_subset(subset)


def _convert_subset(subset: str):
    rows = read_source_csv(f"data/gibson2013/{subset}/materials.csv")
    for r in rows:
        structure, plaus = r["Structure"], r["Plausibility"]
        condition = f"{structure}_{plaus}"
        row_uid = lambda cond: uid("gibson2013", r["Item"], cond, subset)   # noqa: E731
        if plaus == "plausible":
            intended = row_uid(condition)                                    # already the target
        else:
            intended = row_uid(f"{_OTHER_STRUCTURE[structure]}_plausible")   # the one-word edit
        yield StimRow(
            dataset="gibson2013",
            subset=subset,
            item_id=r["Item"],
            condition=condition,
            sentence_orig=r["Sentence"],
            sentence_norm=normalize(r["Sentence"]),
            model_input=standardize(r["Sentence"]),
            plausibility=plaus,
            contrast=_CONTRAST[subset],
            intended_uids=[intended],
            comprehension_q=r["Question"],
            correct_answer=r["Answer"],
            meta={
                "structure": structure,
                "type": r["Type"],
                "same_structure_plausible_uid": row_uid(f"{structure}_plausible"),
            },
        )
