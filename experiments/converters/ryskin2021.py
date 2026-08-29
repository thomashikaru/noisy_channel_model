"""ryskin2021 -- Ryskin, Futrell, Kiran & Gibson (2018/2021): form-based error correction.

``data/ryskin2021/materials.csv`` (``Item,Condition,sentence``): 126 items x 4 conditions = 504
rows.  Item ids run 1..159 with gaps, so they are carried as strings and never assumed dense.

Conditions, all relative to ``Control``:

    Control   "...to prevent an infection."      the intended sentence
    SemCrit   "...to prevent an inflection."     a form neighbour that is also a real word
    Sem       "...to prevent a rhinestone."      a semantically wrong but well-formed word
    Synt      "...to prevent an infections."     a morphological/agreement variant

``Control`` is the intended counterpart for all four (and its own, giving ``edit_type ==
"none"``).  The source is lowercase with punctuation split off, so ``model_input`` is where the
capital and the attached period come from.
"""

from __future__ import annotations

from .common import StimRow, normalize, read_source_csv, standardize, uid

CONDITIONS = ("Control", "SemCrit", "Sem", "Synt")


def convert():
    rows = read_source_csv("data/ryskin2021/materials.csv")
    for r in rows:
        assert r["Condition"] in CONDITIONS, f"unexpected ryskin condition {r['Condition']!r}"
        yield StimRow(
            dataset="ryskin2021",
            item_id=r["Item"],
            condition=r["Condition"],
            sentence_orig="",                       # lost upstream: the source is already normalized
            sentence_norm=normalize(r["sentence"]),
            model_input=standardize(r["sentence"]),
            plausibility="plausible" if r["Condition"] == "Control" else "implausible",
            contrast="word_form",
            intended_uid=uid("ryskin2021", r["Item"], "Control"),
        )
