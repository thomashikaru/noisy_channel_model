"""moses -- the Moses illusion demo item.

``data/moses/materials.csv``: one sentence, "How many animals of each kind did Moses take on the
ark?"  Not a study; it is kept in the harness as a single-item sanity check that a question-final
input survives the pipeline (the only ``?`` outside clark2026), and because the semantic
illusion it names -- readers answer "two" despite the sentence saying Moses rather than Noah --
is the same kind of over-tolerant comprehension the noisy-channel model is meant to explain.

No conditions, no counterpart.
"""

from __future__ import annotations

from .common import StimRow, normalize, read_source_csv, standardize


def convert():
    for r in read_source_csv("data/moses/raw_materials.csv"):
        yield StimRow(
            dataset="moses",
            item_id=r["Item"],
            condition="demo",
            sentence_orig=r["sentence"],
            sentence_norm=normalize(r["sentence"]),
            model_input=standardize(r["sentence"]),
        )
