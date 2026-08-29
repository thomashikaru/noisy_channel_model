"""huang2024 -- the SAP Benchmark's ClassicGP set (Huang, Arehalli, Vasishth, Linzen et al.).

``data/huang2024/items_ClassicGP.pivot.csv``: 24 items x 6 conditions = 144 rows, three
garden-path constructions crossed with ambiguity::

    NPS   "The suspect showed the file deserved further investigation..."       (AMB)
          "The suspect showed that the file deserved further investigation..."  (UAMB)
    NPZ   "Because the suspect changed the file deserved..."                    (AMB)
          "Because the suspect changed, the file deserved..."                   (UAMB)
    MVRR  "The suspect sent the file deserved further investigation..."         (AMB)

The unambiguous sibling is the intended counterpart.  For NPS that is a restored "that" and for
NPZ a restored comma, which is exactly hypothesis 3 in the harness plan: the model may resolve
these garden paths by inferring a missing unit rather than by reanalysis.

``critical_word_idx`` is the disambiguating word.  The stored ``disambPosition_0idx`` indexes
the source sentence's whitespace tokens, and standardization leaves token positions alone
(punctuation stays attached, so "changed," is one token), so the stored index carries over to
``model_input`` unchanged.  Rather than trust that, the converter checks it: the token at that
index must be the same word as the one the item's AMB sibling disambiguates on.  That check
passes on all 144 rows and would fail loudly if the pivot file were ever regenerated with the
comma split off, which is where the off-by-one lives.
"""

from __future__ import annotations

from .common import StimRow, normalize, read_source_csv, standardize, strip_punct, uid

CONSTRUCTIONS = ("NPS", "NPZ", "MVRR")


def convert():
    rows = read_source_csv("data/huang2024/items_ClassicGP.pivot.csv")
    disamb_word = {}                                   # (item, construction) -> the AMB disambiguator
    for r in rows:
        construction, ambiguity = r["condition"].rsplit("_", 1)
        if ambiguity == "AMB":
            toks = standardize(r["Sentence"]).split()
            disamb_word[(r["item"], construction)] = strip_punct(
                toks[int(r["disambPosition_0idx"])]).lower()

    for r in rows:
        construction, ambiguity = r["condition"].rsplit("_", 1)
        assert construction in CONSTRUCTIONS, f"unexpected huang construction {construction!r}"
        model_input = standardize(r["Sentence"])
        idx = int(r["disambPosition_0idx"])
        toks = model_input.split()
        expected = disamb_word[(r["item"], construction)]
        got = strip_punct(toks[idx]).lower() if idx < len(toks) else None
        assert got == expected, (
            f"huang2024 item {r['item']} {r['condition']}: disambPosition_0idx={idx} points at "
            f"{got!r}, but the item's disambiguating word is {expected!r}. The stored index no "
            f"longer matches this sentence's tokens -- recompute it before trusting the file.")
        yield StimRow(
            dataset="huang2024",
            item_id=r["item"],
            condition=r["condition"],
            sentence_orig=r["Sentence"],
            sentence_norm=normalize(r["Sentence"]),
            model_input=model_input,
            contrast="disambiguator",
            intended_uid=uid("huang2024", r["item"], f"{construction}_UAMB"),
            critical_word_idx=idx,
            meta={
                "construction": construction,
                "ambiguity": r["ambiguity"],
                "disamb_position_amb": r["disambPositionAmb"],
                "disamb_position_unamb": r["disambPositionUnamb"],
                "disamb_word": expected,
            },
        )
