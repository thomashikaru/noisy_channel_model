"""qian2023 -- Qian & Levy: agreement attraction under a noisy channel.

``data/qian2023/materials.csv``: 60 items x 8 conditions = 480 rows, all of which are kept.

The ``condition`` code is three s/p letters giving the number of N1 (the head), N2 (the
attractor inside the PP) and the predicate::

    sss  "the location of the star  has  caught attention..."   grammatical
    ssp  "the location of the star  have caught attention..."   ungrammatical
    sps  "the location of the stars has  caught attention..."   grammatical
    spp  "the location of the stars have caught attention..."   ungrammatical, and attracted

The predicate must agree with the head, so ``is_grammatical = cond[0] == cond[2]``; this is the
one dataset where grammaticality is the manipulation rather than plausibility.

**Two co-equal repairs, neither privileged.**  An ungrammatical row is genuinely ambiguous about
which word carries the error, and a noisy-channel reader can resolve it either way::

    observed   The gifts for the kid is hidden under the bed.       (cond "pss")
    verb       The gifts for the kid are hidden under the bed.      (cond "psp")  is -> are
    noun       The gift  for the kid is  hidden under the bed.      (cond "sss")  gifts -> gift

So ``intended_uids`` carries BOTH: condition ``c0 c1 c0`` (predicate number := N1's) and
``c2 c1 c2`` (N1 number := the predicate's).  For the grammatical half the two coincide and the
row is its own single counterpart.  Nothing in the schema ranks them; an analysis that wants one
must say so and say why.  They are very unevenly reachable by the substitution channel (see
``experiments/reachability.py``), which is a fact about the model, not a reason to pick one.

The source's own ``context`` column is a fill-in template for generating the item, not preceding
text, so it goes to ``meta`` and NOT to the harness's ``context`` (LM prime) field.  The source
is lowercase with punctuation split off.
"""

from __future__ import annotations

from .common import StimRow, normalize, read_source_csv, standardize, uid

_META_COLS = ("N1", "preposition", "N2", "predicate", "predicate_sg_pl", "context",
              "N1_number", "N2_number", "predicate_sg_pl_number")


def convert():
    for r in read_source_csv("data/qian2023/materials.csv"):
        cond = r["condition"]
        assert len(cond) == 3 and set(cond) <= {"s", "p"}, f"unexpected qian condition {cond!r}"
        yield StimRow(
            dataset="qian2023",
            item_id=r["item"],
            condition=cond,
            sentence_orig="",                       # lost upstream: the source is already normalized
            sentence_norm=normalize(r["sentence"]),
            model_input=standardize(r["sentence"]),
            is_grammatical=(cond[0] == cond[2]),    # the predicate agrees with the head, not N2
            contrast="agreement",
            intended_uids=sorted({uid("qian2023", r["item"], cond[0] + cond[1] + cond[0]),
                                  uid("qian2023", r["item"], cond[2] + cond[1] + cond[2])}),
            meta={k: r[k] for k in _META_COLS},
        )
