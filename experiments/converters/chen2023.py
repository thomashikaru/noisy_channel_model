"""chen2023 -- Chen, Huang, Ryskin, Gibson et al.: context effects on noisy-channel inference.

Two subsets (``dopo_to``, ``active_passive``), each shipped as three Linger presentation files
(``no-context``, ``supportive``, ``non-supportive``) under ``data/chen2023/<subset>/``.  Each
file holds 20 target items x 4 conditions plus 48 fillers; the fillers are excluded, giving
2 x 3 x 80 = 480 rows.

Linger format::

    # <group> <item> <condition>
    <context sentences and target, on one line>
    ? <comprehension question> <answer>

**Context handling.**  The context and the target share one line, nominally separated by two
spaces -- but the separator is inconsistent, and splitting on ``"  "`` recovers the wrong target
on 27 of the 320 context rows: on 20 (all dopo_to/supportive) the separator before the target is
three spaces, so the target keeps a leading space; on 7 (active_passive/supportive) two context
sentences are separated by a single space, so a whole context sentence lands inside the
"target".  Instead the no-context file is treated as the authority for the target text and the
context is whatever prefix precedes it; that matches on all 320 rows and is asserted per row.
The context becomes the LM prime, so the channel only ever sees the target sentence.

**Encoding.**  All six files are CRLF.  ``dopo-to-supportive.txt`` is doubly encoded: its
right single quotes appear as the mojibake ``‚Äô``, which is the UTF-8 bytes of ``’`` read as
cp1252 and re-encoded.  That exact sequence is repaired, then all right single quotes are
folded to ASCII apostrophes.  The repair is checked, not assumed: the parser asserts nothing
non-ASCII survives.

**Identity.**  ``condition`` carries the context level (``DO_implausible.supportive``) because
the same item x Linger condition appears in all three files and ``stim_uid`` must stay unique.
``meta`` keeps ``linger_cond`` and ``context_type`` as separate fields for filtering.

The no-context ``dopo_to`` targets are byte-identical to gibson2013's ``dopo_to`` materials.
They are kept -- the two studies are separate datasets with separate input lists -- so those 80
sentences get run twice, once under each dataset's ``sentence_id``.
"""

from __future__ import annotations

from .common import StimRow, normalize, read_source_text, standardize, uid

#: subset -> (directory, filename stem)
SUBSETS = {"dopo_to": "dopo-to", "active_passive": "active-passive"}

#: our context label -> the filename suffix that carries it
CONTEXTS = {"none": "no-context", "supportive": "supportive", "non_supportive": "non-supportive"}

#: The double-encoding signature in dopo-to-supportive.txt: UTF-8 e2 80 99 read as cp1252.
_MOJIBAKE = {"‚Äô": "’"}

#: The design axis each subset varies (see common.CONTRASTS).
_CONTRAST = {"dopo_to": "dative", "active_passive": "voice"}


def convert():
    for subset in SUBSETS:
        yield from _convert_subset(subset)


def _read_linger(path: str) -> list[dict]:
    """Parse one Linger file into ``{group, item, cond, text, question, answer}`` records."""
    raw = read_source_text(path)
    for bad, good in _MOJIBAKE.items():
        raw = raw.replace(bad, good)
    raw = raw.replace("’", "'").replace("\r", "")
    non_ascii = sorted({c for c in raw if ord(c) > 127})
    assert not non_ascii, f"{path}: unhandled non-ASCII {non_ascii} -- extend _MOJIBAKE"

    recs: list[dict] = []
    cur: dict | None = None
    for line in raw.split("\n"):
        if line.startswith("# "):
            group, item, cond = line[2:].split()[:3]
            cur = {"group": group, "item": item, "cond": cond}
        elif line.startswith("? "):
            assert cur is not None and "text" in cur, f"{path}: '? ' line before a text line"
            question, answer = line[2:].rsplit(" ", 1)
            cur.update(question=question, answer=answer)
            recs.append(cur)
            cur = None
        elif line.strip():
            assert cur is not None, f"{path}: text line before a '# ' header"
            cur["text"] = " ".join(line.split())
    return recs


def _convert_subset(subset: str):
    stem = SUBSETS[subset]
    by_ctx = {ctx: _read_linger(f"data/chen2023/{subset}/{stem}-{suffix}.txt")
              for ctx, suffix in CONTEXTS.items()}

    # The no-context file defines the target text for every (item, condition).
    targets = {(r["item"], r["cond"]): r["text"]
               for r in by_ctx["none"] if r["cond"] != "filler"}

    for ctx, recs in by_ctx.items():
        for r in recs:
            if r["cond"] == "filler":
                continue
            target = targets[(r["item"], r["cond"])]
            full = r["text"]
            assert full.endswith(target), (
                f"{subset}/{ctx} item {r['item']} {r['cond']}: the no-context target is not a "
                f"suffix of the presented line\n  line   = {full!r}\n  target = {target!r}")
            context = full[: len(full) - len(target)].strip()
            assert (ctx == "none") == (context == ""), \
                f"{subset}/{ctx} item {r['item']}: context presence disagrees with the file"

            voice, plaus = r["cond"].rsplit("_", 1)
            condition = f"{r['cond']}.{ctx}"
            row_uid = lambda cond: uid("chen2023", r["item"], f"{cond}.{ctx}", subset)  # noqa: E731
            if plaus == "plausible":
                intended = row_uid(r["cond"])
            else:
                intended = row_uid(f"{_counterpart(subset, voice)}_plausible")
            yield StimRow(
                dataset="chen2023",
                subset=subset,
                item_id=r["item"],
                condition=condition,
                sentence_orig=target,
                sentence_norm=normalize(target),
                model_input=standardize(target),
                context=standardize(context) if context else "",
                plausibility=plaus,
                contrast=_CONTRAST[subset],
                intended_uid=intended,
                comprehension_q=r["question"],
                correct_answer=r["answer"],
                meta={
                    "group": r["group"],
                    "linger_cond": r["cond"],
                    "context_type": ctx,
                    "structure": voice,
                    "same_structure_plausible_uid": row_uid(f"{voice}_plausible"),
                },
            )


def _counterpart(subset: str, structure: str) -> str:
    """The structure whose plausible row is one minimal edit from this implausible one.

    ``dopo_to`` works exactly like gibson2013: DO_implausible needs "to" inserted, PO_implausible
    needs it deleted.  ``active_passive`` crosses voice, which takes two edits in each direction
    ("The ball kicked the girl." -> "The ball was kicked by the girl." inserts both "was" and
    "by"), so those pairs come out as ``edit_type == "multi"`` with ``edit_ops`` recording the
    homogeneous direction (``ins;ins`` or ``del;del``).
    """
    return {"DO": "PO", "PO": "DO", "active": "passive", "passive": "active"}[structure]
