"""Shared machinery for the experiment-harness stimulus converters.

Each converter reads one study's published materials out of ``data/<study>/`` and yields
:class:`StimRow` records in the common schema documented in ``experiments/README.md``.
Three rules hold for every converter:

**Read-only.**  ``data/`` is gitignored, so the files under it are the only copy in this
checkout.  Converters must never write, move, or truncate them.  :func:`open_source` enforces
this structurally: it reads the file's bytes itself and hands back an in-memory ``StringIO``,
so a converter is never holding a handle to the file on disk in the first place.

**Blind to human data.**  :func:`open_source` refuses any path under :data:`HOLDOUT_PATHS`.
The order of operations for this project is to run the model on the stimuli first and compare
to human behavior only afterwards, so the harness must not be able to see the human data even
by accident.

**Auditable.**  Every file that is opened gets its sha256 recorded in :data:`SOURCES_SEEN`,
which ``build_stimuli.py`` dumps into ``stimuli/MANIFEST.json``.  Since ``data/`` is not
tracked, those hashes plus the converters are what makes the build reproducible.
"""

from __future__ import annotations

import csv
import dataclasses
import difflib
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Files holding human behavioral data.  Never opened by anything under ``experiments/``.
#: A prefix match is enough: a directory entry ends in "/" and covers everything under it.
HOLDOUT_PATHS = (
    "data/clark2026/exp_data_merged.csv",
    "data/clark2026/lists/",
)

#: repo-relative path -> sha256 of the bytes read.  Filled by :func:`open_source`.
SOURCES_SEEN: dict[str, str] = {}


class HoldoutError(PermissionError):
    """Raised when a converter tries to open a reserved human-data file."""


def _rel(path) -> str:
    """Normalize *path* to a repo-relative POSIX string (accepts absolute or relative)."""
    p = Path(path)
    if p.is_absolute():
        p = p.relative_to(REPO_ROOT)
    return p.as_posix()


def open_source(path, encoding: str = "utf-8", errors: str = "strict") -> io.StringIO:
    """Open a source file for reading and record its hash.

    Returns a ``StringIO`` over the decoded bytes rather than a file object, so the caller
    cannot write to the original.  ``newline=""`` is set on the buffer, which is what
    :mod:`csv` wants and what keeps CRLF line endings visible to callers that care.
    """
    rel = _rel(path)
    for holdout in HOLDOUT_PATHS:
        if rel == holdout or rel.startswith(holdout):
            raise HoldoutError(
                f"{rel} is reserved human data (matches HOLDOUT_PATHS entry {holdout!r}). "
                "The harness runs the model blind; nothing here may read it."
            )
    raw = (REPO_ROOT / rel).read_bytes()
    SOURCES_SEEN[rel] = hashlib.sha256(raw).hexdigest()
    return io.StringIO(raw.decode(encoding, errors), newline="")


def read_source_csv(path, delimiter: str = ",", **kw) -> list[dict]:
    """Read a delimited source file into a list of dicts."""
    return list(csv.DictReader(open_source(path, **kw), delimiter=delimiter))


def read_source_text(path, **kw) -> str:
    """Read a source file's full decoded text."""
    return open_source(path, **kw).read()


# --------------------------------------------------------------------------------------
# Text conventions
# --------------------------------------------------------------------------------------

_PUNCT = r"([.,?!;:\"])"


def normalize(s: str) -> str:
    """The old pipeline's ``sentences.txt`` convention: lowercase, punctuation split off.

    Kept so the harmonized stimuli can be joined to legacy per-study files.  Verified to
    reproduce the shipped ``data/<study>/sentences.txt`` line for line for gibson2013 (all
    three subsets), huang2024, clark2026, moses and ryskin2021; qian2023's ``sentences.txt``
    holds only 120 of the 480 rows, and those 120 are a subset of what this produces.
    """
    return " ".join(re.sub(_PUNCT, r" \1", s.lower()).split())


def standardize(s: str) -> str:
    """The model input convention: initial capital, punctuation attached, terminal ``.``/``?``.

    This is what ``calibration_word_action_smc._wellform`` does to the calibration battery, and
    applying it uniformly is what keeps the model's leading-opener artifact from firing on the
    lowercase-only sources (ryskin2021, qian2023).
    """
    s = " ".join(s.split())
    s = re.sub(r"\s+([.,?!;:])", r"\1", s)   # re-attach punctuation the source split off
    s = re.sub(r"\s+'", "'", s)              # re-attach clitic apostrophes (guard; no-op on
                                             # every current source, all of which attach them)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".?!":
        s += "."
    return s


def strip_punct(token: str) -> str:
    """Drop leading/trailing punctuation from a whitespace token."""
    return token.strip(".,?!;:\"'")


def critical_index(model_input: str, word: str) -> int | None:
    """Index of *word* among the whitespace tokens of *model_input*, punctuation stripped."""
    toks = [strip_punct(t).lower() for t in model_input.split()]
    target = strip_punct(word).lower()
    return toks.index(target) if target in toks else None


# --------------------------------------------------------------------------------------
# Edit classification
# --------------------------------------------------------------------------------------
#
# The single-op branch mirrors ``calibration_gate.word_change`` exactly (same difflib call,
# same opcode mapping).  It is restated here rather than imported because that module imports
# jax and the penzai LM at module scope, and the stimulus build is a stdlib-only data step.
# ``test_converters.py`` asserts the two agree on every pair the build produces.


class Edit(NamedTuple):
    """The word-level difference between an observed sentence and its intended counterpart."""

    type: str        # sub | ins | del | none | multi
    ops: str         # every opcode kind in order, e.g. "ins;ins"
    frm: str         # observed words the first opcode consumes ("" for a pure insertion)
    to: str          # intended words the first opcode produces ("" for a pure deletion)
    obs_idx: int | None   # index of the first changed OBSERVED token; None for a pure insertion
    obs_gap: int          # observed-token position the first opcode starts at, insertions included


def classify_edit(observed: str, intended: str) -> Edit:
    """Classify the word-level edit that turns *observed* into *intended*.

    ``type`` is ``sub`` / ``ins`` (the intended has a word the observed lacks, i.e. a missing
    word gets restored) / ``del`` (the observed has a word the intended lacks, i.e. a spurious
    word gets removed) / ``none`` / ``multi`` when more than one difflib opcode is needed.
    ``ops`` spells the opcode kinds out in order, so the homogeneous multi-edit cases --
    chen2023's active/passive pairs need two insertions or two deletions -- stay
    distinguishable from genuinely mixed ones.  A single opcode spanning several adjacent
    words (tabor2004 restores ``"who was"`` in one go) is NOT ``multi``.
    """
    ow, iw = observed.split(), intended.split()
    ops = [op for op in difflib.SequenceMatcher(a=ow, b=iw, autojunk=False).get_opcodes()
           if op[0] != "equal"]
    if not ops:
        return Edit("none", "", "", "", None, 0)
    kinds = {"replace": "sub", "insert": "ins", "delete": "del"}
    ops_str = ";".join(kinds[op[0]] for op in ops)
    tag, i1, i2, j1, j2 = ops[0]
    frm, to = " ".join(ow[i1:i2]), " ".join(iw[j1:j2])
    obs_idx = None if tag == "insert" else i1
    return Edit("multi" if len(ops) > 1 else kinds[tag], ops_str, frm, to, obs_idx, i1)


# --------------------------------------------------------------------------------------
# The common schema
# --------------------------------------------------------------------------------------

#: Vocabulary for ``StimRow.contrast``: the DESIGN-level relation between a row and its intended
#: counterpart, as opposed to ``edit_type``, which is whatever difflib needs at the word level.
#: The two can disagree, and the disagreement is informative rather than a defect.  chen2023's
#: voice alternation is the clearest case: "The ball kicked the girl." -> "The ball was kicked by
#: the girl." is two insertions, but "The truck drove the man." -> "The truck was driven by the
#: man." is one replacement, because the irregular participle breaks the word match.  Both are
#: ``contrast == "voice"``.  Group by ``contrast`` for analysis; read ``edit_type`` / ``edit_ops``
#: for what the channel actually has to do.
CONTRASTS = (
    "dative",       # DO <-> PO: one preposition inserted or deleted (gibson2013, chen2023)
    "transitivity", # transitive <-> prepositional: one preposition (gibson2013)
    "voice",        # active <-> passive (chen2023)
    "word_form",    # a form/semantic neighbour of the intended word (ryskin2021, clark2026)
    "typo",         # a keystroke error (clark2026)
    "agreement",    # subject-verb number (qian2023)
    "disambiguator",# a complementizer, comma or relativizer that blocks a garden path (huang2024)
    "relativizer",  # the "who was" a reduced relative drops (tabor2004)
)


def uid(dataset: str, item_id: str, condition: str, subset: str = "") -> str:
    """Build a ``stim_uid``.

    The one place the format is defined, so a converter's ``intended_uid`` can never drift from
    the ``stim_uid`` it is meant to point at.  Datasets without subsets leave that segment empty
    (``"ryskin2021//56/Control"``).
    """
    return f"{dataset}/{subset}/{item_id}/{condition}"


@dataclass
class StimRow:
    """One (item, condition) stimulus in the common schema.

    Converters fill everything except ``sentence_id``, which ``build_stimuli.py`` assigns, and
    the resolved repair detail, which it derives into ``<dataset>.repairs.csv`` once every row
    exists.
    """

    dataset: str
    item_id: str
    condition: str
    subset: str = ""
    sentence_orig: str = ""          # original orthography, when the source preserves it
    sentence_norm: str = ""          # normalize(): legacy lowercase/split-punctuation form
    model_input: str = ""            # standardize(): WHAT THE MODEL READS
    context: str = ""                # clean preceding text for the LM prime; "" when none
    plausibility: str = ""           # "plausible" / "implausible" / "" when not applicable
    is_grammatical: bool | None = None
    contrast: str = ""               # what the design varies between this row and its counterpart
    intended_uids: list[str] = field(default_factory=list)
    # ^ every reading a noisy-channel reader could recover, with NO ordering or primacy among
    #   them.  Usually one; qian2023 has two (fix the verb or fix the noun) and neither is "the"
    #   intended sentence.  The per-repair detail -- text, edit class, reachability -- lives in
    #   <dataset>.repairs.csv, one row per (stimulus, repair); see build_stimuli.resolve_repairs.
    critical_word_idx: int | None = None   # 0-based, into model_input's whitespace tokens
    comprehension_q: str = ""        # normative answer key, NOT human data
    correct_answer: str = ""
    meta: dict = field(default_factory=dict)
    sentence_id: int = -1            # index into <dataset>.input.jsonl

    @property
    def stim_uid(self) -> str:
        return uid(self.dataset, self.item_id, self.condition, self.subset)

    def as_csv_row(self) -> dict:
        d = dataclasses.asdict(self)
        d["meta"] = json.dumps(self.meta, sort_keys=True, ensure_ascii=False)
        d["intended_uids"] = ";".join(self.intended_uids)   # uids contain no ";"
        d["n_intended"] = len(self.intended_uids)
        d["is_grammatical"] = "" if self.is_grammatical is None else int(self.is_grammatical)
        d["critical_word_idx"] = "" if self.critical_word_idx is None else self.critical_word_idx
        d["stim_uid"] = self.stim_uid
        return d


#: Column order of ``<dataset>.stimuli.csv``.
CSV_FIELDS = [
    "dataset", "subset", "item_id", "condition", "stim_uid",
    "sentence_orig", "sentence_norm", "model_input", "context", "sentence_id",
    "plausibility", "is_grammatical", "contrast",
    "intended_uids", "n_intended",
    "critical_word_idx", "comprehension_q", "correct_answer", "meta",
]

#: Column order of ``<dataset>.repairs.csv`` -- one row per (stimulus, admissible repair).
REPAIR_FIELDS = [
    "dataset", "stim_uid", "intended_uid", "intended_text",
    "edit_type", "edit_ops", "edit_from", "edit_to", "edit_obs_idx", "n_repairs_for_stim",
]
