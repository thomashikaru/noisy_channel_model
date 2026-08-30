#!/usr/bin/env python3
"""Build the harmonized stimuli from the raw study materials.

    python experiments/build_stimuli.py [dataset ...] [--rebuild] [--check]

Runs each converter, assigns stable sentence ids, resolves every row's intended counterpart,
checks the invariants, and writes into ``experiments/stimuli/``:

    <dataset>.stimuli.csv    one row per (item, condition), the common schema
    <dataset>.input.jsonl    the model's input list: unique (context, text) pairs, APPEND-ONLY
    smoke.{stimuli.csv,input.jsonl}    the pipeline smoke set: one item per phenomenon
    probe.{stimuli.csv,input.jsonl}    the cost-probe set: the worst case for runtime and memory
    MANIFEST.json            source files + sha256, counts, converter commit, build time

Stdlib only, and it reads ``data/`` without writing to it (see ``converters/common.py``).

**The input lists are append-only.**  The SLURM worker's per-item resume is keyed by line index
into the input file, so inserting or reordering a line would silently re-map finished results
onto different sentences.  A rebuild that would change an existing line fails; ``--rebuild``
overrides that and means the already-computed results for that dataset must be discarded.

``--check`` builds everything in memory and reports what would change without writing.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from converters import CONVERTERS                                    # noqa: E402
from converters.common import (CONTRASTS, CSV_FIELDS, HOLDOUT_PATHS, REPAIR_FIELDS,  # noqa: E402
                               REPO_ROOT, SOURCES_SEEN, StimRow, classify_edit)

STIMULI_DIR = Path(__file__).resolve().parent / "stimuli"

#: Row counts each converter must produce.  A mismatch means a source file changed under us.
EXPECTED_ROWS = {
    "gibson2013": 240,   # 3 subsets x 20 items x 2 structures x 2 plausibility
    "chen2023": 480,     # 2 subsets x 3 contexts x 20 items x 4 conditions (fillers excluded)
    "ryskin2021": 504,   # 126 items x 4 conditions
    "qian2023": 480,     # 60 items x 8 conditions
    "huang2024": 144,    # 24 items x 3 constructions x 2 ambiguity levels
    "clark2026": 360,    # 36 items x 10 labels
    "tabor2004": 128,    # 32 items x 2 reduced_rel x 2 coherence
    "moses": 1,
}

#: Contrasts whose counterpart is expected to take more than one word edit.  chen2023's voice
#: alternation inserts or deletes both an auxiliary and a "by", so ``multi`` there is the design,
#: not a defect.
MULTI_BY_DESIGN = {"voice"}

#: Anomalies each dataset is known to contain, as of the sources hashed in MANIFEST.json.  These
#: are defects in the published materials, not in the converters -- see experiments/README.md.
#: The build asserts the count so that re-fetching a source can never quietly introduce more.
EXPECTED_ANOMALIES = {
    "gibson2013": 2,   # two items whose plausible counterpart has its own typo
    "huang2024": 1,    # one item whose unambiguous version gained an extra adjective
    "qian2023": 8,     # two items where the N2 number manipulation is missing from the text
}

#: The smoke set: one stimulus per phenomenon in the harness plan, plus a chen2023 item that
#: exercises the context prime.  Selected by stim_uid so the build fails loudly if a converter
#: ever renames a condition, rather than silently smoke-testing something else.
SMOKE_UIDS = [
    "gibson2013/dopo_to/2/DO_implausible",          # missing word: "gave the candle the daughter"
    "gibson2013/dopo_to/2/PO_implausible",          # extra word:   "gave the daughter to the candle"
    "ryskin2021//56/SemCrit",                       # form substitution: inflection -> infection
    "qian2023//35/pss",                             # agreement: "the gifts for the kid is hidden"
    "huang2024//1/NPZ_AMB",                         # classic garden path (NP/Z)
    "clark2026//0/2.1",                             # noisy-channel garden path: licked -> kicked
    "tabor2004//e1_16/reduced_local",               # local coherence: "the player tossed a frisbee"
    "chen2023/dopo_to/2/DO_implausible.supportive",  # the same as row 1, but with a context prime
]


def build_dataset(name: str, convert) -> tuple[list[StimRow], dict[str, str], list[dict]]:
    """Run one converter and fill in sentence ids and the resolved intended fields."""
    seen_before = set(SOURCES_SEEN)
    rows = list(convert())
    sources = {k: v for k, v in SOURCES_SEEN.items() if k not in seen_before}

    if name in EXPECTED_ROWS:
        assert len(rows) == EXPECTED_ROWS[name], \
            f"{name}: expected {EXPECTED_ROWS[name]} rows, converter produced {len(rows)}"

    # Unique (context, model_input) pairs, in first-seen order.  Two rows that present the same
    # text under the same context are one model run.
    inputs: dict[tuple[str, str], int] = {}
    for r in rows:
        r.sentence_id = inputs.setdefault((r.context, r.model_input), len(inputs))

    by_uid = {r.stim_uid: r for r in rows}
    assert len(by_uid) == len(rows), \
        f"{name}: duplicate stim_uid ({len(rows) - len(by_uid)} collisions) -- " \
        f"condition does not uniquely identify a row"

    repairs = resolve_repairs(name, rows, by_uid)
    check_rows(name, rows)
    return rows, sources, repairs


def resolve_repairs(name: str, rows: list[StimRow], by_uid: dict[str, StimRow]) -> list[dict]:
    """One record per (stimulus, admissible repair).

    A stimulus may admit more than one repair, and when it does they are co-equal: qian2023's
    ungrammatical rows can be fixed at the verb or at the noun, and nothing in the design says
    which one the reader recovers.  Keeping them in their own table rather than as a primary
    plus alternatives is what stops an analysis from silently privileging one.
    """
    out: list[dict] = []
    for r in rows:
        for intended_uid in r.intended_uids:
            assert intended_uid in by_uid, \
                f"{name}: {r.stim_uid} names intended_uid {intended_uid!r}, which is not a row"
            intended_text = by_uid[intended_uid].model_input
            edit = classify_edit(r.model_input, intended_text)
            out.append({
                "dataset": name, "stim_uid": r.stim_uid, "intended_uid": intended_uid,
                "intended_text": intended_text, "edit_type": edit.type, "edit_ops": edit.ops,
                "edit_from": edit.frm, "edit_to": edit.to,
                "edit_obs_idx": "" if edit.obs_idx is None else edit.obs_idx,
                "n_repairs_for_stim": len(r.intended_uids),
            })
        # A dataset that names its own critical word has already set the index.  Otherwise take
        # it from the repair, but only when every repair agrees -- qian2023's two routes touch
        # different words, so there is no single critical index and it stays empty.
        if r.critical_word_idx is None:
            idxs = {e["edit_obs_idx"] for e in out[-len(r.intended_uids):]} if r.intended_uids else set()
            if len(idxs) == 1 and (only := idxs.pop()) != "":
                r.critical_word_idx = only
    return out


def find_anomalies(rows: list[StimRow], repairs: list[dict]) -> list[dict]:
    """Defects in the published materials that survive into the harmonized stimuli.

    Two kinds show up across these eight studies:

    * a row whose intended counterpart is more than one edit away in a design that varies one
      word -- which always turns out to be a typo or a stray extra word in the *counterpart*,
      not in the row itself;
    * two conditions of one item that produce identical text, meaning the manipulation those
      conditions differ on never made it into the sentence.

    Neither is fixable here (editing the stimuli would misrepresent what the studies ran), so
    they are recorded, counted and documented instead.  Rows in the first group still get a
    correct ``edit_type``/``edit_ops``; rows in the second share one ``sentence_id`` and so one
    model run, which is right -- the model sees one sentence.
    """
    out: list[dict] = []
    contrast = {r.stim_uid: r.contrast for r in rows}
    text = {r.stim_uid: r.model_input for r in rows}
    for e in repairs:
        if e["edit_type"] == "multi" and contrast[e["stim_uid"]] not in MULTI_BY_DESIGN:
            out.append({"kind": "counterpart_needs_multiple_edits", "stim_uid": e["stim_uid"],
                        "edit_ops": e["edit_ops"], "observed": text[e["stim_uid"]],
                        "intended": e["intended_text"], "intended_uid": e["intended_uid"]})
    by_text: dict[tuple, list[str]] = {}
    for r in rows:
        by_text.setdefault((r.item_id, r.subset, r.context, r.model_input), []).append(r.condition)
    for (item_id, subset, _ctx, text), conds in by_text.items():
        if len(conds) > 1:
            out.append({"kind": "conditions_share_one_sentence", "item_id": item_id,
                        "subset": subset, "conditions": sorted(conds), "text": text})
    return out


def check_rows(name: str, rows: list[StimRow]) -> None:
    """Invariants that must hold for every dataset."""
    for r in rows:
        assert r.contrast in ("",) + CONTRASTS, \
            f"{name}: {r.stim_uid} has contrast {r.contrast!r}, which is not in common.CONTRASTS"
        assert bool(r.contrast) or not r.intended_uids, \
            f"{name}: {r.stim_uid} has a counterpart but no contrast naming the design axis"
        mi = r.model_input
        assert mi and mi == mi.strip(), f"{name}: {r.stim_uid} has empty or padded model_input"
        assert mi[0].isupper() or not mi[0].isalpha(), \
            f"{name}: {r.stim_uid} model_input is not capitalized: {mi!r}"
        assert mi[-1] in ".?!", f"{name}: {r.stim_uid} model_input has no terminal mark: {mi!r}"
        assert "  " not in mi, f"{name}: {r.stim_uid} model_input has a double space: {mi!r}"
        if r.context:
            assert r.context == r.context.strip(), f"{name}: {r.stim_uid} has a padded context"
        for u in r.intended_uids:
            assert u, f"{name}: {r.stim_uid} has an empty string in intended_uids"
        assert len(set(r.intended_uids)) == len(r.intended_uids), \
            f"{name}: {r.stim_uid} lists a duplicate intended_uid"
        if r.critical_word_idx is not None:
            n = len(mi.split())
            assert 0 <= r.critical_word_idx < n, \
                f"{name}: {r.stim_uid} critical_word_idx {r.critical_word_idx} outside 0..{n - 1}"
    ids = sorted({r.sentence_id for r in rows})
    assert ids == list(range(len(ids))), f"{name}: sentence_ids are not dense: {ids[:5]}..."


def input_records(rows: list[StimRow]) -> list[dict]:
    """The model input list: one record per unique (context, text), ordered by sentence_id."""
    seen: dict[int, dict] = {}
    for r in rows:
        seen.setdefault(r.sentence_id,
                        {"sentence_id": r.sentence_id, "text": r.model_input, "context": r.context})
    return [seen[i] for i in range(len(seen))]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_append_only(path: Path, new: list[dict], rebuild: bool) -> str:
    """Refuse to change a line the SLURM worker may already have results for."""
    existing = read_jsonl(path)
    if not existing:
        return "new" if new else "empty"
    prefix = new[:len(existing)]
    if existing == prefix:
        return "unchanged" if len(new) == len(existing) else f"appended +{len(new) - len(existing)}"
    if not rebuild:
        first = next((i for i, (a, b) in enumerate(zip(existing, prefix)) if a != b), len(prefix))
        raise SystemExit(
            f"{path.name}: line {first} changed, and the input list is append-only -- the SLURM\n"
            f"worker resumes per item by line index, so any results already computed for this\n"
            f"dataset would be mis-attributed.\n"
            f"  was: {existing[first] if first < len(existing) else '<missing>'}\n"
            f"  now: {prefix[first] if first < len(prefix) else '<missing>'}\n"
            f"Re-run with --rebuild only if you are also discarding this dataset's results.")
    return "REBUILT (previous results for this dataset are invalid)"


def write_dataset(name: str, rows: list[StimRow], rebuild: bool, dry: bool,
                  repairs: list[dict] | None = None) -> str:
    records = input_records(rows)
    status = check_append_only(STIMULI_DIR / f"{name}.input.jsonl", records, rebuild)
    if dry:
        return status
    STIMULI_DIR.mkdir(parents=True, exist_ok=True)
    with (STIMULI_DIR / f"{name}.stimuli.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(r.as_csv_row() for r in rows)
    with (STIMULI_DIR / f"{name}.input.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if repairs is not None:
        with (STIMULI_DIR / f"{name}.repairs.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=REPAIR_FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(repairs)
    return status


def derived_set(name: str, rows: list[StimRow]) -> list[StimRow]:
    """Re-key a hand-picked selection of rows as its own little dataset (smoke / probe).

    ``intended_uids`` deliberately does not come along -- the uids point into another dataset's
    table -- so a derived set has no repairs table.  ``meta.source_stim_uid`` is the link back to
    the row these were picked from, and to its repairs.
    """
    out = []
    for i, src in enumerate(rows):
        r = StimRow(dataset=name, item_id=str(i), condition=src.condition,
                    sentence_orig=src.sentence_orig, sentence_norm=src.sentence_norm,
                    model_input=src.model_input, context=src.context,
                    plausibility=src.plausibility, is_grammatical=src.is_grammatical,
                    contrast=src.contrast, critical_word_idx=src.critical_word_idx,
                    comprehension_q=src.comprehension_q, correct_answer=src.correct_answer,
                    meta={"source_stim_uid": src.stim_uid, "source_dataset": src.dataset})
        r.sentence_id = i
        out.append(r)
    return out


def pick_smoke(all_rows: dict[str, list[StimRow]]) -> list[StimRow]:
    by_uid = {r.stim_uid: r for rows in all_rows.values() for r in rows}
    missing = [u for u in SMOKE_UIDS if u not in by_uid]
    assert not missing, f"smoke set references stim_uids that no converter produced: {missing}"
    return derived_set("smoke", [by_uid[u] for u in SMOKE_UIDS])


def pick_probe(all_rows: dict[str, list[StimRow]]) -> list[StimRow]:
    """The worst case for runtime and peak memory: cost is driven by units and prime length.

    One row per dataset with the most whitespace tokens, plus the chen2023 row with the longest
    context, since a long prime is a separate cost axis (it sets LCTX and thus the compile shape).
    """
    picked = []
    for name, rows in all_rows.items():
        picked.append(max(rows, key=lambda r: (len(r.model_input.split()), r.stim_uid)))
    chen = all_rows.get("chen2023", [])
    if chen:
        longest_ctx = max(chen, key=lambda r: (len(r.context.split()), r.stim_uid))
        if longest_ctx.stim_uid not in {r.stim_uid for r in picked}:
            picked.append(longest_ctx)
    return derived_set("probe", picked)


#: The files whose content determines what the stimuli look like.
CONVERTER_SOURCES = sorted(
    [Path(__file__).resolve()] + list((Path(__file__).resolve().parent / "converters").glob("*.py")))


def converter_digest() -> str:
    """A content hash of the converter sources.

    This, not a commit sha, is the manifest's authoritative provenance: a manifest can never
    record the commit that contains it (committing the manifest changes that commit), and
    amending would leave it pointing at an orphan.  A content hash has no such circularity, and
    ``test_manifest_records_the_current_converters`` checks the stimuli on disk were built by the
    code that is checked in.
    """
    h = hashlib.sha256()
    for path in CONVERTER_SOURCES:
        h.update(path.relative_to(REPO_ROOT).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def git_state() -> tuple[str, bool]:
    """HEAD at build time, and whether TRACKED files differ from it.

    Recorded for human orientation only -- the commit that carries the manifest is a descendant
    of this one.  Untracked files are ignored: they cannot change what the converters do, and a
    personal scratch file should not mark a build as unreproducible.
    """
    def run(*args):
        return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return run("git", "rev-parse", "HEAD"), bool(run("git", "status", "--porcelain", "-uno"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="build only these (default: all)")
    ap.add_argument("--rebuild", action="store_true",
                    help="allow an input list to change; invalidates that dataset's results")
    ap.add_argument("--check", action="store_true", help="report what would change; write nothing")
    args = ap.parse_args()

    wanted = args.datasets or list(CONVERTERS)
    unknown = [d for d in wanted if d not in CONVERTERS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; known: {list(CONVERTERS)}")

    all_rows: dict[str, list[StimRow]] = {}
    sources: dict[str, dict[str, str]] = {}
    manifest_datasets: dict[str, dict] = {}

    for name in wanted:
        rows, srcs, repairs = build_dataset(name, CONVERTERS[name])
        all_rows[name], sources[name] = rows, srcs
        status = write_dataset(name, rows, args.rebuild, args.check, repairs)
        n_inputs = len({r.sentence_id for r in rows})
        edits = collections.Counter(e["edit_type"] for e in repairs)
        anomalies = find_anomalies(rows, repairs)
        expected = EXPECTED_ANOMALIES.get(name, 0)
        assert len(anomalies) == expected, (
            f"{name}: found {len(anomalies)} source anomalies, expected {expected}. If a source "
            f"file was re-fetched, review them and update EXPECTED_ANOMALIES:\n" +
            "\n".join(f"  {a}" for a in anomalies))
        manifest_datasets[name] = {
            "sources": srcs,
            "n_rows": len(rows),
            "n_inputs": n_inputs,
            "n_with_context": sum(1 for r in rows if r.context),
            "rows_per_condition": dict(sorted(collections.Counter(
                f"{r.subset}/{r.condition}" if r.subset else r.condition for r in rows).items())),
            "n_repairs": len(repairs),
            "n_stimuli_with_multiple_repairs": sum(1 for r in rows if len(r.intended_uids) > 1),
            "edit_types": dict(sorted(edits.items())),
            "contrasts": dict(sorted(collections.Counter(
                r.contrast for r in rows if r.contrast).items())),
            "anomalies": anomalies,
        }
        print(f"{name:12s} rows={len(rows):4d}  inputs={n_inputs:4d}  "
              f"ctx={manifest_datasets[name]['n_with_context']:4d}  "
              f"repairs={len(repairs):4d} edits={dict(sorted(edits.items()))}"
              f"{f'  anomalies={len(anomalies)}' if anomalies else ''}  [{status}]")

    if set(wanted) == set(CONVERTERS):
        for name, picker in (("smoke", pick_smoke), ("probe", pick_probe)):
            rows = picker(all_rows)
            status = write_dataset(name, rows, args.rebuild, args.check)
            manifest_datasets[name] = {
                "sources": {},
                "n_rows": len(rows),
                "n_inputs": len(rows),
                "n_with_context": sum(1 for r in rows if r.context),
                "from": [r.meta["source_stim_uid"] for r in rows],
            }
            print(f"{name:12s} rows={len(rows):4d}  [{status}]")
    else:
        print("(smoke/probe sets need every dataset; skipped)")

    leaked = [p for p in SOURCES_SEEN for h in HOLDOUT_PATHS if p == h or p.startswith(h)]
    assert not leaked, f"a converter opened reserved human data: {leaked}"

    if args.check:
        print("\n--check: nothing written")
        return

    sha, dirty = git_state()
    manifest = {
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "converter_sha256": converter_digest(),
        "git_head_at_build": sha,
        "git_tree_dirty_at_build": dirty,
        "python": sys.version.split()[0],
        "holdout_paths": list(HOLDOUT_PATHS),
        "datasets": manifest_datasets,
    }
    existing = {}
    manifest_path = STIMULI_DIR / "MANIFEST.json"
    if manifest_path.exists() and set(wanted) != set(CONVERTERS):
        existing = json.loads(manifest_path.read_text()).get("datasets", {})
    manifest["datasets"] = {**existing, **manifest_datasets}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")

    total = sum(d["n_rows"] for n, d in manifest_datasets.items() if n in CONVERTERS)
    inputs = sum(d["n_inputs"] for n, d in manifest_datasets.items() if n in CONVERTERS)
    print(f"\n{total} stimulus rows, {inputs} unique model inputs -> {STIMULI_DIR}")
    if dirty:
        print("NOTE: tracked files differ from HEAD; MANIFEST records that, and the converter "
              "content hash either way.")


if __name__ == "__main__":
    main()
