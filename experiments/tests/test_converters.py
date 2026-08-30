"""Gates for the stimulus build.

    conda run -n ncgenjax python -m pytest -q experiments/tests/

Stdlib only except for one test that checks ``common.classify_edit`` still agrees with the
model-side ``calibration_gate.word_change`` it mirrors; that one skips if jax is not importable.
"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

import build_stimuli as bs                                  # noqa: E402
from converters import CONVERTERS                           # noqa: E402
from converters import common                               # noqa: E402
from converters.common import REPO_ROOT, classify_edit, normalize, standardize   # noqa: E402


def _hash_tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def built() -> dict[str, list]:
    """Every dataset, built once in memory (nothing written)."""
    return {name: bs.build_dataset(name, conv)[0] for name, conv in CONVERTERS.items()}


@pytest.fixture(scope="module")
def repairs() -> dict[str, list]:
    """The (stimulus, repair) records for every dataset."""
    return {name: bs.build_dataset(name, conv)[2] for name, conv in CONVERTERS.items()}


# ---------------------------------------------------------------------------------------
# The source materials are read-only
# ---------------------------------------------------------------------------------------

def test_build_does_not_modify_the_source_materials():
    """A full build must leave every file under data/ byte-identical.

    data/ is gitignored, so those files are the only copy in a checkout and a converter that
    wrote to one would destroy it silently.  This hashes the whole tree either side of a build.
    """
    before = _hash_tree(REPO_ROOT / "data")
    for name, conv in CONVERTERS.items():
        bs.build_dataset(name, conv)
    after = _hash_tree(REPO_ROOT / "data")
    assert before == after, \
        f"the build changed files under data/: {sorted(set(before) ^ set(after)) or
           [k for k in before if before[k] != after.get(k)]}"


def test_open_source_refuses_every_holdout_path():
    for holdout in common.HOLDOUT_PATHS:
        probe = holdout if not holdout.endswith("/") else holdout + "anything.csv"
        with pytest.raises(common.HoldoutError):
            common.open_source(probe)


def test_open_source_refuses_a_holdout_path_given_as_absolute():
    with pytest.raises(common.HoldoutError):
        common.open_source(REPO_ROOT / "data/clark2026/exp_data_merged.csv")


def test_open_source_returns_a_detached_buffer():
    """Not a file handle: a converter cannot reach the file on disk through it."""
    fh = common.open_source("data/moses/materials.csv")
    assert isinstance(fh, io.StringIO)
    fh.write("this stays in memory")           # would be a disaster on a real handle
    assert (REPO_ROOT / "data/moses/materials.csv").read_text().startswith("Item,sentence")


def test_every_opened_source_is_hashed(built):
    assert common.SOURCES_SEEN, "no sources recorded"
    for path, digest in common.SOURCES_SEEN.items():
        actual = hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        assert digest == actual, f"{path}: recorded hash does not match the file"


def test_no_holdout_file_was_opened(built):
    leaked = [p for p in common.SOURCES_SEEN
              for h in common.HOLDOUT_PATHS if p == h or p.startswith(h)]
    assert not leaked, f"reserved human data was opened: {leaked}"


# ---------------------------------------------------------------------------------------
# Text conventions
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("dataset,path", [
    ("gibson2013/dopo_to", "data/gibson2013/dopo_to/sentences.txt"),
    ("gibson2013/dopo_for", "data/gibson2013/dopo_for/sentences.txt"),
    ("gibson2013/transitive_intransitive", "data/gibson2013/transitive_intransitive/sentences.txt"),
    ("huang2024", "data/huang2024/sentences.txt"),
    ("clark2026", "data/clark2026/sentences.txt"),
    ("ryskin2021", "data/ryskin2021/sentences.txt"),
    ("moses", "data/moses/sentences.txt"),
])
def test_normalize_reproduces_the_legacy_sentence_files(built, dataset, path):
    """sentence_norm must still join to the old pipeline's per-study sentences.txt."""
    ds, _, subset = dataset.partition("/")
    rows = [r for r in built[ds] if r.subset == subset]
    legacy = [ln for ln in (REPO_ROOT / path).read_text().splitlines() if ln.strip()]
    assert [r.sentence_norm for r in rows] == legacy


def test_qian_legacy_file_is_a_subset(built):
    """qian2023's shipped sentences.txt holds only 120 of the 480 rows.

    118 of those 120 lines are distinct: items 55 and 57 repeat, which is the same source defect
    that makes qian's 480 rows collapse to 472 model inputs (see EXPECTED_ANOMALIES).
    """
    legacy = [ln for ln in (REPO_ROOT / "data/qian2023/sentences.txt").read_text().splitlines()
              if ln.strip()]
    produced = {r.sentence_norm for r in built["qian2023"]}
    assert len(legacy) == 120 and len(set(legacy)) == 118
    assert set(legacy) <= produced


def test_standardize_is_idempotent(built):
    for rows in built.values():
        for r in rows:
            assert standardize(r.model_input) == r.model_input
            if r.context:
                assert standardize(r.context) == r.context


def test_standardize_examples():
    assert standardize("the gifts for the kid is hidden under the bed .") == \
        "The gifts for the kid is hidden under the bed."
    assert standardize("how many animals did moses take ?") == "How many animals did moses take?"
    assert standardize("Already fine.") == "Already fine."
    assert standardize("no terminal mark") == "No terminal mark."


def test_normalize_and_standardize_agree_on_words(built):
    """The two conventions may differ in case and punctuation attachment, nothing else."""
    for rows in built.values():
        for r in rows:
            a = normalize(r.model_input)
            b = r.sentence_norm
            assert a == b, f"{r.stim_uid}: normalize(model_input)={a!r} != sentence_norm={b!r}"


# ---------------------------------------------------------------------------------------
# Edit classification
# ---------------------------------------------------------------------------------------

def test_classify_edit_shapes():
    assert classify_edit("a b c", "a b c").type == "none"
    assert classify_edit("The mother gave the candle the daughter.",
                         "The mother gave the candle to the daughter.")[:2] == ("ins", "ins")
    assert classify_edit("The mother gave the daughter to the candle.",
                         "The mother gave the daughter the candle.")[:2] == ("del", "del")
    # one opcode spanning two words is still a single edit, not "multi"
    e = classify_edit("the player tossed a frisbee", "the player who was tossed a frisbee")
    assert (e.type, e.to, e.obs_gap) == ("ins", "who was", 2)
    # two separate opcodes are
    assert classify_edit("The ball kicked the girl.",
                         "The ball was kicked by the girl.").ops == "ins;ins"


def test_classify_edit_matches_the_model_side_word_change(built, repairs):
    """common.classify_edit restates calibration_gate.word_change; they must not drift."""
    pytest.importorskip("jax", reason="calibration_gate imports jax and the penzai LM")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from genjax_port import calibration_gate

    checked = 0
    for name, rows in built.items():
        by_uid = {r.stim_uid: r for r in rows}
        for e in repairs[name]:
            observed = by_uid[e["stim_uid"]].model_input
            mine = classify_edit(observed, e["intended_text"])
            theirs = calibration_gate.word_change(observed, e["intended_text"])
            if mine.type == "multi":
                continue                       # word_change reports only its first opcode
            assert (mine.type, mine.frm, mine.to) == theirs, \
                f"{e['stim_uid']}: {mine[:4]} vs {theirs}"
            checked += 1
    assert checked > 2000, f"only {checked} pairs compared"


# ---------------------------------------------------------------------------------------
# Dataset invariants
# ---------------------------------------------------------------------------------------

def test_row_counts(built):
    assert {k: len(v) for k, v in built.items()} == bs.EXPECTED_ROWS


def test_every_intended_uid_resolves(built, repairs):
    for name, rows in built.items():
        uids = {r.stim_uid for r in rows}
        for r in rows:
            for u in r.intended_uids:
                assert u in uids, f"{name}: {r.stim_uid} -> {u}"
        by_stim = collections.Counter(e["stim_uid"] for e in repairs[name])
        for r in rows:
            assert by_stim[r.stim_uid] == len(r.intended_uids), \
                f"{name}: {r.stim_uid} has {by_stim[r.stim_uid]} repair rows for " \
                f"{len(r.intended_uids)} intended_uids"
        for e in repairs[name]:
            assert e["intended_text"], f"{name}: {e['stim_uid']} has an unresolved intended_text"


def test_qian_carries_both_repairs_with_no_primacy(built, repairs):
    """The sentence is ambiguous about which word is wrong; the schema must not pick one."""
    stim = {r.stim_uid: r for r in built["qian2023"]}
    by_stim = collections.defaultdict(list)
    for e in repairs["qian2023"]:
        by_stim[e["stim_uid"]].append(e)
    n_two = 0
    for uid_, es in by_stim.items():
        cond = stim[uid_].condition
        targets = {e["intended_uid"].rsplit("/", 1)[1] for e in es}
        verb_route, noun_route = cond[0] + cond[1] + cond[0], cond[2] + cond[1] + cond[2]
        assert targets == {verb_route, noun_route}, f"{uid_}: {targets}"
        if cond[0] != cond[2]:
            n_two += 1
            assert len(es) == 2, f"{uid_}: ungrammatical rows need both routes"
            # the two routes touch DIFFERENT words -- that is what makes the row ambiguous
            assert len({e["edit_obs_idx"] for e in es}) == 2, f"{uid_}: routes edit the same word"
        else:
            assert len(es) == 1 and es[0]["edit_type"] == "none"
    assert n_two == 240, n_two


def test_stim_uids_are_unique_across_all_datasets(built):
    seen: set[str] = set()
    for rows in built.values():
        for r in rows:
            assert r.stim_uid not in seen, f"duplicate stim_uid {r.stim_uid}"
            seen.add(r.stim_uid)


def test_sentence_ids_are_dense_and_consistent(built):
    for name, rows in built.items():
        by_id: dict[int, tuple[str, str]] = {}
        for r in rows:
            key = (r.context, r.model_input)
            assert by_id.setdefault(r.sentence_id, key) == key, \
                f"{name}: sentence_id {r.sentence_id} maps to two different inputs"
        assert sorted(by_id) == list(range(len(by_id))), f"{name}: sentence_ids are not dense"


def test_model_inputs_follow_the_battery_convention(built):
    for name, rows in built.items():
        for r in rows:
            mi = r.model_input
            assert mi[0].isupper(), f"{name}: {r.stim_uid} is not capitalized: {mi!r}"
            assert mi[-1] in ".?!", f"{name}: {r.stim_uid} has no terminal mark: {mi!r}"
            assert "  " not in mi and mi == mi.strip()


def test_only_chen2023_carries_a_context(built):
    for name, rows in built.items():
        with_ctx = [r for r in rows if r.context]
        assert bool(with_ctx) == (name == "chen2023"), f"{name} unexpectedly has contexts"
    assert len([r for r in built["chen2023"] if r.context]) == 320   # 2 subsets x 2 contexts x 80


def test_chen2023_targets_match_the_no_context_file(built):
    """The context conditions must present exactly the no-context target, prime aside."""
    rows = built["chen2023"]
    none = {(r.subset, r.item_id, r.meta["linger_cond"]): r.model_input
            for r in rows if r.meta["context_type"] == "none"}
    for r in rows:
        key = (r.subset, r.item_id, r.meta["linger_cond"])
        assert r.model_input == none[key], f"{r.stim_uid}: target differs from the no-context file"


def test_chen2023_dopo_to_no_context_matches_gibson2013(built):
    """A documented overlap: the same 80 sentences appear in both studies."""
    chen = {r.model_input for r in built["chen2023"]
            if r.subset == "dopo_to" and r.meta["context_type"] == "none"}
    gibson = {r.model_input for r in built["gibson2013"] if r.subset == "dopo_to"}
    assert chen == gibson and len(chen) == 80


def test_critical_word_indices_point_at_the_expected_word(built):
    """huang2024, clark2026 and tabor2004 each name their critical word; check the index."""
    checks = {
        "huang2024": lambda r: r.meta["disamb_word"],
        "clark2026": lambda r: r.meta["critical_word"],
        "tabor2004": lambda r: r.meta["participle"],
    }
    for name, expected in checks.items():
        for r in built[name]:
            assert r.critical_word_idx is not None, f"{name}: {r.stim_uid} has no critical index"
            tok = common.strip_punct(r.model_input.split()[r.critical_word_idx]).lower()
            assert tok == common.strip_punct(expected(r)).lower(), \
                f"{name}: {r.stim_uid} index {r.critical_word_idx} -> {tok!r}, want {expected(r)!r}"


def test_qian_grammaticality(built):
    for r in built["qian2023"]:
        assert r.is_grammatical == (r.condition[0] == r.condition[2])
        assert len(r.intended_uids) == (1 if r.is_grammatical else 2)


def test_gibson_counterpart_is_one_preposition(built, repairs):
    """The whole point of the design: implausible rows are one word from a plausible reading."""
    plaus = {r.stim_uid: r.plausibility for r in built["gibson2013"]}
    got = {"ins": 0, "del": 0}
    for e in repairs["gibson2013"]:
        if plaus[e["stim_uid"]] != "implausible":
            continue
        if e["edit_type"] in got:
            got[e["edit_type"]] += 1
            word = (e["edit_to"] or e["edit_from"]).lower()
            assert len(word.split()) == 1, f"{e['stim_uid']}: edit is not one word: {word!r}"
            # dopo_to varies "to", dopo_for "for", transitive_intransitive "from" or "inside"
            assert word in {"to", "for", "from", "inside"}, \
                f"{e['stim_uid']}: unexpected edit word {word!r}"
    assert got == {"ins": 58, "del": 60}, got   # 2 rows are source typos; see EXPECTED_ANOMALIES


def test_known_source_anomalies(built, repairs):
    for name, rows in built.items():
        found = bs.find_anomalies(rows, repairs[name])
        assert len(found) == bs.EXPECTED_ANOMALIES.get(name, 0), \
            f"{name}: {len(found)} anomalies, expected {bs.EXPECTED_ANOMALIES.get(name, 0)}"


def test_holdout_paths_still_exist_to_be_protected():
    """A guard that protects a path that has moved is not protecting anything."""
    for holdout in common.HOLDOUT_PATHS:
        assert (REPO_ROOT / holdout).exists(), f"HOLDOUT_PATHS entry {holdout} no longer exists"


# ---------------------------------------------------------------------------------------
# Build outputs
# ---------------------------------------------------------------------------------------

def test_smoke_set_covers_every_phenomenon(built):
    rows = bs.pick_smoke(built)
    assert len(rows) == len(bs.SMOKE_UIDS)
    sources = [r.meta["source_dataset"] for r in rows]
    assert set(sources) == {"gibson2013", "chen2023", "ryskin2021", "qian2023",
                            "huang2024", "clark2026", "tabor2004"}
    assert sum(1 for r in rows if r.context) == 1, "one smoke item must exercise a context prime"


def test_probe_set_is_the_worst_case_per_dataset(built):
    """One longest-sentence row per dataset, plus the longest chen2023 context.

    chen2023 therefore contributes two rows: cost has two axes, sentence length and prime length,
    and the row that maximizes one is not the row that maximizes the other.
    """
    rows = bs.pick_probe(built)
    picked: dict[str, list] = {}
    for r in rows:
        picked.setdefault(r.meta["source_dataset"], []).append(r)
    for name, dataset_rows in built.items():
        longest = max(len(r.model_input.split()) for r in dataset_rows)
        assert longest in [len(r.model_input.split()) for r in picked[name]], \
            f"{name}: no probe row has the longest sentence ({longest} words)"
    assert max(len(r.context.split()) for r in rows) == \
        max(len(r.context.split()) for r in built["chen2023"])


def test_append_only_guard_fires(tmp_path):
    path = tmp_path / "x.input.jsonl"
    original = [{"sentence_id": 0, "text": "A.", "context": ""},
                {"sentence_id": 1, "text": "B.", "context": ""}]
    path.write_text("".join(json.dumps(r) + "\n" for r in original))

    assert bs.check_append_only(path, original, rebuild=False) == "unchanged"
    appended = original + [{"sentence_id": 2, "text": "C.", "context": ""}]
    assert bs.check_append_only(path, appended, rebuild=False) == "appended +1"

    reordered = [original[1], original[0]]
    with pytest.raises(SystemExit, match="append-only"):
        bs.check_append_only(path, reordered, rebuild=False)
    assert "REBUILT" in bs.check_append_only(path, reordered, rebuild=True)


def test_manifest_records_the_current_converters():
    """stimuli/ must have been built by the converter code that is checked in.

    A content hash rather than a commit sha: a manifest cannot record the commit that contains
    it, so the sha would always be one behind (or, after an amend, an orphan).
    """
    manifest_path = bs.STIMULI_DIR / "MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip("stimuli not built yet; run experiments/build_stimuli.py")
    recorded = json.loads(manifest_path.read_text())["converter_sha256"]
    assert recorded == bs.converter_digest(), (
        "experiments/stimuli/ is stale: the converters have changed since it was built. "
        "Re-run experiments/build_stimuli.py and commit the result.")


def test_repairs_table_matches_a_fresh_build(repairs):
    import csv
    for name, recs in repairs.items():
        path = bs.STIMULI_DIR / f"{name}.repairs.csv"
        if not path.exists():
            pytest.skip(f"{path.name} not built yet; run experiments/build_stimuli.py")
        on_disk = list(csv.DictReader(path.open()))
        assert len(on_disk) == len(recs), f"{name}: {len(on_disk)} on disk, {len(recs)} fresh"
        for a, b in zip(on_disk, recs):
            assert (a["stim_uid"], a["intended_uid"], a["edit_type"]) == \
                   (b["stim_uid"], b["intended_uid"], b["edit_type"])


def test_written_stimuli_match_a_fresh_build(built):
    """experiments/stimuli/ on disk must be what the converters produce right now."""
    import csv
    for name, rows in built.items():
        csv_path = bs.STIMULI_DIR / f"{name}.stimuli.csv"
        if not csv_path.exists():
            pytest.skip(f"{csv_path.name} not built yet; run experiments/build_stimuli.py")
        on_disk = list(csv.DictReader(csv_path.open()))
        assert len(on_disk) == len(rows), f"{name}: {len(on_disk)} rows on disk, {len(rows)} fresh"
        for a, b in zip(on_disk, rows):
            assert a["stim_uid"] == b.stim_uid
            assert a["intended_uids"] == ";".join(b.intended_uids)
            assert a["model_input"] == b.model_input
            assert a["context"] == b.context
            assert int(a["sentence_id"]) == b.sentence_id
