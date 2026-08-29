# `experiments/` — the noisy-channel experiment harness

Runs the genjax noisy-channel model over the published stimulus sets in `data/<study>/`, on the MIT ORCD
SLURM cluster, and collects the results into analysis-friendly tables.

**Status: Phase 1 complete** (stimulus harmonization). The design, the per-phase work and the decisions
taken with the user live in
[`../planning/NOISY_CHANNEL_HARNESS_IMPLEMENTATION_PLAN.md`](../planning/NOISY_CHANNEL_HARNESS_IMPLEMENTATION_PLAN.md);
the goals it serves live in
[`../planning/NOISY_CHANNEL_EXPERIMENT_HARNESS.md`](../planning/NOISY_CHANNEL_EXPERIMENT_HARNESS.md).
This file becomes the full reproduction guide in Phase 6.

## Layout

| path | what it is | phase |
|---|---|---|
| `converters/` | one module per study: raw `data/<study>/…` -> the common stimulus schema | 1 ✅ |
| `build_stimuli.py` | runs the converters -> `stimuli/`, checks invariants, writes `MANIFEST.json` | 1 ✅ |
| `stimuli/` | the harmonized stimuli (**tracked** — this is the reproducibility anchor) | 1 ✅ |
| `tests/` | gates for the build: read-only sources, conventions, per-dataset invariants | 1 ✅ |
| `configs/` | named model configs, as env files for `slurm/submit_nc_batch.sh` | 4 |
| `run.sh` | `fetch-tabor \| build \| smoke-local \| probe \| submit \| status \| pull \| collect` | 4–5 |
| `collect.py` | `results_nc/**/item_*.json` -> `outputs/<config>/<dataset>/*.csv.gz` | 3 |
| `outputs/` | tidy results (**untracked**; the raw per-item JSON stays in `results_nc/`) | 5 |
| `RUNLOG.md` | append-only record of every cluster launch | 5 |

The compute layer is the existing `slurm/run_nc_batch.py` + `slurm/submit_nc_batch.sh`, which this harness
wraps rather than replaces.

## Running it

```sh
python experiments/build_stimuli.py                 # build everything
python experiments/build_stimuli.py --check         # report what would change, write nothing
python experiments/build_stimuli.py qian2023        # rebuild one dataset
conda run -n ncgenjax python -m pytest -q experiments/tests/      # 37 gates
```

The build is stdlib-only and takes under a second. The test suite needs `pytest`; one test additionally
imports jax (it checks `classify_edit` still agrees with the model-side `calibration_gate.word_change`)
and skips if jax is unavailable, which is why the `ncgenjax` env is used above.

## The blind protocol

The order of operations is: run the model on the stimuli **blind to the human data**, and only compare to
human behavior once the model results are in. Concretely, nothing under `experiments/` may read a
human-data file. `converters/common.py` enforces this:

- every source file is opened through `open_source()`, which refuses the paths listed in `HOLDOUT_PATHS`
  (currently `data/clark2026/exp_data_merged.csv` and `data/clark2026/lists/`);
- `open_source()` returns an in-memory `StringIO` over the file's bytes rather than a file handle, so a
  converter is never in a position to write to, truncate or move a source file. `data/` is gitignored,
  so those files are the only copy in a checkout — `test_build_does_not_modify_the_source_materials`
  hashes the whole tree either side of a full build and asserts it is byte-identical;
- every file that *is* opened gets its sha256 recorded into `stimuli/MANIFEST.json`. Since `data/` is not
  tracked, those hashes plus the converters are what makes the build reproducible.

`MANIFEST.json` also carries a `converter_sha256` — a content hash of the converter sources — and
`test_manifest_records_the_current_converters` asserts `stimuli/` was built by the code that is checked in.
It is a content hash rather than a commit sha because a manifest cannot record the commit that contains it:
committing the manifest changes that commit, and amending would leave it pointing at an orphan. `git_head_at_build`
is recorded alongside for human orientation; the commit carrying the manifest is a descendant of it.

## Outputs

`build_stimuli.py` writes, per dataset:

- **`<dataset>.stimuli.csv`** — one row per (item, condition) in the common schema below.
- **`<dataset>.input.jsonl`** — what the model actually reads: the unique `(context, model_input)` pairs,
  one JSON record per line, `{"sentence_id", "text", "context"}`.

plus `smoke.*` (one item per phenomenon, for the pipeline smoke test), `probe.*` (the worst case for
runtime and memory, for the cost probe) and `MANIFEST.json`.

### The input lists are append-only

The SLURM worker resumes per item by **line index** into the input file, so inserting or reordering a line
would silently re-map finished results onto different sentences. A rebuild that would change an existing
line fails with an explanation. `--rebuild` overrides it and means that dataset's computed results must be
discarded.

### Common schema

| field | notes |
|---|---|
| `dataset`, `subset` | e.g. `gibson2013`, `dopo_to`; `subset` is empty when the study has none |
| `item_id`, `condition`, `stim_uid` | `stim_uid = dataset/subset/item_id/condition`, unique across all datasets |
| `sentence_orig` | original orthography where the source preserves it; empty for ryskin2021 / qian2023, whose materials ship already normalized |
| `sentence_norm` | lowercase, punctuation split off — the old pipeline's `sentences.txt` convention, kept so these tables still join to legacy per-study files |
| `model_input` | **what the model reads.** Initial capital, punctuation attached, terminal `.`/`?` |
| `context` | clean preceding text for the LM prime; non-empty only for chen2023 |
| `sentence_id` | index into `<dataset>.input.jsonl`; rows with identical `(context, model_input)` share one |
| `plausibility`, `is_grammatical` | whichever the study manipulates; empty when it manipulates neither |
| `contrast` | the **design-level** relation to the counterpart (see below) |
| `intended_uid`, `intended_text` | the counterpart row and its `model_input`; empty when the design defines none |
| `edit_type`, `edit_ops`, `edit_from`, `edit_to` | the **word-level** difference from difflib |
| `critical_word_idx` | 0-based index into `model_input`'s whitespace tokens |
| `comprehension_q`, `correct_answer` | gibson2013 / chen2023 / tabor2004 — a normative answer key, not human data |
| `meta` | JSON of the dataset-specific source columns |

### `contrast` vs `edit_type`

`contrast` says what the *design* varies; `edit_type` says what difflib needs at the *word* level. They can
disagree, and the disagreement is informative rather than a defect. chen2023's voice alternation is the
clearest case:

```
The ball kicked the girl.   -> The ball was kicked by the girl.    two insertions  -> edit_ops "ins;ins"
The truck drove the man.    -> The truck was driven by the man.    one replacement -> edit_ops "sub"
```

Both are `contrast == "voice"`; the second only looks different because the irregular participle breaks the
word match. **Group by `contrast` for analysis; read `edit_type` / `edit_ops` for what the channel has to
do.** `edit_type` is `sub` / `ins` (a missing word is restored) / `del` (a spurious word is removed) /
`none` / `multi` (more than one difflib opcode). A single opcode spanning several adjacent words — tabor2004
restoring `"who was"` — is *not* `multi`.

### Standardization

Every dataset gets the same treatment, which is the convention the calibration battery already uses
(`calibration_word_action_smc._wellform`): collapse whitespace, re-attach punctuation the source split off,
capitalize the initial letter, ensure a terminal `.`/`?`. Applying it uniformly is what keeps the model's
leading-opener artifact from firing on the lowercase-only sources (ryskin2021, qian2023). `sentence_norm`
preserves the old convention alongside it, and is verified line-for-line against every `sentences.txt` the
studies ship.

## The datasets

| dataset | rows | inputs | source | condition | counterpart | contrast |
|---|---|---|---|---|---|---|
| gibson2013 | 240 | 240 | `<subset>/materials.csv` × 3 subsets | `Structure_Plausibility` | plausible row of the **other** structure | `dative`, `transitivity` |
| chen2023 | 480 | 480 | 6 Linger `.txt` × 2 subsets | `<linger_cond>.<context>` | as gibson2013 | `dative`, `voice` |
| ryskin2021 | 504 | 504 | `materials.csv` | `Control`/`SemCrit`/`Sem`/`Synt` | the `Control` row | `word_form` |
| qian2023 | 480 | 472 | `materials.csv` | `sss`…`ppp` (N1, N2, verb number) | verb number := N1's | `agreement` |
| huang2024 | 144 | 144 | `items_ClassicGP.pivot.csv` | `{NPS,NPZ,MVRR}_{AMB,UAMB}` | the `_UAMB` sibling | `disambiguator` |
| clark2026 | 360 | 360 | `materials.csv` (+ `raw_materials.csv`) | `Label` | see below | `typo`, `word_form` |
| tabor2004 | 128 | 128 | `data/tabor2004/items.csv` | `<reduced_rel>_<coherence>` | the `nonreduced` sibling | `relativizer` |
| moses | 1 | 1 | `raw_materials.csv` | `demo` | none | — |

**2,337 stimulus rows, 2,329 unique model inputs.**

### Choosing the counterpart

`intended_uid` names *the sentence a noisy-channel reader would recover*, which for most designs is the
minimal edit away — not necessarily the same-condition control.

- **gibson2013 / chen2023 dative.** The design's point is that an implausible sentence is one word from a
  plausible sentence of the *other* structure, and that the two directions are not equally likely:

  ```
  DO_implausible  The mother gave the candle the daughter.     --insert "to"-->  PO_plausible
  PO_implausible  The mother gave the daughter to the candle.  --delete "to"-->  DO_plausible
  ```

  These are exactly phenomena 1 and 2 in the harness plan. The same-structure plausible sibling — a role
  swap, not a one-word edit — is kept in `meta.same_structure_plausible_uid` for anyone who wants it.
  transitive_intransitive works the same way, varying `from` or `inside`.
- **clark2026.** `Typo1.1 -> 1.1` and `Typo2.2 -> 2.2` restore the typo'd verb; `2.1 -> 1.1` and
  `1.2 -> 2.2` are the noisy-channel garden paths (implausible as written, one keystroke from a plausible
  reading with the other critical verb). The four control labels have no one-edit repair and get no
  counterpart.
- **Rows that are already the target** (plausible, grammatical, unambiguous, nonreduced) point at
  themselves and get `edit_type == "none"`, so joins are uniform.

### chen2023 parsing notes

The context and the target share one line, nominally separated by two spaces — but the separator is
inconsistent, and splitting on `"  "` recovers the wrong target on **27 of the 320** context rows: on 20
(all `dopo_to`/supportive) the separator before the target is three spaces, so the target keeps a leading
space; on 7 (`active_passive`/supportive) two context sentences are separated by a single space, so a whole
context sentence lands inside the "target". Instead the no-context file is the authority for the target text
and the context is whatever prefix precedes it; that matches on all 320 rows and is asserted per row. `dopo-to-supportive.txt` is doubly encoded (right single quotes appear as `‚Äô`); the
parser repairs that specific sequence and then asserts nothing non-ASCII survives. Fillers are excluded.

The no-context `dopo_to` targets are byte-identical to gibson2013's `dopo_to` materials. They are kept —
the two studies are separate datasets with separate input lists — so those 80 sentences are run twice, once
under each dataset's `sentence_id`.

## Known defects in the published materials

The build detects these, records them under `MANIFEST.json` -> `datasets.<name>.anomalies`, and asserts the
count against `EXPECTED_ANOMALIES` so that re-fetching a source can never quietly introduce more. They are
not repaired: editing the stimuli would misrepresent what the studies actually ran.

| dataset | what | effect |
|---|---|---|
| gibson2013 | `dopo_for` item 7's PO_plausible reads "The wife reserved table for her husband." — missing "a" | that item's counterpart is 2 edits away, so `edit_type == "multi"` |
| gibson2013 | `transitive_intransitive` item 13's Preposition_plausible reads "The plastic **handled** melted from the hot stove." | same |
| huang2024 | item 14's `NPZ_UAMB` has "another **political** controversy" where every other condition has "another controversy" | same |
| qian2023 | items 55 and 57 never pluralize N2 ("at the restaurant", "in the bowl"), so the N2-number manipulation is absent from the text | 4 condition pairs per item collapse to one sentence: 480 rows -> 472 inputs. Both conditions share one `sentence_id` and so one model run, which is correct — the model sees one sentence |

Two further points that are *not* defects but are worth knowing:

- **gibson2013 `Answer`** is the answer under the sentence as literally written, and which participant the
  question asks about is counterbalanced across items — exactly half of each Structure × Plausibility cell
  answers "Yes". It cannot be derived from the condition, so it is carried verbatim.
- **tabor2004's relativizer** is not always "who was": across the 64 reduced/nonreduced pairs it is
  "who was" (41), "that was" (14), "who were" (3), "which was" (2), "who is" (2), "that were" (2). The
  converter reads it off the diff rather than assuming, and asserts the one-insertion shape on every pair.

## Adding a dataset

Write one module in `converters/` exposing `convert()`, a generator of `common.StimRow`; add it to
`CONVERTERS` in `converters/__init__.py`; add its expected row count to `EXPECTED_ROWS` in
`build_stimuli.py`. Nothing else needs to change. Read only through `common.open_source()`, and if the
study ships human data, add its path to `HOLDOUT_PATHS` first.
