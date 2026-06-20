# Calibration battery — draft v0 (P2)

**The enumerated set lives in `planning/calibration_battery_v0.csv`** (57 items, 25 matched
edit/keep pairs, balanced 28 edit / 29 keep). This doc is the design rationale; the CSV is the
fit-ready artifact. Columns: `item_id, pair_id, family, condition, plausibility, observed, intended,
edit_type, edit_distance, expected, literal_interp, edited_interp, notes`.

Companion to `CALIBRATION_PLAN.md` (§2 blind-design, §5 identifiability) and
`CALIBRATION_P0_OPERATING_POINT.md`. These items are **freshly drafted** in the spirit of the
published noisy-channel paradigms (Ryskin et al.; Gibson, Bergen & Piantadosi 2013). They were
written **without reading** the reserved `data/ryskin2021/` or `data/gibson2013/` materials — the
battery must stay blind to the hold-out. Treat every "expected" note as a *hypothesis to verify*,
not a label.

**Status of each item: unverified.** Before any item enters the fit it must pass the P3 gates:
(1) **LM-noticeability** — the LM scores the intended reading above the literal one,
`slp(intended) > slp(literal)` by a real margin; (2) **candidate-reachability** — the intended word
is retrievable from the observed within `max_dist` (=2 today); (3) **representability** — the fix is
not a dropped *multi-token* word. Items that fail are model-capacity issues, not calibration, and
are dropped (Plan §1.2–1.3).

---

## 1. What each family identifies

| Family | Edit the comprehender infers | Parameter it pins | Seed example |
|---|---|---|---|
| **SUB-W** real-word malapropism (Ryskin-style) | substitute one real word for a near neighbor | θ_sub **and** the λ / plausibility interaction (the literal is itself a word with a prior) | "told an amusing **antidote**" → *anecdote* |
| **SUB-N** non-word typo, graded distance | substitute a non-word for its intended word | θ_sub and its **edit-distance slope** (literal has ~no prior) | "did you **recieve** the message" → *receive* |
| **DEL** missing function word (Gibson-style) | restore a dropped *to/for/from* | `WDEL` (missing-word cost) | "gave the candle **a** daughter" → *to a daughter* |
| **INS** doubled / spurious word | delete a spurious unit | `ρ_ins` (insertion rate) | "the boy **handed handed** the pencil" → *handed* |
| **CTRL** no-edit controls | none — leave it alone | pins the **don't-over-edit** side of every cost | "The pirate buried the treasure." |

The **plausibility crossing** (below) is what isolates λ: the same surface edit is offered when the
literal reading is implausible (edit motivated) vs. plausible (edit not motivated).

---

## 2. SUB-W — real-word malapropisms (Ryskin-style)

The observed word is a *real* word but contextually anomalous; the intended word is an
orthographic/phonological near-neighbor that the context strongly predicts. These are the graded,
interesting cases: the channel must overcome both the edit cost *and* the literal word's own prior,
so humans edit only *sometimes* — exactly the uncertainty we want to capture. Signature: contextual
surprisal ≫ unigram surprisal (the rejuvenation gate's own trigger).

| id | observed | intended | edit (dist) | hypothesis to verify |
|---|---|---|---|---|
| SUBW-01 | the storyteller told an amusing **antidote** | anecdote | sub ×2 (d=2) | strong edit; antidote is anomalous after "amusing/told" |
| SUBW-02 | the medics treated the wound to prevent an **inflection** | infection | del 1 char (d=1) | strong edit (matches the user's "medics" golden case) |
| SUBW-03 | the judge set an important **president** | precedent | sub ×2 (d=2) | strong edit; legal frame predicts *precedent* |
| SUBW-04 | the explorers trekked across the scorching **dessert** | desert | del 1 char (d=1) | strong edit; "scorching" predicts *desert* |
| SUBW-05 | she wore a very **causal** outfit to the party | casual | transpose (d=1) | strong edit; transposition is the cheap channel arc |
| SUBW-06 | the waiter brought a **complimentary** dessert | (keep) | n/a | **near-control**: real word, contextually fine → should *not* edit |

SUBW-06 is deliberately a real word with a tempting neighbor ("complementary") that context does
*not* favor — it checks that the model doesn't edit a word that's already right.

---

## 3. SUB-N — non-word typos at graded distance (θ_sub slope)

Here the literal is a non-word, so the LM crushes it and the only thing standing between observed and
correction is the channel cost. Varying edit distance over many items traces the per-edit cost slope
(don't lean on one contrived d=1/2/3 triple — pool across items).

| id | observed | intended | dist | note |
|---|---|---|---|---|
| SUBN-01 | did you **recieve** the message | receive | 1 (transpose) | |
| SUBN-02 | the boy did an **experimemt** today | experiment | 1 (sub) | |
| SUBN-03 | she **definately** agreed with the plan | definitely | 1 (sub) | |
| SUBN-04 | they **reciesofted** the parcel yesterday | received | ~3 | far edit — expect **low** edit rate; tests the slope's tail |
| SUBN-05 | the **mountian** was covered in snow | mountain | 1 (transpose) | |

SUBN-04 is intentionally near the `max_dist` edge: it should be *rarely* corrected, and if the
candidate generator can't even reach the intended word it's a reachability-gate drop, not a low
human rate — record which.

---

## 4. DEL — missing function word (Gibson-style)

Observed is a double-object frame whose literal reading is implausible (the first object can't be a
recipient); the intended reading inserts *to/for/from*, i.e. the channel **deleted** that word.
Pins `WDEL`. (`to/for/from` are single-token, so no multi-token-deletion caveat.)

**Dative (to):**
| id | observed | intended | missing |
|---|---|---|---|
| DEL-to-01 | The mother gave the candle a daughter | The mother gave the candle **to** a daughter | to |
| DEL-to-02 | The waiter served the soup the customers | …served the soup **to** the customers | to |
| DEL-to-03 | The teacher read the story the children | …read the story **to** the children | to |

**Benefactive (for):**
| id | observed | intended | missing |
|---|---|---|---|
| DEL-for-01 | The tailor sewed the dress the bride | …sewed the dress **for** the bride | for |
| DEL-for-02 | The father cooked the dinner the family | …cooked the dinner **for** the family | for |

**Transitive→intransitive (from) — Gibson's other frame:**
| id | observed | intended | missing |
|---|---|---|---|
| DEL-from-01 | The businessman benefited the tax law | …benefited **from** the tax law | from |
| DEL-from-02 | The patient slowly recovered the illness | …recovered **from** the illness | from |

---

## 5. INS — doubled / spurious word (ρ_ins)

| id | observed | intended | spurious |
|---|---|---|---|
| INS-01 | the boy handed handed the pencil to the girl | …handed the pencil… | doubled "handed" |
| INS-02 | the cat sat on on the mat | the cat sat on the mat | doubled "on" |
| INS-03 | she quickly quickly finished her lunch | …quickly finished… | doubled "quickly" |

Note (from the operating-point work + a prior finding): a *single* doubled content word can be a
genuine model near-tie at weak LMs — INS items may need the sharper LM (410m) to read cleanly. Record
the LM used; don't fix it by moving ρ_ins.

---

## 6. CTRL — no-edit controls (pin the don't-over-edit side)

Clean, plausible sentences a person leaves alone (target adopt-edit ≈ 0). Includes **rare-but-correct**
words (the frequency-aware-insertion concern: a rare word must not be dropped/edited as spurious) and
**plausible versions of the edited frames** (so the same words in a fine context are left alone).

| id | observed | why it's here |
|---|---|---|
| CTRL-01 | The pirate buried the treasure. | clean; rare content word kept |
| CTRL-02 | The chef cooked the salmon. | clean baseline |
| CTRL-03 | The storyteller told an amusing anecdote. | the *corrected* SUBW-01 — must stay put |
| CTRL-04 | The mother gave the daughter a candle. | plausible DO of DEL-to-01 — no "to" inferred |
| CTRL-05 | The businessman benefited from the tax law. | the *complete* DEL-from-01 — must stay put |

---

## 7. The plausibility crossing (the λ identifier)

The single most important *design* move: cross **literal plausibility** × a **fixed-distance
alternative**, holding the surface edit constant. Editing in the implausible cell but not the
plausible cell is the behavioral signature of λ (how far the LM prior is allowed to override the
text). Minimal pairs already embedded above:

| Frame | implausible literal (edit motivated) | plausible literal (edit *not* motivated) |
|---|---|---|
| dative | DEL-to-01 "gave the candle a daughter" | CTRL-04 "gave the daughter a candle" |
| from-frame | DEL-from-01 "benefited the tax law" | CTRL-05 "benefited from the tax law" |
| malaprop | SUBW-01 "amusing antidote" | CTRL-03 "amusing anecdote" |

To make λ well-identified we want **several graded plausibility levels**, not just a 2-way split —
e.g. dative recipients ranging from impossible (candle) → odd → fine — so the edit rate traces a
curve against plausibility rather than a step. That graded set is the main expansion for v1.

---

## 8. Per-item metadata schema (so items are fit-ready)

Each item carries, for the fidelity gate and the Beta-Binomial fit:

```
id, family, observed, intended, response_classes (the partition for aggregation map A),
edit_type {sub, del-to, del-for, del-from, ins, none}, edit_distance (char or word),
literal_plausible {yes, no, graded-level}, expected_direction {edit, keep},
gates: {lm_noticeable?, within_max_dist?, multitoken_deletion?}  // filled at P3
```

`response_classes` is the comprehension-question partition that maps model posterior → human
response space (Plan §3): e.g. for DEL-to-01, {literal: "the candle is the recipient", edited: "a
daughter is the recipient (to-dative)"}; for SUBW-01, {literal: "antidote", edited: "anecdote"}.

---

## 8b. P3 verification results (ran `calibration_gate.py`, pythia-70m, prime ".", max_dist 2)

**27/28 edit items fit-ready; 29/29 keep controls fluent; 0 weak controls.** The one drop is
SUBN-05a, the deliberate far typo (gain +36 but beyond max_dist) — a reachability boundary marker,
kept in the CSV but excluded from the fit set. Gated output: `calibration_battery_v0_gated.csv`.

The design produces the two structured signals it was meant to, *measured* not asserted:

- **Ladder grades monotonically** (slp-gain = LM's preference for the correction, by plausibility):
  give frame 2.19 → 0.33 → 0(keep); send frame 4.79 → 2.40 → 0(keep). The edit-wantedness smoothly
  weakens as the literal DO becomes plausible — the graded λ signal.
- **SUBW plausibility contrast is clean on every pair:** large positive gain in the implausible
  context (+6.6 … +12.7) and *negative* gain for the same edit in the plausible context (−1.2 …
  −8.1). The same word is corrected when context disfavors it and kept when context supports it.

**Identifiability read (which items pressure which parameter):** gain magnitude ≈ how hard the LM
already decides, so *low-gain, near-boundary* items are the ones that constrain a cost.
- SUBN/SUBW/INS gains are huge (+10…+11): the LM is near-certain, so these mostly pin behavior in
  the easy regime and weakly constrain the channel costs.
- **Datives are low-gain (DEL_TO +2.75, DEL_FOR +2.06) — a feature, not a weakness:** they sit near
  the decision boundary, exactly where `WDEL` is identifiable (a small `WDEL` change flips them). The
  ladder mid/low rungs (+0.3…+4.8) play the same near-boundary role for λ.

## 9. What this v0 is and isn't

- **Coverage:** all five families + the no-edit side + an embedded plausibility crossing — enough to
  exercise every free parameter (θ_sub, `WDEL`, `ρ_ins`, λ) and the over-edit guard.
- **Not yet:** (a) the graded plausibility ladder (§7) that makes λ well-identified; (b) item counts
  per cell large enough for stable Beta-Binomial fits (v0 is ~5/cell — a design skeleton, not the
  final N); (c) verification — **no item is confirmed LM-noticeable yet** (that's P3, and needs the LM
  loaded). Some drafted edits may not survive the gate (e.g. SUBW-03 "president"→"precedent" if the
  70m LM doesn't prefer it; SUBN-04 if unreachable).
- **Next:** expand to the graded ladder + adequate counts, then run P3 (fidelity + reachability +
  noticeability) to filter to the fit-ready set.
```
