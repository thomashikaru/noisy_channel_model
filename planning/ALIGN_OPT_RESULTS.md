# Align-channel parameter optimization — results & recommendation (2026-06-21)

**Goal.** On the `align` path (copy+sub merged into one action; no separate copy/sub), find free-parameter
settings that jointly maximize **E** = posterior mass on the *intended correction* for **implausible**
items (e.g. "…amusing antidote" → "…amusing anecdote") and **L** = posterior mass on the *literal string*
for **plausible** items (no correction). Levers explored: K (align substitution cost), ACTION_ALPHA
(ins/del concentration), LM (pythia-70m vs 410m), particle count P, rejuvenation. Branch
`align-action-channel`, channel `align`, rejuv=gibbs (deployment), unless noted.

## Headline

| config (align, gibbs, 70m, P=128) | E_edit | L_keep | combined | junk |
|-----------------------------------|--------|--------|----------|------|
| **old align default** K=−3.26, α=(200,1,1) | 0.141 | 0.765 | 0.453 | 9/40 |
| **NEW default** K=−4.5, α=(200,2,2) | **0.291** | **0.927** | **0.609** | ~1/40 |

3-seed mean on the full 40-item battery. **E roughly doubles (0.14→0.29), L rises to 0.93, junk is
near-eliminated (9/40→~1/40)** — a clean simultaneous gain on both objectives. Applied to
`ALIGN_ALPHA_DEFAULT` / `ALIGN_SLOPE` in `pythia_word_caprop.py` (gated channel; does NOT touch the
`word_action` deployment default). *Not committed — for your review.*

Metric: E = mean over implausible items of mass on the exact intended sentence; L = mean over plausible
items of mass on the exact literal. junk = items with >0.5 mass on neither clean reading (over-edit /
restructure / leading-insertion). Reading-match is case/punctuation-insensitive (`_norm`).

## Diagnosis: the failure was bimodal, and K + alpha are two independent knobs

The old align default failed in two *opposite* ways at once, which is why a single threshold (the prior
Phase-4 K-sweep) could never fix it:

1. **Substitution over-editing** → restructure junk. Real-word typos got corrected but the weak LM also
   restructured the frame ("the storyteller told an amusing antidote" → "She is a storyteller and an
   amusing anecdote", junk=0.91). *Cause:* in the align channel a substitution pays only
   `log(p_align)+K·d ≈ K·d` — there is **no per-word sub penalty** (a sub and a copy are the same
   action). At the old K=log(1/26)=−3.26 a d=1 sub costs only −3.26 nats, ~5 nats **cheaper** than the
   calibrated word_action operating point (reduction-gate equivalent K=−8.56). Phase-4 only swept
   K∈[−4.1,−2.7] (the garage-flip window) and never approached the calibrated threshold. **K is the
   substitution knob.**

2. **Insertion/deletion under-editing AND passive-hallucination keeps.** Dropped function words were not
   restored ("he is good man" kept at p=0.97), while *clean* keeps were over-edited by pythia-70m's
   passive-voice prior ("the boy handed the pencil…" → "the boy **is** handed…", p=0.98; "the waiter
   **was** served…"). Both are governed by `p_ins`/`p_del` (the alpha ins/del entries), **independent of
   K**. The del-action does double duty — legitimately restoring "of"/"a" and illegitimately inserting
   "is"/"was" — so alpha trades them off; the right LM (which knows "he is a good man" > "he is good man"
   *and* "the boy handed" > "the boy is handed") is what truly separates them. **alpha is the ins/del knob.**

The user's goal drops the Phase-4 *garage-flip* constraint, which freed K to sharpen past the narrow
garage window.

## Lever 1 — K (substitution cost), 12-item bimodal subset, gibbs, 70m, P=128, seed 0

| K | E_edit | L_keep | combined | note |
|------|--------|--------|----------|------|
| −3.26 (old) | 0.177 | 0.620 | 0.398 | restructure junk on real-word subs |
| **−4.5** | **0.313** | 0.562 | 0.438 | SUBW-01a 0.09→0.85 (junk 0.91→0.05); knee |
| −5.5 | 0.265 | 0.590 | 0.427 | real-word sub starts dying (SUBW 0.55) |
| −8.0 | 0.177 | 0.830 | 0.503 | ≈ word_action: real-word sub dead (0.00), high L |

K moves **only** the two substitution items; all ins/del items are flat. SUBW-01a (real-word) peaks at
**K=−4.5** then dies; SUBN (nonword, huge LM gain) survives all K. K=−8.0 reproduces word_action (high L,
low E) exactly as the reduction gate predicts. **Pick K=−4.5** — the knee that maximizes real-word
corrections.

## Lever 2 — alpha (ins/del concentration) at K=−4.5, **full 40-item battery**, seed 0

| alpha (align,ins,del) | E_edit | L_keep | combined | junk |
|-----------------------|--------|--------|----------|------|
| (200,1,1) (K only)    | 0.176 | 0.752 | 0.464 | 8/40 |
| **(200,2,2)**         | **0.323** | **0.946** | **0.634** | **0/40** |
| (200,4,4)             | 0.225 | 0.829 | 0.527 | 8/40 |

**alpha=2 is a sharp Pareto knee.** It is enough ins/del to (a) kill the passive hallucinations
(INS-01b 0.00→0.87, DELTO-02b 0.40→0.90) and (b) restore dropped words (DEL-of 0.11→0.70, INS-02a 0→0.95,
SUBW-02a 0→0.98), but **not** so much that the del-action starts inserting plausible words into clean
keeps — which is exactly what alpha=4 does ("the patient's **body** recovered…", DELFROM keeps crash
0.93→0.42 / 0.78→0.23; "experiment **on him**", SUBN-02a 1.00→0.00). alpha=4 overshoots on *both* axes.

## Lever 3 — LM: pythia-410m (12-subset, K=−4.5, α=4, seed 0) — NOT a clear win

| LM | E_edit | L_keep | combined |
|------|--------|--------|----------|
| 70m  | 0.332 | 0.917 | 0.624 |
| 410m | 0.307 | 0.797 | 0.552 |

410m genuinely **fixes** some items the 70m signal can't (DEL-the 0→0.99 "we went to **the** store";
DELFOR-01b keep 0.61→0.98) — confirming the under-editing is partly a 70m signal-quality problem. But it
is offset by: (i) a **new leading-list artifact** — 410m prepends document markers ("**3.** Did you
receive the message"), cratering SUBN to 0.00 even though the correction is present (a known
leading-insertion / prime artifact, out of scope per prior guidance); (ii) it genuinely *accepts* some
"errors" ("He is good man" p=0.98 — 410m judges the article-drop fine); (iii) ~5× slower. **Net wash; 70m
preferred** for this objective. The leading-junk artifact would need a separate prime/LM-strength fix to
give 410m a fair shot.

## Lever 4 — particle count P (12-subset, K=−4.5, α=2, seed 0)

| P | E_edit | L_keep | combined |
|------|--------|--------|----------|
| 128 | 0.307 | 0.885 | 0.596 |
| 256 | 0.333 | 0.952 | 0.642 |

P=256 modestly improves retention and variance (DELFOR-01b 0.63→0.78, INS-01b 0.87→1.00, SUBW-01b
0.91→1.00) at 2× compute — a nice-to-have. **It does NOT break the residual E ceiling**: DELTO-02a still
keeps "served the soup the customers" (p=0.89), DEL-the still floats "We *want* to store", INS-01a still
keeps "handed handed". So the dative under-edits are **LM-gain-limited, not particle-limited** — confirming
they are a signal ceiling, not an inference one. P=128 is the cost-effective default; P=256 is an optional
+0.05 combined if compute allows.

## Robustness — finalist K=−4.5, α=(200,2,2), full 40-item battery, 3 seeds

| seed | E_edit | L_keep | combined |
|------|--------|--------|----------|
| 0 | 0.323 | 0.946 | 0.634 |
| 1 | 0.270 | 0.903 | 0.587 |
| 2 | 0.280 | 0.932 | 0.606 |
| **mean** | **0.291** | **0.927** | **0.609** |

Stable (combined σ≈0.02). A few bimodal items still flip across seeds (SUBW-02a real-word sub: 0.98/0.00/1.00;
DELFOR-01b keep), but the aggregate is solid.

## Residual ceiling (what these levers do NOT move)

E is capped ~0.29 by ~8 items that **under-edit on 70m** and do not respond to K or alpha:
- **Dative / PP function-word restorations**: DELTO "to", DELFOR "for", DELFROM "from", INS-to — 70m does
  not prefer the restored sentence enough (these are the items 410m *did* start to fix → an LM-gain
  ceiling, not an inference one).
- **Duplicate-word removal** (INS-01a "handed handed"): rejuvenation can substitute/insert but **cannot
  delete a word** (a known structural gap), so the duplicate is never removed.

These are the right targets for further work, not parameter tuning.

## Recommendation

1. **Adopt K=−4.5, α=(200,2,2) as the align default** (done in `pythia_word_caprop.py`, uncommitted).
   Robust ~2× E and +0.16 L over the old align default, junk→~0, no compute cost (70m, P=128).
2. **Stay on pythia-70m** for this objective — 410m is a wash (leading-list artifact + slower).
3. Further E gains require non-parameter work: (a) a word-**deletion** rejuvenation move (unblocks
   duplicate removal), (b) raising LM-gain for dative function-word restorations — either a stronger LM
   *with* the leading-insertion artifact handled, or a frequency/position-aware del cost for dropped
   high-frequency function words.

## Artifacts (all under `planning/`)
- subset + tooling: `align_opt_subset.txt` (12 bimodal items), `align_opt_summ.py` (E/L/junk parser),
  sweep scripts `align_opt_{ksweep,alpha_sweep,validate,410m,full40,full40_alpha}.sh`
- K-sweep: `align_opt_K{4.5,5.5,8.0}_gibbs.txt`; alpha: `align_opt_a{0.5,2,4}_gibbs.txt`
- seed/robustness: `align_opt_a4_s{1,2}.txt`, `align_opt_a{1,8}_s*.txt`
- 410m: `align_opt_410m_K-4.5_a200,4,4.txt`
- full-40: `align_opt_full40_K4.5_a{1,2,4}.txt`, `align_opt_full40_K4.5_a2_s{1,2}.txt`
- baseline (old default, full-40): `align_subsample_K26_gibbs.txt` (Phase-4 run)
