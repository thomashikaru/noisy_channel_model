# Targeted-rejuvenation gating signal — probe findings (2026-09-02)

*Probe: `planning/rejuv_gating_signal_probe.py` (compute → `signal_<ds>.csv`, analyze →
`SUMMARY.md` + `ops_eval.csv`). Signal = per-unit contextual LM surprisal (the harness's own
`lm_word_surprisals`, context-as-prime) minus `unigram_surprisal` — the Gen.jl
conditional-rejuvenation signal, computed purely from the OBSERVED sentence (kernel-safe).
Repair sites = difflib ops from `model_input` vs `repairs.csv` `intended_text` (evaluation-only).
All 2,329 experiment inputs; 1,591 ops / 1,498 repairs / 1,258 edited stimuli. Total LM cost:
~12 min on the Mac, compile-dominated.*

## Conclusions

1. **An absolute threshold on relative surprisal does NOT work as the gate.** Recall collapses
   before compression gets interesting (τ=0/±1: 84% op recall but 62% of positions kept;
   τ=2/±1: 43% recall). And at the item level the signal does not separate edit-needing from
   clean items (τ=3 fires on 75% vs 67%) — **no whole-item skip from this signal**. The
   experiment's "clean" conditions are garden-pathy/marked by design, so they spike too.

2. **Rank-based (top-k per item) gating localizes INSERTIONS — the expensive grid dimension —
   very well.** Insert-op site recall, gating gaps within ±w units of the item's top-k
   relative-surprisal units:

   | k | ±0 | ±1 | ±2 |
   |---|-----|-----|-----|
   | 2 | 0.41 | 0.55 | 0.78 |
   | 3 | 0.57 | 0.81 | 0.96 |
   | 4 | 0.70 | **0.94** | 0.99 |
   | 6 | 0.92 | 1.00 | 1.00 |

   Per-dataset at k=4/±1: chen2023 voice 0.99, gibson2013 dative 0.97, tabor2004 relativizer
   0.95, huang2024 comma 0.73 — huang's evidence (the disambiguating verb) sits ~2 units
   DOWNSTREAM of the gap; at ±2 huang is 1.00. An asymmetric window (gap allowed to sit a few
   units upstream of a spike) would buy huang's coverage without paying both directions.
   This is exactly bd's remaining job post-`lookahead_proposal`: the band-2 structural
   insertions at disambiguation points.

3. **Deletions localize poorly** (0.38 at τ=0/±1; 0.58 at k=4/±1) — a spurious word is usually
   locally unsurprising; the implausibility is diffuse. But deletions are the CHEAP grid column
   (no ×Kc candidate factor), so the right move is to **keep the deletion column exact and gate
   only the insertion gaps**.

4. **Substitution sites localize well** (0.92 at τ=0/±1, 0.93 at k=4/±1), so the fixed 6-word
   sub-sweep window could also be replaced/augmented by top-k targeting — a smaller but free win.

5. **The single biggest spike is usually DOWNSTREAM of the repair site** (+3 or beyond for
   1,063/1,591 ops — end-of-sentence and disambiguation effects). Argmax-only policies
   mislocalize; a small top-k set with a ±1–2 window is the right shape.

## Honest compression accounting

Sentences are short (mean 12.5 units incl. EOS), so position-gating alone is a moderate cut on
the insertion grid, not an order of magnitude: positions kept ≈ 0.36 (k=4/±0), 0.54 (k=3/±1),
0.67 (k=4/±1), 0.86 (k=4/±2) → **1.2–2.8× on the dominant Wmax×Kc term at 70–96% insert-site
recall**. Bigger total savings need the OTHER factor — how often the full sweep runs (once per
ESS-resample event today) — which this static probe cannot measure. Next: measure from
existing bd outputs where accepted indel moves actually land (rejuv stats are in the words
block), and how many sweep events per item re-score an unchanged conclusion.

## Recommended policy shape (for the implementation step)

Per sweep event, choose the insertion-gap subset by an observation-derived rule: gaps within
the (asymmetric) window of the item's top-k relative-surprisal units, PLUS with some
probability ε use the full grid instead (or cycle a random extra gap block). Every component
kernel is a valid Gibbs move on a fixed sub-grid chosen independently of particle state, so the
mixture leaves the target invariant and full support is retained across events while the
expected per-event cost drops. Validation per the standing rule: A/B against the exact full
sweep on a battery subset + a dataset-level check on chen/tabor/huang items (the battery is
band-1-safe and cannot test the structural class).
