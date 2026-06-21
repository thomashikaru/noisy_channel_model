# Align channel — Phase 4 results + Phase 5 recommendation

**Status:** Phases 1–4 complete on branch `align-action-channel`. Phase 5 = decision, **awaiting user
approval.** No defaults or docs changed. Recommendation below: **do NOT promote align as the default**;
keep it as the gated, reversible opt-in it already is.

## What works (the mechanism is validated)

- **Correctness:** 31/31 gates pass; `word_action`/`char_copy` byte-identical. align is certified by a
  reduction gate (align at `(p_align,p_ins,p_del)` ≡ word_action at `(p_copy=p_align, p_sub=p_align, …)`)
  plus align-path sweep gates at a distinct K.
- **The garage class flips.** `channel="align"` on "The garage needs to be tossed out." infers
  **"The garbage needs to be tossed out."** (p=0.82); word_action hallucinates "The garage door…" (p=1.00).
- **Curated sub-vs-indel family (8 items, P=128, rejuv=gibbs):** align **8/8**, word_action **5/8**, with
  **zero KEEP over-edit regression** on those curated items. align flips GARAGE + DESERT (word_action
  over-edited them into spurious insertions); both keep the 3 over-edit guards.

## What breaks (the realistic-battery regression)

40-item calibration subsample, rejuv=gibbs, P=128 (E = should-edit correction mass; L = should-keep
retention; junk = mass on neither clean reading = over-edit / spurious-insertion):

| config                         | E     | L     | junk (>0.5) |
|--------------------------------|-------|-------|-------------|
| word_action α=200 (baseline)   | 0.213 | 0.992 | 0 / 40      |
| align K=log(1/26)              | 0.198 | 0.764 | 9 / 40      |
| align K=log(1/40)              | 0.197 | 0.723 | 10 / 40     |

align **over-edits the realistic battery** — L 0.99→0.76, junk 0→9 — including destroying clean
sentences (e.g. "the boy handed the pencil to the girl" → junk=1.00). On **pure** substitution items
(SUBW/SUBN), word_action is actually *better*: E≈0.89 vs align 0.75, L=1.00 vs 0.94, junk 0 vs 1–2/8.

## K cannot thread the needle — and why it is NOT an expressiveness gap

- **garage flip survives only in a narrow K-band at ≈log(1/26).** Sharpening to log(1/40) loses the flip
  (literal p=0.96) and does NOT recover retention (L 0.76→0.72). The SUB-item E/L is flat across
  K∈{1/15…1/60}. So in the garage-surviving K-window, retention is already ~0.76.
- This is **not** because align (one knob K) is weaker than word_action (p_sub + form). At a single
  d=1 edit they are **equally expressive**: align with K=−8.56 reproduces word_action's exact threshold
  (reduction gate). word_action **cannot thread it either** — tuned to fix garage (lower its sub
  penalty), it would over-edit the battery identically; it "keeps" the battery only by sitting at a
  threshold so high (|Δ|=8.56) it refuses garage **and** ~80% of real corrections (E=0.21).
- **The real wall:** under pythia-70m, genuine corrections (garage gain ≈4.76 nats) and the battery's
  spurious over-edits have **overlapping LM-gains**. A per-edit cost only thresholds LM-gain, so no
  threshold — in either channel — admits garage while rejecting the junk. The garage class is narrow:
  a typo whose literal is *itself a valid word* with a higher-frequency real neighbour; the battery's
  SUB typos have non-word literals (huge gain) that word_action already corrects.
- (Minor genuine difference: align and word_action diverge only at d≥2 — align's linear `K·d` makes
  multi-char edits cheaper. Not the garage mechanism.)

## Recommendation

**Do not promote align as the deployment default.** It fixes a narrow, real class (real-word→real-word
typos word_action misses) but regresses the calibrated battery operating point, and that regression is
not tunable away with K. align's value is design-level: it makes the substitution threshold a single,
interpretable, surface-driven knob decoupled from the action-rate prior — but it does not by itself
solve over-editing, because over-editing here is a **signal-quality problem, not a parameterization one.**

Keep align as the gated, reversible opt-in it already is (`--channel align`). Promising next experiments,
all out of this plan's scope (would need their own branch + approval):
1. **`lm_temp < 1` with align** — flatten pythia-70m's noisy battery preferences so good corrections
   out-gain spurious ones; may keep the garage fix while curbing battery over-editing. (Most promising.)
2. **align on a stronger LM** (pythia-410m+) — the battery junk is partly the 70m LM's bad long-sentence
   preferences.
3. **A convex / frequency-aware distance cost** — make d=1 cheap (garage) while keeping far/again
   spurious edits expensive (a second knob the flat K lacks).

## Artifacts
- gates: `src/genjax_port/tests/test_pairhmm_exact.py` (7 align gates)
- family + behavioural run: `src/genjax_port/align_sub_indel_check.py`, `planning/align_sub_indel_out.txt`
- battery + sweep: `planning/align_subsample_K{26,40}_gibbs.txt`, `planning/align_sweep_K{15,26,40,60}_gibbs.txt`
- drivers: `planning/align_phase4_sweep.sh` (superseded), `planning/align_phase4b_sweep.sh`
