# The forward filter loses the plain-literal parse under the deployed align channel

**Date:** 2026-08-31 (found during the Phase-4 cluster smoke, before any Phase-5 submission).
**Status: Phase-5 fan-out is ON HOLD pending a decision** (see "The decision" at the end).
**Probes:** `planning/leading_del_probe/*.py` (run from the repo root in the `ncgenjax` env).

## What was observed

The first cluster smoke records (align, P=64, band=2, rejuv=off, 4 seeds, item
"The mother gave the candle the daughter.") report, in the per-word block:
`del_before("The") = 1.99` and `p_err_positional ≈ 1` at almost every unit — while
`p_literal = 0.84` and every top hypothesis reads as the literal sentence. Both are faithful:
the final cloud's dominant particles are the intended sequence

```
"\n" "\n" "The" mother gave the candle the daughter .     (token ids 187, 187, 510, ...)
```

i.e. the literal sentence PLUS two deleted intended units that are newline tokens. The decoder's
`.strip()` removes them from every reported string, so `map`/`hypotheses`/`p_literal` silently
absorb the artifact; the alignment posteriors see it. 5 of the 8 smoke items carry it
(unit-0 `del_before` = 1.99 / 0.32 / 1.72 / 0 / 0 / 1.94 / 2.0 / 0).

## The model does NOT prefer that parse — this is an inference failure

Scoring both paths directly (probe `lm_paths.py`, pythia-70m, prime "<|endoftext|>."):

- LM: literal −46.90, newline path −47.64 → the LM DISFAVORS the newline path by 0.74 nats.
- Channel (from the run's own `diag`, probe `particles.py`): the newline particle pays
  ~−7.9 nats (two deletions at `wdel_p = −3.81` under the align channel's Dirichlet θ).
- So the model's joint prefers the plain literal by ≈ 8.4 nats (posterior odds ≈ 4,000 : 1) —
  yet the plain-literal parse has **weight exactly 0** in the final cloud: **P = 16, 64, 256 ×
  keys fold_in(PRNGKey(0|1|2), 0), every run** (probes `dedup_ab.py`, `mech_test.py`).
- `dedup` is exonerated: on/off is bit-identical (same logZ, same cloud) in all six A/B runs.
- logZ tells the same story: −63.58 (band=2) vs −53.13 (band=1). Band 2's path space strictly
  contains band 1's, so its true log Z is ≥; an estimate 10.5 nats LOWER means the band-2
  forward pass misses the dominant mode. (gibbs+bd on the same item: −49.5.)

## Mechanism: deletion-priced loitering in the intermediate targets

The SMC is intended-word-synchronous: at step k, particles are compared after k intended words,
however many OBSERVED words each has consumed. A particle that "deletes" (LM emits a word, no
observation consumed) pays LM + log p_del but banks no observation cost; a punctual particle
pays the observed word's full LM surprisal immediately ("mother" costs −8.6 here). With the
align channel's deletion at `log p_del ≈ −4.6` (α = (200,2,2) ⇒ p_del ≈ 1%), loitering is
cheap, and the band (2) lets a particle stay 2 observations behind for the whole sentence —
a persistent head start roughly equal to the LM cost of 2 words (~10–15 nats) at every
resampling event. The trace (probe `trace_steps.py`) shows it live: after 2 steps, 45% of the
mass has consumed ZERO observed words; the punctual literal lineage is resampled away by step
3; the loiterers pay their debt only at the terminal correction, when the literal is long gone.

Direct tests of the mechanism (all on the same item/key unless noted; probe `mech_test.py`):

| configuration                                | w(plain literal) | del_before(The) | logZ |
|----------------------------------------------|-----------------|-----------------|--------|
| align, α=(200,2,2), band=2  (DEPLOYED)        | 0.000 (all runs) | 2.00           | −63.58 |
| align, α=(200,2,0.02) ⇒ p_del ≈ 1e-4          | 0.86 / 0.97 (2 keys) | 0.00       | −53.13 |
| align, band=1                                 | 1.000            | 0.00           | −53.13 |
| align, band=0                                 | 0.988            | 0.00           | −53.13 |
| char_copy (flat WDEL = −9)                    | 0.985 / 1.000 (2 keys) | 0.00     | −56.10¹ |
| align band=2, rejuv=gibbs+bd (cluster smoke)  | p_literal 0.48–0.63 | 0.00        | −49.5  |

¹ different channel ⇒ different Z; listed for the survival column, not the logZ comparison.

Everything fits: expensive deletions (char_copy −9, or tiny α_del) price the head start out;
band=1 halves the allowed lag; the Gibbs indel move REPAIRS the artifact post hoc (its full
conditional deletes the "\n" units: `n_chosen_del` 3–7 per seed on item 0, del_before → 0).

## What this corrects

- "Leading junk is an LM-prior/prime artifact" (earlier sessions; the P=16 '"- "' junk, the
  no-structural-edit-bans discussion): for this class the LM actively disfavors the junk path —
  the artifact is made by the SMC's intermediate target, not by the LM prior.
- The align channel's battery over-editing was attributed to SIGNAL (overlapping LM gains).
  At least the leading-deletion component is INFERENCE, and it does not heal with more
  particles (P=256 identical). Some part of gibbs+bd's +15/87 on the battery is likely
  "the indel move repairing the forward pass's own artifact".
- More particles cannot fix it: the weight deficit during the lag (~e^10) dwarfs any
  practical P; the mode dies at the first or second resampling event regardless.

## Consequences for the experiment if run as-is (rejuv=off arm)

- `surprisal_nc` for every word is computed under a cloud whose LM context is shifted by the
  phantom "\n\n" (and by whatever mid-sentence loitering produced); the S_nc-vs-S_lm
  comparison then mixes the artifact into the quantity of interest.
- `p_literal` and the MAP strings overstate literalness (the junk is stripped from view).
- logZ (the evidence) is underestimated by ~10 nats on affected items; the evidence-weighted
  seed merge then weights seeds by how badly each failed.
- The off-vs-bd contrast largely measures the repair of this artifact rather than
  "rejuvenation vs none" on equal footing.

## The decision (user's call — none of these is mine to make)

1. **Run Phase 5 as configured anyway**: it measures the deployed system exactly as calibrated
   (battery + calibration history carry the same artifact). Cheapest; interpretation caveats above.
2. **Fix the intermediate target first** (e.g. charge each particle, at every resampling
   event, an estimated LM cost of its unconsumed observed words — a lookahead/twist that
   cancels at the terminal step and leaves log Z unbiased), then re-smoke and fan out.
   An inference change, needs its own validation gate (toy exactness + battery A/B).
3. **Change the operating point** (band=1, or a smaller α_del, or both): removes the artifact
   in these probes but changes the model mid-experiment and shrinks the reachable edit space
   (band=1 still reaches every single-word repair in the stimuli; band was 2 "to keep
   everything else constant" with the benchmark).
4. **Run only the gibbs+bd arm** (it self-repairs here) and drop or postpone the off arm.

Sizing facts for whichever run happens: main_off MEM should be 24G (cluster off-arm peak
12.97 GB on a 13-word item; stimuli go to 19 words); main_bd 32G held for the smoke
(completed shards 10.7–12.7 GB so far).
