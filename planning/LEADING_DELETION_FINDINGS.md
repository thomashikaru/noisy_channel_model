# The forward filter loses the plain-literal parse under the deployed align channel

**Date:** 2026-08-31 (found during the Phase-4 cluster smoke, before any Phase-5 submission).
**Status: the full Phase-5 runs are ON HOLD pending a decision** (see the end).
**Probes:** `planning/leading_del_probe/*.py` (run from the repo root in the `ncgenjax` env).

## The contradiction that started it

For the smoke item "The mother gave the candle the daughter." (align, P=64, band=2, rejuv=off,
4 seeds), the record says two things at once: the posterior is 84% "the sentence is fine as
written" (`p_literal`, and every printed hypothesis is the plain sentence) — and about two
intended words were deleted before "The" (`del_before(unit 0) = 1.99`). Both cannot be true.

Inspecting the particles resolves it: every surviving particle's intended sentence begins with
two NEWLINE tokens —

```
"\n" "\n" "The" mother gave the candle the daughter .     (token ids 187, 187, 510, ...)
```

— marked as words the writer deleted. The decoder strips whitespace when printing, so these
particles PRINT as the plain sentence. `map`/`hypotheses`/`p_literal` therefore hide the
artifact; the per-word alignment posteriors expose it. 5 of the 8 smoke items carry it
(unit-0 `del_before` = 1.99 / 0.32 / 1.72 / 0 / 0 / 1.94 / 2.0 / 0).

## This is an inference failure, not the model's preference

Scoring both intended sentences under the model directly (probe `lm_paths.py`, pythia-70m,
prime "<|endoftext|>."):

- The LM gives the newline version LOWER probability: −47.64 vs −46.90 (0.74 nats against it).
- The channel charges it two deletions: ~−7.9 nats (per-particle `wdel_p = −3.81`, probe
  `particles.py`).
- So the model's joint prefers the plain sentence by ≈ 8.4 nats — posterior odds ≈ 4,000 : 1.

Correct inference must return the plain sentence with nearly all the mass. Instead it ends with
weight EXACTLY 0 — at P = 16, 64, and 256, under every random key tried (probes `dedup_ab.py`,
`mech_test.py`). Two corroborating checks:

- Evidence: band=1 gives logZ −53.13; band=2 gives −63.58 for the SAME channel. Band 2 allows
  strictly more alignments, so its true evidence cannot be lower; a 10.5-nat drop means the
  sampler lost the dominant hypothesis. (gibbs+bd on the same item: −49.5.)
- The dedup optimization is innocent: on/off is bit-identical in all six A/B runs.

## Mechanism: particles that fall behind on the observations look better mid-run

The SMC extends particles one INTENDED word per step and resamples on the weights at that
step — but particles at the same step may have consumed different numbers of OBSERVED words:

- A particle that posits "a word was deleted here" adds an intended word without consuming an
  observed word. It pays the deletion penalty plus that word's LM cost.
- A particle that takes the sentence as written consumes one observed word per step and pays
  its LM cost immediately ("mother" alone costs 8.6 nats here).

Mid-sentence, a particle that has fallen 2 observed words behind (the most band=2 allows) holds
a weight that has not yet paid for two observed words — worth roughly their LM cost, 10–15
nats — while an up-to-date particle has paid in full. At every resampling step the up-to-date
particles therefore look worse, and within two or three steps they are eliminated. The books
balance only at the terminal step, and by then the correct hypothesis is extinct. More
particles do not help: the ~e^10 mid-run weight gap applies at every resample regardless of P.
The trace shows it live (probe `trace_steps.py`): after two steps, 45% of the mass has consumed
ZERO observed words.

Why align and not char_copy: char_copy charges a flat −9 per deletion, so falling 2 words
behind costs ~18 nats — more than it buys. Align's Dirichlet prior α=(200,2,2) prices a
deletion at ~−4.6 nats, so falling behind is a net win in the mid-run weights.

Direct tests (probe `mech_test.py`; same item, worker key unless noted):

| configuration                                | w(plain literal) | del_before(The) | logZ |
|----------------------------------------------|-----------------|-----------------|--------|
| align, α=(200,2,2), band=2  (DEPLOYED)        | 0.000 (all runs) | 2.00           | −63.58 |
| align, α=(200,2,0.02) ⇒ p_del ≈ 1e-4          | 0.86 / 0.97 (2 keys) | 0.00       | −53.13 |
| align, band=1                                 | 1.000            | 0.00           | −53.13 |
| align, band=0                                 | 0.988            | 0.00           | −53.13 |
| char_copy (flat WDEL = −9)                    | 0.985 / 1.000 (2 keys) | 0.00     | −56.10¹ |
| align band=2, rejuv=gibbs+bd (cluster smoke)  | p_literal 0.48–0.63 | 0.00        | −49.5  |

¹ different channel ⇒ different Z; listed for the survival column, not the logZ comparison.

Everything fits: expensive deletions price the head start out; band=1 halves the allowed lag;
and the Gibbs indel move REPAIRS the artifact after the fact (its full conditional deletes the
"\n" units: `n_chosen_del` 3–7 per seed on item 0, `del_before` → 0).

## What this corrects

- "Leading junk is an LM-prior/prime artifact" (earlier sessions): for this class the LM
  actively disfavors the junk path — the artifact is made by the sampler's mid-run weights,
  not by the LM prior.
- The align channel's battery over-editing was attributed to SIGNAL (overlapping LM gains).
  At least the leading-deletion component is INFERENCE, and it does not heal with more
  particles. Some part of gibbs+bd's +15/87 on the battery is likely "the deletion move
  repairing the forward pass's own artifact".

## Consequences for the experiment if run as-is (rejuv=off arm)

- `surprisal_nc` — the experiment's main output — is computed under a posterior whose intended
  sentence starts with a phantom "\n\n". That changes the LM context for EVERY word, not just
  the first, so the contamination is not confined to unit 0.
- `p_literal` and the printed hypotheses overstate literalness (whitespace stripped from view).
- logZ is underestimated by ~10 nats on affected items, and the evidence-weighted seed merge
  then weights seeds by how badly each one failed.
- The off-vs-bd contrast largely measures the repair of this artifact rather than
  "rejuvenation vs none" on equal footing.

## The decision (user's call — none of these is mine to make)

1. **Run Phase 5 as configured**: measures the deployed system exactly as calibrated (the
   battery and calibration history carry the same artifact). Cheapest; caveats above apply.
2. **Fix the resampling weights first**: charge each particle an LM-based estimate for the
   observed words it has not yet consumed; the charge cancels at the terminal step, so logZ
   stays unbiased. An inference change — needs its own validation (toy exactness + battery A/B).
3. **Shrink the deletion allowance** (band=1 and/or a smaller α_del): removes the artifact in
   these probes but changes the model mid-experiment (band=1 still reaches every single-word
   repair in the stimuli; band was 2 to keep everything constant with the benchmark).
4. **Run only the gibbs+bd arm** (it self-repairs here) and drop or postpone the off arm.

Sizing facts for whichever run happens: main_off MEM should be 24G (cluster off-arm peak
12.97 GB on a 13-word item; stimuli go to 19 words); main_bd 32G held for the smoke
(completed shards 10.7–12.7 GB so far).
