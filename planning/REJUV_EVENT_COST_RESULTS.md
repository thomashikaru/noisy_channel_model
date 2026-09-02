# Rejuvenation cost/activity split — probe results (2026-09-02)

*Probe: `planning/rejuv_event_cost_probe.py` → `rejuv_event_cost_probe.results.jsonl`.
Smoke set (8 items), one seed, harness worker's `_run_one`, main.env config (align, P=64,
band 2, lb 6, LOOKAHEAD=1, LA_PROPOSAL=1), three arms differing ONLY in `rejuv`:
off / gibbs / gibbs+bd. Targeted-rejuv step 2 (after the step-1 signal probe,
`planning/rejuv_gating_probe/FINDINGS.md`).*

## Code fact this probe confirms empirically

The deployed `bd_mode="gibbs"` indel move fires **once, post-loop, on the finished cloud**
(`pairhmm_smc.py` ~line 1029) — only the substitution sweep repeats per resample event.
(The per-event firing branch belongs to the failed `mh`/`smcp3` modes.)

## Results

| item | T_off | T_gibbs | T_bd | sub cost | indel cost | indel p_noop | chosen ins/del |
|---|---|---|---|---|---|---|---|
| 0 candle           | 19 | 16 | 54  | −2 | +38  | 0.003 | 0/64 |
| 1 candle-PO        | 16 | 13 | 62  | −3 | +48  | 0.991 | 0/2 |
| 2 Medics           | 26 | 25 | 99  | −1 | +73  | 0.996 | 1/28 |
| 3 gifts-agreement  | 18 | 15 | 100 | −3 | +85  | 0.998 | 0/1 |
| 4 NPZ-suspect      | 33 | 39 | 209 | +6 | +170 | 0.805 | 5/1 |
| 5 licked-ball      | 14 | 34 | 107 | +19| +73  | 0.967 | 1/1 |
| 6 coach-frisbee    | 24 | 27 | 185 | +4 | +158 | 0.967 | 5/1 |
| 7 candle+context   | 15 | 19 | 99  | +4 | +80  | 0.865 | 0/9 |

**Totals: off 166 s, gibbs 188 s, gibbs+bd 914 s. Of the rejuvenation overhead, the one-shot
indel grid is 97% (725 s) and all per-event substitution sweeps together are 3% (23 s).**
bd/off = 5.5× here (local, 1 seed, post-fix) vs 12.7× on the pre-lap 4-seed cluster smoke —
same order, and the Phase-4 sizing (SPI=1320) retains its margin.

## Conclusions

1. **There is no scheduling lever.** The component that repeats (sub sweep) is already
   essentially free; the component that is expensive (indel grid) already fires exactly once.
   "Rejuvenate less often" cannot save anything material.
2. **The indel move's cost on clean items is pure confirmation.** On items 1–3 the full
   conditional puts ≥99% mass on no-op: the grid is paid to certify that no edit helps. That
   is the price of exactness — and the step-1 signal probe showed no reliable observation-side
   item skip exists (clean experiment items spike like edited ones).
3. **Where it acts, it acts sensibly.** Item 0: the off cloud's junk sub-edits ("candle in
   the daughter") are cleaned back to the literal (p_literal → 1.0) — the known bimodal
   near-tie consolidating. Item 7 (context prime): the "to"-repair stays MAP in all arms.
4. **Post-fix, bd's logZ edge on smoke is small** (max +0.42; several ≈0/negative vs off,
   1 seed) — the off arm's inference fixes closed most of the June-era gap on THESE items.
   bd's remaining distinctive value is the band-2 structural class (chen voice, tabor/huang
   relativizer+comma), which smoke does not contain; that is what main_bd measures.
5. **Therefore the remaining options are exactly two:** shrink the grid itself (the parked
   targeted-allocation policy — real complexity, moderate savings: step-1 measured 1.2–2.8×
   on the insertion grid at 70–96% site recall), or run main_bd as-is. Given the one-time
   nature of the run, the modest measured multiplier, and the interpretive value of the exact
   kernel ("arms differ ONLY in REJUV"), **run main_bd as-is**.
