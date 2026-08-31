# Plan: lookahead charge at resampling (fix for the leading-deletion inference failure)

**Written 2026-08-31 by the session that found the bug; to be EXECUTED in a fresh session.**
**Read first:** `planning/LEADING_DELETION_FINDINGS.md` (what the bug is and the evidence),
probes in `planning/leading_del_probe/`. User decision on record: fix the resampling weights
BEFORE any Phase-5 submission (option 2 of the findings doc).

## The change in one paragraph

Mid-run, particles at the same intended-word step may have consumed different numbers of
observed words, and a particle that is behind holds a weight that has not yet paid for the
observed words it skipped. Fix: at each resampling event, weight each particle's resampling
probability by an estimate of that unpaid cost — the plain-LM baseline surprisal of the
observed units past its position — then carry the inverse of that factor as residual weight so
every estimator stays unbiased (the standard auxiliary-particle-filter arrangement). The
estimate comes from `lm_word_surprisals`, which the worker already computes per item.

## Exact math (log domain)

Current resample block (`pairhmm_smc.run`, the `if ess_pre < 0.5 * P:` block, ~line 845):

```python
ess_pre = float(_ess(log_w))
if ess_pre < 0.5 * P:
    logZ = logZ + logsumexp(log_w) - jnp.log(P)
    anc  = jax.random.categorical(sub, log_w, shape=(P,))
    ...gather state/costs by anc...
    log_w = jnp.zeros(P)
```

New behavior when a charge vector is supplied (`lookahead_lp is not None`):

```python
psi   = twist(state)                        # (P,), see below; 0.0 where not finite
l     = log_w + psi
ess_pre = float(_ess(l))                    # trigger on what we would sample
if ess_pre < 0.5 * P:
    logZ  = logZ + logsumexp(l) - jnp.log(P)
    anc   = jax.random.categorical(sub, l, shape=(P,))
    ...gather state/costs by anc (unchanged)...
    log_w = -psi[anc]                       # residual: undo the charge after selection
```

With `lookahead_lp=None`, psi ≡ 0 and this is EXACTLY the current code — the off gate is
bit-identity by construction. Unbiasedness: selection ∝ w·e^psi with residual w/e^psi is the
auxiliary particle filter; every later fold (including the terminal one) uses proper weights.
No change to the terminal-correction block.

The twist, per particle (state[5] = `log_alpha`, shape (P, M+1), −inf outside the band;
`log_alpha[p, k]` = log P(intended prefix, exactly k observed units consumed)):

```
C[k]   = sum over i in (k, M] of lm_temp * loglik_unit[i]          # suffix sums, C[M] = 0
psi[p] = logsumexp_k(log_alpha[p, k] + C[k]) - logsumexp_k(log_alpha[p, k])
```

`loglik_unit[i] = -surprisal_lm[i-1]` from `pythia_word_caprop.lm_word_surprisals(text, prime)`
(per observed UNIT over its whole token span — same indexing as log_alpha's k). Notes:
- C[k] ≤ 0: it is a charge; up-to-date particles (mass at k = M) get ~0, laggards get the
  LM cost of what they skipped. Particles AHEAD via spurious insertions are charged less —
  symmetric, as intended.
- EOS term (`surprisal_end_lm`): EXCLUDE. It follows all units, so it would add the same
  constant to every C[k] and cancel in the normalized resampling weights.
- Guard: where logsumexp(log_alpha) is −inf/NaN (impossible parse), set psi = 0 (that particle
  already has −inf weight; mirror the NaN guard used in the terminal correction).
- Done particles need no special case: their alpha mass sits at k = M, so psi ≈ 0.

## Code touch-points

1. `src/genjax_port/pairhmm_smc.py :: run` — new kwarg `lookahead_lp=None` (an (M,) array of
   per-unit log-liks, or None). Precompute the suffix vector C once before the step loop;
   apply the block above at the ONE resample site. Nothing else in the loop changes: the
   rejuvenation moves run after resampling exactly as now (the residual −psi[anc] is a
   per-particle constant in log_w; the moves fold their own weights on top, which is correct).
2. `src/genjax_port/pythia_word_caprop.py :: run` — new kwarg `lookahead=False`; when True,
   call `lm_word_surprisals(observed, prime=prime)` and pass
   `lookahead_lp = -surprisal_lm * 1.0` down (temper by `lm_temp` inside pairhmm_smc where C
   is built, or here — one place, documented).
3. `slurm/run_nc_batch.py` — `--lookahead` flag (default off). Compute the baseline ONCE per
   item BEFORE `pwc.run` and reuse it for the words block (`_words_block` currently calls
   `lm_word_surprisals` itself — accept the precomputed dict to avoid a second LM forward).
   Add the flag to `config_slug(a)` (~line 82; only when ON, mirroring how other optional
   flags are encoded so existing result dirs keep their names) and to `_resolved_config`.
4. `slurm/submit_nc_batch.sh` — `LOOKAHEAD` env var (default 0) → `--lookahead` in EXTRA.
5. `experiments/run.sh` — add `LOOKAHEAD` to `CFG_VARS`; set `LOOKAHEAD=1` in
   `experiments/configs/main.env` (both arms inherit it, so `check_arm_parity` stays satisfied)
   ONLY AFTER the gates pass and the user confirms the operating point.

## Decisions already made (do not re-litigate without new evidence)

- ESS trigger on the TWISTED weights `l` (that is the distribution being sampled).
- Default OFF everywhere; `lookahead_lp=None` must be bit-identical (it is, by construction).
  Whether Phase 5 runs with it ON is the user's explicit call after the gates.
- The charge uses the literal-context baseline. Any positive twist preserves unbiasedness;
  quality only affects variance — do not over-engineer a context-conditional estimate.
- rejuv=off / gibbs / gibbs+bd all get the same twist (it lives at the resample site).

## Known risk to verify, not assume

The §3.1 prefix-mass accumulator (`word_stats`, `ws_acc.add(em, ..., log_w, logZ)` PRE-resample;
identity `sum(S_k) + S_end == −logZ` exact on disk) assumes the (logZ, log_w) bookkeeping it
mirrors. The twist changes the fold (`logsumexp(l)`) and the residual (−psi). Re-derive the
telescoping with the residual in place; if the identity breaks, the accumulator needs the same
psi bookkeeping (it receives log_w and logZ already — the fix is local). The on-disk equality
gate will catch any error exactly.

## Gates (all must pass before any cluster submission)

1. Bit-identity: `lookahead` off ⇒ byte-identical records on the smoke set (trivial but run it).
2. Toy exactness: extend the existing exact-enumeration toy tests with a lookahead-on case —
   posterior matches enumeration within the existing tolerances and logZ is unbiased over seeds.
3. The probe flips: `planning/leading_del_probe/mech_test.py` + a lookahead arm ⇒ on
   "The mother gave the candle the daughter." at align/band=2/P=64 expect
   w(plain literal) ≥ ~0.9 (from 0.000) and logZ ≈ −53 (from −63.58). Run ≥ 3 keys.
4. §3.1 identity on disk: `sum(S)+S_end == −logZ` exact after a smoke-local run with lookahead ON.
5. Full suite (113 tests) green: `conda run -n ncgenjax python -m pytest -q src/genjax_port/tests/ experiments/tests/`.
6. Smoke set A/B (local, cheap): unit-0 `del_before` should drop to ~0 on the 5 affected items;
   report per-item p_literal/logZ shifts to the user BEFORE any battery run.
7. Battery A/B (87 items, seeds 0+1, P=64, like the morph regression run): OPTIONAL and the
   user's call — it is the arbiter for "did anything else move".

## Suggested phases (commit each)

- P1: pairhmm_smc twist + threading (`lookahead_lp`), toy gate, bit-identity gate.
- P2: probe extension + the 3-key flip result recorded in the plan/RUNLOG.
- P3: worker + submit script + run.sh plumbing, smoke-local with lookahead ON, §3.1 gate,
  full suite. Report gate results to the user; ask about the battery A/B and about turning
  LOOKAHEAD on in main.env.
- P4 (after user OK): push, sync cluster, re-run the cluster smoke (both arms — new config
  dirs, nothing overwritten), then resume the Phase-5 runbook in the harness memo
  (main_off with MEM=24G, then main_bd shortest-first at 32G).

## Execution results (2026-08-31, the executing session)

**One correction to the design above.** "Done particles need no special case: their alpha mass
sits at k = M" is wrong: the kernel leaves `log_alpha` UNCHANGED when a particle chooses EOS
(`jnp.where(advance, new_alpha, log_alpha)`), so a done particle's stored row is the stale
pre-EOS one, spread over k < M. Under caprop the done particle already folded
`alpha[M] − logsumexp(alpha)` into its EOS score, so its unpaid cost is exactly 0 — the twist
therefore forces psi = 0 on done particles (still unbiased; without the guard the up-to-date
literal parses would be mis-charged, partially recreating the artifact). Consequence:
`lookahead_lp` requires `proposal="caprop"` (bootstrap's done particles have NOT paid the
terminal term, so psi=0 would be wrong there); enforced with a ValueError.

**Gate results.**

1. Bit-identity (off): PASS — toy 9-case capture (peaked/band2/align, 3 seeds each, full log_w)
   and the Pythia probe item (align/off/P=64, worker key, logZ −63.575538635253906) are
   byte-identical before/after the change. Plus a stronger form: an all-ZERO charge vector
   (psi ≡ 0 through the new arithmetic) is bit-identical to the certified path
   (`test_lookahead_zero_charge_bit_identical`).
2. Toy exactness: PASS — `test_lookahead_logZ_and_posterior_match_exact` (logZ unbiased over 4
   seeds within the existing 0.08 tolerance; MAP + TV match enumeration), plus an
   align+gibbs+lookahead end-to-end MAP gate and the input-contract gate.
3. The probe flips: PASS in substance, with a finding the expectation missed. 3 keys at
   align/band=2/P=64/off + lookahead on "The mother gave the candle the daughter.":

   | key | logZ | w(plain literal) | del_before(The) | top hypothesis |
   |-----|--------|------|------|----------------------------------------------|
   | 0 | −53.82 | 0.875 | 0.00 | the plain sentence @0.88 |
   | 1 | −51.45 | 0.000 | 0.00 | "…gave the candle TO the daughter." @1.00 |
   | 2 | −54.45 | 0.000 | 0.00 | "…gave the candle TO the daughter." @1.00 |

   The leading-deletion artifact is GONE on every key (del_before 2.00 → 0.00; no "\n" parses;
   logZ up ~10 nats). The expectation "w(plain literal) ≥ 0.9" assumed the literal dominates
   everything; it dominates the NEWLINE parse by 8.4 nats, but the findings never scored the
   DATIVE REPAIR. Scoring it directly (penzai, prime "."): LM(to-sentence+EOS) − LM(plain+EOS)
   = +4.62 nats, vs the align prior-mean deletion price log(2/204) = −4.62 — a numerical
   coincidence, and a near-exact JOINT TIE. The posterior on this item is genuinely bimodal
   (plain literal vs to-repair, ≈ 50/50); at P=64 each seed collapses onto one mode, and the
   4-seed evidence merge is the mechanism that reports both. Note what this means for the
   experiment: with the twist, the OFF arm now reaches a mid-sentence deletion repair on this
   item — part of "rejuv=off cannot reach deletions" was this same mid-run weight artifact.

4. §3.1 identity on disk: PASS, exact — `sum(S) + S_end == −logZ` with |diff| = 0.0 on all 8
   records of a local lookahead-ON P16 off run. As derived before implementing: the accumulator
   needed NO changes (its `add` runs pre-resample where the mapping is the identity, and the APF
   residual keeps `(logZ, log_w)` properly weighted across the twisted resample; the on-disk
   identity itself telescopes by construction).
5. Full suite: PASS — 117 tests (the 113 + the 4 new lookahead gates),
   `src/genjax_port/tests/ + experiments/tests/`.
6. Smoke A/B (local, off arm at the MAIN operating point P=64 × 4 seeds, slug `...__la__nseed4`,
   vs the pulled cluster off arm; bd-arm reference from cluster job 21654049 in brackets):

   | item | del_before[0] off→la | p_literal off→la | logZ off→la [bd] | note |
   |---|---|---|---|---|
   | 0 candle       | 1.99→0.00 | 0.84→0.99 | −62.4→−53.4 [−49.5] | fixed |
   | 1 daughter/candle | 0.32→0.00 | 0.15→1.00 | −57.5→−49.5 [−46.2] | fixed; junk MAP repaired |
   | 2 Medics       | 1.72→1.99 | 0.0→0.0  | −89.4→−90.8 [−89.7] | NOT fixed — see below |
   | 3 gifts        | 0.00→0.00 | 0.00→0.06 | −55.6→−54.2 [−52.0] | unaffected item; MAP improved |
   | 4 suspect      | 0.00→0.00 | 0.0→0.0  | −103.0→−100.0 [−97.6] | unaffected item |
   | 5 licked       | 1.94→0.00 | 0.02→0.0 | −73.3→−72.2 [−59.6] | artifact gone; BAD la collapse |
   | 6 coach        | 2.00→0.00 | 0.51→0.0 | −91.2→−85.9 [−75.6] | artifact gone; literal lost |
   | 7 candle+ctx   | 0.00→0.00 | 0.05→0.38 | −25.1→−26.4 [−24.9] | unaffected; bimodal as item 0 |

   What the table says, honestly:
   - The leading-deletion artifact is ELIMINATED on 4 of the 5 affected items (0/1/5/6), with
     logZ up 1–9 nats and item 1's junk MAP ("gave birth to the daughter to the candle")
     repaired to the plain sentence.
   - **Item 2 is a DIFFERENT failure the twist cannot fix by design.** With no context, the
     literal "Medics" costs ~14 nats at step 1, so the one-step fully-adapted proposal
     essentially never proposes the COPY at P=64 — the plain-literal hypothesis is never
     INSTANTIATED, and a resampling twist cannot resurrect a particle that does not exist.
     Verified it is still an inference failure, not a model preference: the joint prefers the
     plain sentence over both junk MAPs by ~17–18 nats (LM scored with the item's own prime;
     "It was …" −7.5 LM gain − 9.25 deletions; "This article medic …" −4.7 LM − 13.75 channel).
     gibbs+bd only PARTIALLY repairs it (del_before 0.83, p_literal 0.0). This is
     proposal-support myopia (the step-1 intermediate target), out of this fix's scope.
   - **New cost at P=64: heavier per-seed tails.** The residual −psi[anc] hands a surviving
     laggard a large positive weight; the cloud then often ends the run collapsed on one mode
     per seed. On item 5 all four la seeds lost both the literal and the "kicked" repair
     (merged MAP "The boy looked licked from …", whose joint is ~19 nats WORSE than the kicked
     repair — scored directly; seed logZ spread 2.4→7.0), and on item 6 the la cloud dropped
     the literal (p_literal 0.51→0.0) for the "and"-repair while bd keeps p_literal 0.94.
     Everything stays unbiased (logZ still rose on 6 of 8 items), but per-item posterior
     quality at P=64 is noisier where several modes compete. The battery A/B (gate 7, the
     user's call) is the arbiter of the net effect.

7. Battery A/B: RUN (user-approved; cluster jobs 21666107 baseline / 21666110 lookahead,
   87 items, off arm, P=64, 2 seeds; diff `planning/la_vs_off_diff.py` →
   `planning/calibration_la_vs_off.csv`; full outcome in the RUNLOG entry). Headline: the
   artifact was on **46/87** baseline items — far beyond the smoke's 5/8 — and lookahead clears
   it on 39, leaving 7 (Medics-class proposal-support failures and partial clears). logZ mean
   +4.05 / median +1.49 (51 up, 14 down); matches-expected 42→44 exact / 47→49 case-insensitive;
   edited-rate 32→28 with visibly junky MAPs cleaned. Cost confirmed but bounded: 11
   newly-correct vs 9 newly-wrong MAPs (several of the losses are the P=64 heavy-tail collapse),
   and the 2-seed logZ spread rose 2.72→2.99. Note the experiment runs 4 seeds, not 2.

## Session bootstrap (state as of 2026-08-31, end of the finding session)

- Branch `experiment-harness`, local HEAD ahead of the cluster: cluster is at `80a4722`;
  local has `2c79d9b` (findings+probes), `dc2ffdb` (doc rewrite), plus this plan's commit.
  Before any submit: `git push origin experiment-harness` and ff-pull on the cluster
  (`run.sh submit` refuses on mismatch and prints the commands).
- Cluster smoke: `smoke × main_off` job 21654051 COMPLETE (8/8, 24:03, MaxRSS 12.97G ⇒ use
  MEM=24G for the off arm). `smoke × main_bd` job 21654049: shards 0 (43:44, 12.67G) and
  2 (30:20, 10.71G) complete; **shards 1 and 3 were still running at handoff — check
  `sacct -j 21654049 -P --format=JobID,State,Elapsed,MaxRSS,ExitCode`, then
  `bash experiments/run.sh pull smoke`, and append the outcome line to the 15:11:20Z RUNLOG
  entry** (it still says "(append when finished)").
- Cluster env: `regex==2026.8.31` was installed into ncgenjax (see the RUNLOG env-fix note and
  the `orcd-env-usersite-dependence` memory — the env borrows numpy/wordfreq from `~/.local`).
- ssh: user must have run `ssh -fN orcd`; verify `ssh -O check orcd` before remote steps.
- Gotchas: `conda run` swallows heredoc stdin (write scripts to files); REJUV has no default —
  every entry point needs an explicit off/gibbs/gibbs+bd; do not restore defaults.
