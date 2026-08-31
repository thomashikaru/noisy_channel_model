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
