# Option C — merge copy + substitution into a single "align" action

**Status:** drafted 2026-06-21, **not yet implemented.** Branch: `align-action-channel` (forked from
`word-action-rejuv` @ `5cf7c0c`). This plan is self-contained — a fresh session should be able to execute
it end-to-end. **Reversibility contract:** all work is on this branch; the new behaviour is gated behind a
**new channel name** (`channel="align"`) so the existing `word_action` (deployment) and `char_copy`
(certification anchor) paths stay byte-for-byte untouched; no defaults change until Phase 5 (user
approval). To abandon: `git checkout word-action-rejuv` and delete this branch. Nothing else is affected.

---

## 0. Why we're doing this (the motivating failure)

Running the deployed word-action channel (α=200) on **"The garage needs to be tossed out."** returns
**"The garage door needs to be tossed out."** with p=1.00 — it restores a spurious word "door" instead of
correcting the obvious typo **garage → garbage** ("The garbage needs to be tossed out."). We traced it (a
full diagnosis is in the session that wrote this plan; reproduce the key numbers in the Appendix):

- **Candidate retrieval is fine** — `" garbage"` is the *nearest* candidate (single-token, Damerau-Levenshtein
  distance 1, token id 22630). It is in the pool and still gets **zero mass** anywhere in the trace.
- **The LM agrees with our intuition** — pythia-70m scores `LM("…garbage…")` = −42.0 > `LM("…garage door…")`
  = −45.2 > `LM(literal "…garage…")` = −46.3. The LM *prefers* garbage by +3.2 nats over garage-door.
- **The channel over-penalizes the substitution.** A substitution pays `log p_sub + SUB_FORM_LP` (−8.56 nats
  at α=200, d=1); restoring a dropped word pays only `log p_del` (−5.31) with **no form cost**. The 3.26-nat
  `SUB_FORM_LP` (=log 1/26) gap almost exactly cancels the LM's 3.2-nat preference → joint(garage door) beats
  joint(garbage) by **0.024 nats**. A knife-edge that the channel's anti-substitution bias tips the wrong way.

**Root cause (the structural one we're fixing):** copy and substitution are modelled as **separate discrete
actions**. Copy is favoured at α_copy=200; substitution is a rare α_sub=1 event **plus** the per-edit form
cost. So reading a surface as a 1-char near-miss instead of a perfect copy costs the full copy→sub prior drop
(**5.3 nats**) *and* the form (**3.26 nats**). Decompose the two gaps at α=200, d=1:
- substitution vs **copy** = (log p_sub − log p_copy) + form = **−5.3 (α) − 3.26 (form) = −8.56**
- substitution vs **deletion** = (log p_sub − log p_del) + form = **0 + (−3.26) = −3.26**  ← the action prior
  is symmetric between sub and del (α_sub=α_del=1), so the *entire* sub-vs-deletion over-penalty is the form
  cost; α plays no role there.

## 1. The design — one "align" action with a smooth distance emission

Replace the 4-way action space **{copy, sub, insert, delete}** with the 3-way **{align, insert, delete}**:

- **align** = "this intended word corresponds to this observed unit," at *any* edit distance. A copy is just
  `align` at d=0; a near-substitution is `align` at d=1. There is **no categorical copy/sub jump** — the
  Dirichlet now governs only align-vs-insert-vs-delete, and "how good is the match" lives entirely in the
  emission, a smooth function of edit distance.
- **Emission** for aligning observed unit `o_k` to candidate surface `s`:
  `emit[k,s] = log p_align + K · d_ci(o_k, s)` where `K = ALIGN_SLOPE` (a per-edit log-cost; start at the
  existing `SUB_FORM_LP = log(1/26) ≈ −3.258`, then sweep) and `d_ci` is the **case-insensitive** edit
  distance (so a pure capitalization "she"→"She" is d=0, free — see §3 note). In the current code the form
  table already *is* `SUB_FORM_LP · d` with copy=0, so this is: **emission = form_table + log p_align**, with
  the copy/sub offset removed.
- **deletion** (up arc): `wdel_p = log p_del` (unchanged mechanism).
- **insertion** (spurious pass): `wins_p = log p_ins + content` (unchanged mechanism; content = −unigram surprisal).

The over-editing control moves from "substitutions are a-priori rare" (prior-driven) to "near-misses pay a
distance cost K" (surface-driven). K is the one knob; it is decoupled from α and from the insert/delete rates.

**Expected effect on the garage case** (α_align=200 → p_align=200/202, p_del=1/202; K=log(1/26)):
- garbage = align(d=1): `log p_align + K` = −0.0099 − 3.258 = **−3.27**
- garage-door = align(garage, d=0) + del(door): −0.0099 + (−5.31) = **−5.32**
- channel now favours garbage by **+2.05**; with the LM's +3.22, **joint(garbage) − joint(garage-door) ≈ +5.3**
  → garbage is the MAP. And garbage beats the *literal* by `LM_gain(+4.28) − K(3.26) ≈ +1.0` — the correct
  "correct only when the LM prefers the neighbour by more than the edit cost" behaviour. **Verify these two
  margins after implementing (Phase 3).**

Sanity on over-editing: a clean in-context word's literal is already the LM's top choice, so no neighbour beats
it by K=3.26 → clean words are left alone. The garage case corrects *because* the LM genuinely prefers garbage
by 4.28 over the literal. If Phase 4 shows over-editing, raise K; if under-editing, lower it.

## 2. Reversibility & safety contract (read before touching code)

1. All work on branch `align-action-channel`. Merge target (back into `word-action-rejuv` or `master`) is the
   user's choice at Phase 5 — **do not merge without approval.**
2. Implement `align` as a **new `channel` value**, parallel to `word_action` and `char_copy`. The existing two
   paths must stay **bit-identical** — re-run the full suite (Phase 1 gate) and confirm 24/24 still pass and the
   `word_action`/`char_copy` outputs are unchanged.
3. **No default changes** (ACTION_ALPHA_DEFAULT, `--channel` default, `--selftest`, `run_example_native.sh`)
   until Phase 5, and only on approval.
4. Commit per phase with clear messages so any step is individually revertible.

## 3. Implementation (Phase 1) — the `align` channel

All in `src/genjax_port/`. The 4-way word-action machinery to mirror for the 3-way align variant:

- **`pythia_word_caprop.py`**
  - Add `ALIGN_ALPHA_DEFAULT = (200.0, 1.0, 1.0)` (align, ins, del) and `ALIGN_SLOPE = SUB_FORM_LP` (the
    sweepable per-edit cost; keep it a named constant so Phase 4 can vary it).
  - Extend the `channel` selector (currently `run()` ~L307–316 and the `--channel` argparse ~L427): accept
    `"align"`. `"align"` requires a length-3 `action_alpha`; validate (mirror the `word_action` length-4 check).
    Keep `channel=None` inference unchanged (still → word_action / char_copy).
  - The form channel for align is the existing `channel_form_logpdf` but with its per-edit cost = `ALIGN_SLOPE`
    instead of the hardcoded `SUB_FORM_LP` (parameterize `_channel_dp`'s sub/ins/del char cost so Phase 4 can
    sweep K without editing source each time — e.g. thread `ALIGN_SLOPE` through `_pythia_model`).
- **`pairhmm_smc.py`** (the core; the 4-way pieces are at the line refs below — verify them, the file may have
  moved):
  - **`_theta_to_costs` (~L375):** add a 3-way variant returning `(lp_align, wdel_p, wins_p)` from a
    `theta` of shape `(P,3)` over (align, ins, del). `lp_align = log θ[:,0]`, `wdel_p = log θ[:,2]`,
    `wins_p = log θ[:,1] + wins_vec`.
  - **theta init (~L520):** when `channel=="align"`, draw `theta ~ Dirichlet(align_alpha, (P,3))` and call the
    3-way costs. (Keep the 4-way branch for `word_action`.)
  - **Emission offset (~L258 in `_caprop_scores`, ~L611 in `_word_step`, and `word_dp.channel_carry` ~L86):**
    replace `emit_cols + lp_sub + (lp_copy − lp_sub)*is_copy` with **`emit_cols + lp_align`** for the align
    channel. Note `emit_cols` is the form table (case-insensitive distance × ALIGN_SLOPE). **Case subtlety:**
    the current form table is case-*sensitive*; the case-insensitivity was supplied by `copy_mask` in the
    offset. Under align you still need a capitalization to be d=0/free. Two options — recommend (b):
      - (a) quick: `emit = where(copy_mask, 0.0, emit_form) + lp_align` (zeroes the form for exact
        case-insensitive copies; a d≥1 sub with mixed case is still slightly over-counted).
      - (b) correct: compute the form channel on **case-folded** surfaces (lower-case both before the char DP),
        so `d_ci` is right at every distance; then `copy_mask` is no longer needed in the emission.
  - **`_action_counts` (~L389):** add a 3-way variant `(n_align, n_ins, n_del)`: `n_align` = active slots with
    positional index `< M`; `n_del` = active slots with index `≥ M`; `n_ins = max(0, M − n_words)`. No
    `copy_mask` needed (align doesn't distinguish copy vs sub).
  - **theta refresh (~L689):** when align, `counts = _action_counts_align(...)`; `theta ~ Dir(align_alpha +
    counts)` (3-way); recompute `(lp_align, wdel_p, wins_p)` and the forward carry.
  - **`channel_carry` (`word_dp.py` ~L58):** add a 3-way path (pass `lp_align` instead of `lp_copy/lp_sub`,
    drop `copy_mask` if using option (b)). Keep the existing 4-way signature for `word_action`.
- **Gate:** after Phase 1, run the full suite (Phase 1 acceptance = 24/24 unchanged) and a `word_action`
  spot-run to confirm byte-identical output vs `master`/`word-action-rejuv`.

## 4. Re-certification (Phase 2) — extend the exact-enumeration gate

`tests/` is `src/genjax_port/tests/`. `test_pairhmm_exact.py` certifies the filter against brute-force exact
enumeration on the toy bigram; it already has `word_action` variants (`test_wa_*`) that build the exact
word-action posterior at a fixed θ and check the sweep matches. Mirror them for align:
- Add an `align` exact posterior helper: score each enumerated intended sentence with the **3-way** channel
  (emission = form + log p_align; del = log p_del; ins = log p_ins + content), summed over alignments by the
  shared `channel_carry` (3-way path). Use the toy `channel_form_logpdf` with `ALIGN_SLOPE`.
- Add gates: `test_align_logZ_matches_exact`, `test_align_map_matches_exact`,
  `test_align_posterior_mass_matches_exact`, `test_align_rejuv_invariant`, `test_align_smcp3_weight_zero`.
- **Acceptance:** new align gates pass AND the existing 24 still pass (`PYTHONPATH=src conda run -n ncgenjax
  python -m pytest src/genjax_port/tests/ -q`).

## 5. The decisive test (Phase 3) — does the garage case flip?

Run the garage sentence under `channel="align"` (α=(200,1,1), K=log(1/26), rejuv=gibbs, P=128):
- **Pass:** MAP = "The garbage needs to be tossed out."; verify joint(garbage) − joint(literal) ≈ +1.0 and
  joint(garbage) − joint(garage-door) ≈ +5.3 (script the joint as LM + 3-way channel, as in the Appendix).
- Add 4–6 more **sub-vs-indel minimal pairs** (a substitution into a real word competing with a
  word-restoration or a spurious-insertion), varying neighbourhood density and LM-gap. This is the
  calibration signal the old battery lacked (it has only ordinary-density, isolated d=1 subs). Author them so
  the *intended* correction is unambiguous (verify the LM gain first, per [[noisy-channel-test-examples]]).

## 6. Battery regression + K / α sweep (Phase 4)

- Re-run the existing calibration subsample (`planning/wa_alpha_subsample.txt`, 40 items) under `channel="align"`
  using the existing runner (`calibration_word_action_smc.py`; it will need a small hook to pass `channel="align"`
  + the 3-way alpha — add an env switch, do **not** change its word_action default).
- **Sweep K** = ALIGN_SLOPE ∈ {log(1/8), log(1/15), log(1/26), log(1/40)} and optionally α_align ∈ {100, 200, 400}.
- **Decision surface** (same metrics as the word-action sweep, via `calibration_battery_analyze.py`):
  should-KEEP literal retention **L** high (no new over-editing — the main risk of cheaper subs), should-EDIT
  correction **E** up (the SUBN/SUBW families should improve), within-pair tracks-manipulation up, junk low.
  Plus the new sub-vs-indel items from Phase 3 must pass.
- Pick the K that fixes garage + the new items **without** dropping KEEP retention below the word-action
  baseline (L≈0.99 under gibbs). Expect a gentler K (less negative than log(1/26)) only if needed.

## 7. Decision & promotion (Phase 5 — REQUIRES USER APPROVAL)

Summarize: does align (best K) fix the garage class while holding the battery? Present the numbers. **Only on
approval:**
- Promote `align` to the deployment default: set the `--channel` default and `ACTION_ALPHA_DEFAULT` analogue,
  rebuild `--selftest`, update `run_example_native.sh` (skip-worktree — edit freely now, see
  [[keep-run-example-script-current]]), and update `model_current.tex` (§3.4 becomes the align action; the
  reference table). Consider whether `word_action` is retired or kept as an opt-in.
- Merge the branch to the user's chosen target.

## 8. Environment & commands

- Conda env **`ncgenjax`** (arm64); **pytest is installed** — use it, don't hand-roll a collector
  ([[genjax-conda-env]]). Always redirect long runs to a file, never pipe ([[never-pipe-expensive-output]]);
  the Bash shell is zsh (`${=VAR}` to word-split). `conda run` buffers stdout.
```
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh
PYTHONPATH=src conda run -n ncgenjax python -m pytest src/genjax_port/tests/ -q     # 24/24 + new align gates
NC_LM=EleutherAI/pythia-70m ./run_example_native.sh "The garage needs to be tossed out." 128   # garage check
```

## 9. Deliverables
The `align` channel (gated, existing paths bit-identical) + its exact-enumeration gates; a Phase-3 result
showing garage flips with margins; a Phase-4 K/α sweep + battery regression; a small sub-vs-indel calibration
family; and a one-paragraph recommendation (promote align / keep tuning / abandon). No default or doc changes
land before Phase 5 approval.

---

## Appendix — verified baseline numbers (reproduce to confirm the starting point)

Current `word_action` @ α=200 on "The garage needs to be tossed out." (P=128, rejuv=gibbs): MAP = "The garage
door needs to be tossed out." p=1.00; ESS collapsed to 2.4/128; "garbage" appears 0× in the trace; "door"
1036×. LM logprobs (pythia-70m, EOS-seeded, incl. terminal EOS):

| sentence | LM logprob |
|---|---|
| literal "The garage needs to be tossed out." | −46.29 |
| "The garbage needs to be tossed out." | **−42.01** |
| "The garage door needs to be tossed out." | −45.23 |
| "The garbage door needs to be tossed out." (one-step neighbour of the winner) | −50.42 |

Channel @ α=200 (prior mean): log p_copy=−0.015, log p_sub=log p_del=−5.313, SUB_FORM_LP=−3.258.
Current joint(garbage) − joint(garage door) = `LM_gap(+3.22) + [log p_sub+SUB_FORM_LP − (log p_copy+log p_del)]`
= 3.22 − 3.243 = **−0.024** (garage-door wins). Under align the bracket becomes `K − log p_del` =
−3.258 − (−5.313) = +2.05, so the joint flips to **+5.27** for garbage. Candidate pools (max_dist=2, geom base
1/26): garage → 26 cands (3 at d=1: garbage/farage/garages, 23 at d=2); garbage → 6 cands (1 at d=1: garage).
