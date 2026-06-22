# Birth/death rejuvenation — a single involutive add/remove-a-word move

**Status:** drafted 2026-06-21, **not yet implemented.** Branch: `rejuv-birth-death` (forked from
`align-action-channel` @ `efe204d`). Self-contained — a fresh session should execute it end-to-end.
**Reversibility contract:** all work is on this branch; the new move is a **separate sweep** gated behind a
new `rejuv` selector value (`rejuv="gibbs+bd"`), so the existing substitution-only `rejuv="gibbs"` path and
the certified `rejuv="off"` exact-enumeration path stay byte-for-byte untouched; no defaults change until a
final user-approval phase. To abandon: `git checkout align-action-channel` and delete this branch.

---

## 0. The gap we're closing

The current rejuvenation sweep (`pairhmm_rejuv.py`) is **substitution-only**: for each existing intended-word
slot `w` it resamples the word from `[COPY] ++ pool[w]` (`_candidates`, `pairhmm_rejuv.py:273`) and the move
is gated by `active = w < n_words` (`_apply_move`, `:345`). It **never changes `n_words`** — it can neither
remove a slot (delete a word) nor add one (insert a word). The number of intended words is fixed once the
forward filter commits it (`kernel`, `pairhmm_smc.py:298`, `n_words + 1` on advance).

Consequence, flagged in `ALIGN_OPT_RESULTS.md:127`: a duplicated intended word (`INS-01a "handed handed"`)
is never removed, because nothing downstream of the forward pass can drop a word. More generally, all
word-count corrections during rejuvenation are impossible — restoring a dropped word can *only* happen in
the forward pass (via the `alpha` del-action), and removing a spurious one can happen *nowhere*.

**The fix:** a rejuvenation move that changes `n_words` by ±1, leaving the posterior invariant.

### Why one observed word can be "removed" with no new channel code

The channel forward DP (`word_dp.channel_carry` → `alpha[M]`) already marginalizes the full edit alignment,
including channel **insertions** (a spurious *observed* word, cost `WINS`). So when a death move removes an
intended "handed", the now-unmatched observed "handed" is automatically re-explained by the DP as a WINS
insertion — the target `π(y')` for the shorter sentence is already correct. No alignment bookkeeping is added;
we only need to *propose* the shorter (or longer) sentence and score it through the existing DP.

## 1. The design — one involutive birth/death move

A single move that, at a chosen position, **either inserts or removes one intended word**, with an
involution `φ` that flips the two directions. Self-inverse ⇒ dimension matching is automatic and the
"Jacobian" is 1 (everything is discrete).

Augmented variable `(y, d, w, x)`:
- `y` — the intended-word sequence (the latent we rejuvenate),
- `d ∈ {birth, death}` — direction,
- `w` — a position/gap index,
- `x` — a word; the **auxiliary that matches dimensions**.

The involution:

```
φ(y, birth, w, x) = ( insert(y, w, x), death, w, ∅   )
φ(y, death, w, ∅) = ( delete(y, w),    birth, w, y_w )
```

`φ∘φ = id`. Birth *draws* a word (randomness +1, state +1); death *recovers* the deleted word into `x`
deterministically (randomness −1, state −1) — so total (state+randomness) dimension is preserved, the
involutive-MCMC / reversible-jump requirement. This is genuinely **one** move (direction sampled inside it)
and **its own inverse**, per the request.

It runs as a **separate sweep** beside the substitution sweep; both target the same `π` and compose. Keeping
them separate (rather than folding DELETE/INSERT candidates into the existing per-slot categorical) avoids
mixing a fixed-dim full-conditional Gibbs step (weight ≈ 0) with a trans-dim step (weight ≠ 0) in one
normalization, and keeps the certified substitution path bit-identical and the new move independently
gateable and testable.

## 2. The weight (SMCP3 / reversible-jump, Jacobian 1)

Target `π = P_LM^λ · P_channel` — exactly what the current sweep scores (`lm_temp * chain + chan`,
`pairhmm_rejuv.py:392`). For a **birth** proposing word `x` at gap `w` (reverse is a death at that position
in `y'`):

```
log W = [ logπ(y') − logπ(y) ]                                  ← target ratio
      + [ log p_death(y') + log q_del(w | y') ]                 ← reverse (death) proposal
      − [ log p_birth(y)  + log κ(w | y) + log q_ins(x | w, y) ] ← forward (birth) proposal
```

and the mirror for a **death** (forward `p_death·q_del`; reverse `p_birth·κ·q_ins` re-inserting the removed
word). Two efficiencies:

- **Channel:** `logπ_channel(y')` is one extra `channel_carry → alpha[M]` call — the DP already exists and is
  length-agnostic. (For DONE particles read `alpha[M]`; for mid-loop ones `logsumexp(alpha)`, exactly as
  `_chan_scores`, `pairhmm_rejuv.py:291`.)
- **LM:** birth/death at `w` changes the sentence only from `w` onward, so the LM **prefix cancels** in the
  ratio — score only the suffix tail, reusing the KV suffix-tail scorer the sweep already drives
  (`_tail_inputs`/`tail_fn`, `:312`/`:391`).

Unlike the substitution sweep (full-conditional Gibbs, weight ≈ 0 — asserted as a build-it-right check in
`_smcp3_move`), this move carries **real weight**; the plumbing to fold `move_logw` into `log_w` before
resampling already exists (the sweep returns `move_logw`; the filter folds it pre-resample). Correctness is
independent of proposal quality; **proposal quality only controls weight variance / ESS** (the main practical
risk), tuned in §3.

I will **hand-compute this weight vectorized over P** (as `_smcp3_move` already inlines the recipe rather than
threading per-particle scores through genjax's trace API) — the trans-dim ratio is the clean closed form above.

## 3. Proposals (efficiency, not correctness)

- **Death** `q_del(w | y)` — near-conditional: enumerate removing each current word, score each `(n−1)`-word
  target, sample `w ∝ softmax`. Concentrates on duplicates / low-LM-gain / WINS-explainable words. Suffix-LM
  sharing makes the `n` evals cheap.
- **Birth** `q_ins(w, x | y)` — place `w` where the channel currently pays an alignment/WDEL penalty; draw `x`
  from the **forward filter's own inventory**: the channel-compatible pool plus the top-J LM "bridge" words
  (`_rejuv_pool_from_inventory`, `pairhmm_smc.py:178`; the `_caprop_scores` bridge set, `:213`). Word choice can
  itself be a local full conditional over that pool.
- `p_birth/p_death` = ½, forced at the boundaries `n = 0` (birth only) and `n = Wmax` (death only); the weight
  formula absorbs the boundary asymmetry.

## 4. Implementation mapping (where the code goes)

- **Buffers** (`word_tok [P,Wmax,T_max]`, `word_len`, `word_surf`, `n_words`): insert/delete a slot = a shifted
  copy of those arrays + the existing `_pack` cumsum re-pack (`pairhmm_rejuv.py:116`), which already handles
  variable spans. `n_words ± 1`. Births need headroom `n_words < Wmax`; `Wmax = M + slack` — may need to bump
  `slack` (cheap) if births are frequent.
- **New code**, all additive to `pairhmm_rejuv.py`: `insert(y,w,x)` / `delete(y,w)` + auxiliary recovery; the
  position/word kernels (`q_del`, `q_ins`, `κ`); the vectorized `log W`; and a `birth_death_sweep(...)`
  returning `(state', move_logw)` in the same shape contract as `make_sweep`'s `sweep`.
- **Hook:** new `rejuv="gibbs+bd"` selector in `pairhmm_smc.run` (`:462`) that runs the substitution sweep then
  the birth/death sweep per resample event. `rejuv="off"` and `rejuv="gibbs"` unchanged.

## 5. Phased execution

The move is **atomic** — both directions exist from the first working version. A death's SMCP3 weight
references the birth/insert reverse density `q_ins` (the `+ reverse_prop` term in §2), and a one-directional
kernel cannot be π-invariant (it only ever shrinks `n_words`). So there is **no "death-only" build**. What
stages is (a) **proposal sophistication** and (b) **which cases we validate against** — not direction.

- **Phase 0 — involution core.** `insert(y,w,x)` / `delete(y,w)` + auxiliary recovery on the buffer reps, and
  the `φ` round-trip — **both** directions (the involution *is* the pair). No scoring yet. Gate: **involution
  round-trip test** (apply, re-apply with recovered aux, assert exact state + fwd/bwd density consistency) —
  the analogue of the `weight ≈ 0` build-it-right assertion in `_smcp3_move`.
- **Phase 1 — the complete move, cheap proposals.** Both directions, the trans-dim weight, folded into
  `log_w`; start with cheap proposals (uniform `κ`, near-conditional `q_del`, top-of-pool `q_ins`). Validate
  first on the **death-dominated** motivating case (`INS-01a "handed handed"` duplicate removed) and the toy
  **posterior-invariance** test (small enumerable variable-length target: equally-weighted-in ⇒ correctly
  reweighted); certified `off`/`gibbs` paths bit-identical.
- **Phase 2 — informed proposals.** Upgrade `q_ins` to the forward-filter inventory + LM bridges and `q_del`
  to the full near-conditional. Gate: **birth-dominated** restorations the forward filter already does
  (`DEL-of`, `INS-02a`, the `ALIGN_OPT_RESULTS.md:65` wins) don't regress, and ESS isn't materially worse
  than `gibbs`.
- **Phase 3 — battery + writeup.** 40-item battery (E/L/junk), `gibbs` vs `gibbs+bd`; record in
  `REJUV_BIRTH_DEATH_RESULTS.md`. Recommend default only on a clean win.

## 6. Tests / correctness ledger

1. **Involution round-trip** (Phase 0) — `φ∘φ = id` on state + fwd/bwd density consistency.
2. **Posterior invariance on the toy** (Phase 1) — enumerable variable-length target; the move leaves it
   invariant.
3. **Certified path untouched** — `rejuv="off"` exact-enumeration gates and `rejuv="gibbs"` results
   bit-identical with the move off.
4. **Motivating case** — `INS-01a "handed handed"` duplicate removed (Phase 1).
5. **No-regression** — forward-pass restorations preserved under `gibbs+bd` (Phase 2).

## 7. Risks

- **Weight variance / ESS** — the only real risk; trans-dim weight ≠ 0. Mitigated by near-conditional proposals
  (§3); measured in Phase 2/3.
- **Two restoration paths** — birth duplicates the forward filter's deletion-restoration. Same posterior, but
  watch for double-restoration instability (Phase 2 no-regression gate).
- **Headroom** — births capped at `Wmax`; bump `slack` if needed.

## 8. Open choices (decide before Phase 2)

- Run birth/death **per position** (like the substitution sweep) vs a fixed **K attempts** per resample event.
- Whether `q_ins` word choice is a full local conditional (lower variance, costlier) or a cheap top-1 from the
  pool.
