# Goal 1 — Cut rejuvenation's LM-forward cost (dedup the move; un-fuse vs. compact)

> **⚠️ PARTIALLY OVERTAKEN BY EVENTS (2026-06-16, session 2). Read this note first.** The move changed:
> Goal 2 replaced `manual_subflip_move`'s **two** `[P,M]` forwards with **one** `[P*K,M]` forward (K
> candidate buffers, suffix-aware proposal). So the "two forwards" framing below is stale. Cost cuts
> since: `MAX_SUB_CANDIDATES=32` (K 129→33, ~4×) + per-window K-bucketing. **The CHOSEN perf direction
> is now the prefix-KV-cache** (removes the K-fold shared-prefix recompute; win ~`M/tail`, independent
> of K) — BOTH de-risking spikes have PASSED (see the `rejuv-prefix-kv-cache-spike` memory + the
> migration memory NEXT section); it is the primary next task, not the dedup/compaction below. The
> dedup idea here (Step 1A) is still valid and COMPLEMENTARY (it dedups the P axis; the cache removes
> the prefix recompute), but lower priority. Step 0 (measure degeneracy) is still worth doing before
> investing further in dedup specifically.


## Problem
Interleaved rejuvenation slows runs ~10x for the same particle count. The cost is LM forwards,
and the move multiplies them: per *fired* word it runs `2 * n_sweeps * nwin` full `[P, M, V]`
forwards (current + proposed buffer, no prefix reuse), over **all P particles regardless of how
many particles' gates fired**. The per-particle gate controls which results are *kept*
(`jnp.where(gate, ...)`), not how much LM compute is *done*. The only compute the gate currently
saves is the whole-word skip when zero particles fire.

This goal removes wasted forwards. It is a **pure performance change**: posterior must be
bit-identical (given the same RNG) before and after — there is a detailed-balance/regression test
that must keep passing.

## Read these first
- `src/genjax_port/rejuv_bridge.py`
  - `manual_subflip_move` (~L431): the move; **two** `L._raw_logits(buf)` calls (~L455, ~L468).
  - `_aligned_window_move_fn` (~L478): wraps the whole window×sweeps loop in **one `jax.jit`**.
  - `_make_aligned_subflip_hook` / `run_smc_conditional_rejuv_aligned` (~L598, ~L646): the
    production interleaved path that calls the above per word.
- `src/genjax_port/cache_dedup.py`: the filter's host-side dedup. `_dedup_apply` finds unique
  rows by `buf[:i_len].tobytes()`, runs the LM on uniques (padded to a bucket ladder), scatters
  back via an inverse index. `make_dedup_fns` returns drop-ins for `next_token_*`.
- `src/genjax_port/lm_penzai.py`: `_raw_logits(token_buf) -> [rows, M, V]` (all positions; takes
  no `i_len`); `next_token_*` (next-token only, `[rows, V]`).
- `src/genjax_port/tests/test_rejuv_bridge.py`: detailed-balance + parity tests (the guardrail).

## The central tension (decides the whole approach)
Dedup is **host-side** (a Python dict over buffer bytes) so it must run **outside `jax.jit`**. The
production move is deliberately **fused into one jitted graph** to avoid eager-dispatch overhead.
You cannot have both as-is: dedup forces un-fusing back to an eager Python loop that calls a
per-move jit.

Dedup and gate-compaction exploit the **same** degeneracy and are for different regimes:
- **Degenerate set** (few distinct readings — the usual case right after resample-every-word):
  **dedup** shrinks the forward to the handful of unique buffers, irrespective of the gate.
- **Diverse set but few gated** (many distinct readings, few surprised): **compaction** (gather
  gated particles into a dense sub-batch) is the tool; dedup wouldn't help.

Post-resample you are almost always degenerate, so dedup is expected to capture most of the win
and compaction is likely redundant. **Do not build both blind. Measure first.**

## Step 0 — Measure degeneracy (do this before any change)
Add a temporary counter (mirror `cache_dedup.DedupStats`) inside the rejuv hook: for each fired
word, record `n_gate`, `P`, and the number of **unique** gated buffers (`buf[:i_len].tobytes()`
over gated rows). Run the eval set (`src/genjax_port/tests/eval_rejuv.py`) and the example script
(`run_example_native.sh`). Report, over fired words: median `n_gate/P` and median
`unique_gated/n_gate`.

Decision rule:
- If `unique_gated` is typically small (e.g. ≤ ~⅓ of `n_gate`): **dedup wins** → do Step 1A.
- If sets are diverse (`unique_gated ≈ n_gate`) but `n_gate ≪ P`: **compaction wins** → Step 1B.
- If both `n_gate ≈ P` and `unique_gated ≈ n_gate`: neither helps much; instead cut `n_sweeps`,
  shrink `lookback`, or harden the gate (raise `logprob_thresh`) — and stop here.

Delete the counter (or leave it behind a `stats=` kwarg like `DedupStats`) once decided.

## Step 1A — Dedup the move's forward (degenerate regime)
1. Add a raw-logits dedup wrapper in `cache_dedup.py`. `_dedup_apply` is reusable as-is: its
   `forward_fn` is called as `forward_fn(buf[rep_idx], i_len[rep_idx])` and it scatters `out[inverse]`
   on the leading axis, which works for `[U, M, V]` too. So:
   ```
   def make_raw_dedup_fn(stats=None):
       def raw(token_bufs, i_lens):
           return _dedup_apply(lambda b, _il: L._raw_logits(b), token_bufs, i_lens, stats)
       return raw
   ```
   Keying on `buf[:i_len]` is **correct** for the move: the move only ever reads logits at
   positions `< i_len` (`chain_from_pos` masks `idx < i_len`; `lm_at` reads `posc-1 < i_len`), and
   under causal attention those depend only on the filled prefix — so byte-equal filled prefixes
   give identical relevant logits.
2. Un-fuse the window loop. Replace the single fused `_aligned_window_move_fn` with an **eager
   Python loop** over `n_sweeps × window` that calls a `jax.jit(manual_subflip_move)` per move,
   with `manual_subflip_move`'s two `L._raw_logits` calls routed through the injected
   `raw_logits_fn` (default = `make_raw_dedup_fn()`), and `log_softmax` applied **outside** dedup
   (dedup returns logits; softmax stays in-jit on the small unique/scattered result, or apply it in
   the move after scatter). Keep the per-move function jitted so each move is still one compiled
   unit — you lose cross-move fusion, not per-move compilation.
3. **RNG parity:** the fused body splits `key` per move (`key, mk = jax.random.split(key)`),
   `n_sweeps` then `nwin` order. Reproduce that exact split order in the eager loop so the posterior
   is unchanged.
4. Put it behind a flag (e.g. `rejuv_dedup=True`) threaded from `run_smc_conditional_rejuv_aligned`
   so you can A/B against the fused path.

## Step 1B — Compact gated particles (diverse regime; only if Step 0 says so)
1. In the hook, after computing `gate [P]`, get padded gated indices with a fixed bucket size:
   `idx = jnp.nonzero(gate, size=B, fill_value=P)[0]` where `B` is the smallest rung of a small
   ladder (e.g. `(8, 16, 32, 64)`) `>= n_gate` (you already read `n_gate = int(jnp.sum(gate))` for
   the zero-skip, so the host sync is already paid). Use `fill_value=P` (a scratch row) so pad rows
   never collide with a real particle.
2. Allocate buffers with **one extra scratch row** (`P+1`). Gather `buf[idx]`, `i_len[idx]`,
   `pos_win[idx]`, `cand_l_win[:, idx]` into the `[B, ...]` sub-batch; run the existing fused move
   on `[B, M]`; scatter back `buf = buf.at[idx].set(moved)` — pad rows write to scratch row `P` and
   are discarded.
3. One compile per `(B, nwin, n_sweeps)` — the ladder keeps that to a handful.
4. **RNG parity is harder here** (the move now sees `B` rows, not `P`): you cannot keep bit-exact
   parity with the un-compacted path. So this changes the sampled outcome. Validate by *posterior
   quality* (Validation below), not bit-parity, and gate it behind a flag.

## Independent cheap win (do regardless): collapse the two forwards
`manual_subflip_move` runs two full `[P, M, V]` forwards for buffers that differ at exactly one
position; everything left of the flip is identical and cancels in the weight. Investigate computing
the proposed-buffer suffix logits without recomputing the shared prefix (prefix-KV reuse, or at
minimum confirm XLA isn't already CSE-ing it — profile both forwards). If a KV-prefix cache is not
readily available via penzai, note it and defer (it's the harder cross-step cache from the
`genjax-port-cache-trie` memo). Document findings either way.

## Validation
- **Bit-parity (Step 1A only):** `pytest src/genjax_port/tests/test_rejuv_bridge.py` must pass
  unchanged; a fixed-seed run of `run_smc_conditional_rejuv_aligned` must produce identical
  sentences and `log_marginal` with `rejuv_dedup` on vs. off.
- **Speed:** time `run_example_native.sh` and `eval_rejuv.py` before/after; report forward count
  (via the `stats` counter) and wall-clock. Success = materially fewer forwards on fired words and
  lower wall-clock with **no** posterior change (1A) / no quality regression (1B).
- **Quality (Step 1B / compaction):** run `eval_rejuv.py`; the noisy-channel eval mean must not
  drop vs. the current path.

## Done when
Profiling shows the rejuv forward cost scales with `unique_gated` (1A) or `n_gate` (1B) rather than
`P`, wall-clock on the example + eval is down, and the relevant validation above passes. Update
`run_example_native.sh` if the production entry point's flags change (see the
`keep-run-example-script-current` memo).
