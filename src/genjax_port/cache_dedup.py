"""Prefix-dedup LM forward wrapper (prototype) -- the JAX-friendly stand-in for hfppl's
trie cache (see the genjax-port-cache-trie memo).

Idea: the particle filter resamples every step, so right after a resample many particles are
exact copies of the same parent -> identical intended-prefix buffers. The lookahead block is
worse-than-linear (P*K rows), but its K candidate tokens are a deterministic top-K of the
prefix LM logits, so same-prefix particles produce identical K-row blocks too. A transformer
forward treats each batch row independently, so we can:

    1. find the unique rows (keyed on the filled prefix bytes),
    2. run the LM on just those (padded up to a power-of-2 bucket to bound JIT recompiles),
    3. scatter results back to all rows via an inverse index.

This is NUMERICALLY EXACT (not an approximation): a deduped run reproduces the plain filter's
posterior bit-for-bit given the same RNG. The win is wall-clock + compute, which grows with
how degenerate the particle set is (i.e. with resampling pressure and sentence length).

Unlike a persistent trie this keeps no cross-step state, so it sidesteps the resample-permute
cost that made the per-particle KV cache O(T^2); the trade is we recompute shared prefixes
from scratch each step (no KV reuse across steps -- that's the next, harder layer).
"""

import numpy as np
import jax.numpy as jnp

from . import lm_penzai as L


class DedupStats:
    """Counts rows asked-for vs rows actually fed to the LM, to quantify the dedup win
    independently of wall-clock (which on short CPU runs is muddied by JIT recompiles)."""

    def __init__(self):
        self.calls = 0
        self.rows_in = 0        # total rows requested across all calls
        self.rows_computed = 0  # rows actually run through the LM (incl. bucket padding)

    @property
    def saved_frac(self):
        return 0.0 if not self.rows_in else 1.0 - self.rows_computed / self.rows_in

    def __repr__(self):
        return (f"DedupStats(calls={self.calls}, rows_in={self.rows_in}, "
                f"rows_computed={self.rows_computed}, saved={self.saved_frac:.1%})")


# Coarse fixed ladder instead of every power of two: JIT recompiles (one per distinct LM input
# shape) were the dominant overhead, so we want only a handful of shapes. Most steps are very
# degenerate (avg ~12 unique rows), so a small/medium rung covers them; the full size (cap) is
# the fallback for the diverse early steps. Across both call-sites this is ~4 distinct shapes
# total vs ~18 for power-of-two bucketing.
_BUCKET_LADDER = (16, 64)


def _bucket_size(n, cap):
    """Smallest ladder rung >= n (else the full input size `cap`), capped at `cap` so we never
    compute more rows than were asked for."""
    for b in _BUCKET_LADDER:
        if n <= b:
            return min(b, cap)
    return cap


def _dedup_apply(forward_fn, token_bufs, i_lens, stats):
    """Run `forward_fn` (a jitted LM forward [rows, M] x [rows] -> [rows, V]) on only the
    unique rows of (token_bufs, i_lens), keyed on the filled prefix, and scatter back."""
    tb = np.asarray(token_bufs)
    il = np.asarray(i_lens).astype(np.int64)
    n_rows = tb.shape[0]

    slot_of = {}
    reps = []                                   # representative input-row index per unique key
    inverse = np.empty(n_rows, dtype=np.int64)  # for each row, its unique slot
    for r in range(n_rows):
        key = tb[r, :il[r]].tobytes()           # output depends only on the filled prefix
        slot = slot_of.get(key)
        if slot is None:
            slot = len(reps)
            slot_of[key] = slot
            reps.append(r)
        inverse[r] = slot

    U = len(reps)
    Ub = _bucket_size(U, n_rows)
    # Pad the unique set up to the bucket size by repeating the first rep (a valid row, so no
    # NaNs); padded outputs are never indexed by `inverse`.
    rep_idx = np.array(reps + [reps[0]] * (Ub - U), dtype=np.int64)

    out = forward_fn(jnp.asarray(tb[rep_idx]), jnp.asarray(il[rep_idx]))  # [Ub, V]

    if stats is not None:
        stats.calls += 1
        stats.rows_in += n_rows
        stats.rows_computed += Ub

    return out[jnp.asarray(inverse)]            # [n_rows, V]


def make_dedup_fns(stats=None):
    """Return (logprobs_fn, logits_fn) drop-in replacements for L.next_token_{logprobs,logits}
    that dedup their input rows. Pass these into the filter's injectable LM-forward seams
    (run_smc_substitution / run_particle_filter_unified). Pass a DedupStats() to collect ratios."""

    def logprobs(token_bufs, i_lens):
        return _dedup_apply(L.next_token_logprobs, token_bufs, i_lens, stats)

    def logits(token_bufs, i_lens):
        return _dedup_apply(L.next_token_logits, token_bufs, i_lens, stats)

    return logprobs, logits


def make_forward_dedup(forward_fn, stats=None):
    """Wrap an arbitrary batched forward ``(token_bufs [P, M], i_lens [P]) -> [P, ...]`` so it runs
    only on the unique filled-prefix rows (keyed on ``buf[:i_len]``) and scatters the result back.

    EXACT under causal attention: a next-token output depends only on the filled prefix, so byte-equal
    prefixes give byte-equal rows. This is the live-filter drop-in for the dead-stack
    :func:`make_dedup_fns` -- inject it as ``pairhmm_smc.PairHMMModel.lm_fn`` to dedup the SMC forward
    over a post-resample degenerate cloud. The downstream per-particle sampling is untouched (the same
    logits are scattered to duplicate rows, each then sampled with its own RNG key), so the posterior
    is bit-identical given the same RNG -- only redundant LM forwards are removed. Works for any
    forward whose rows are independent (``next_token_logprobs``/``logits``). Pass a ``DedupStats`` to
    quantify the rows-saved ratio."""

    def deduped(token_bufs, i_lens):
        return _dedup_apply(forward_fn, token_bufs, i_lens, stats)

    return deduped
