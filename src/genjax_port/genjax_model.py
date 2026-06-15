"""Genjax-native noisy-channel model -- scaffolding (M0).

This module is the in-repo home for the native ``@gen`` model that replaces the hand-rolled
``particle_filter_unified.py``. As of M0 it holds the two building blocks the de-risking spikes
proved, so M1 can assemble them rather than rediscover them:

1. :func:`make_lm_scan_model` -- the autoregressive *intended-sentence* generator as a ``@gen``
   ``Scan`` kernel (spike 1). The trace auto-records a ``"tok"`` choice per step; ``importance``
   with constrained tokens reproduces the LM chain-rule exactly.
2. :data:`obs_dist` + :func:`token_candidates` -- the substitution **noisy channel** as a
   table-lookup ``exact_density`` over an observed token given an intended token, with the
   candidate set passed as arguments (spike 2). ``importance`` on the fully-constrained joint
   matches the manual joint log-density.

**M1 target (not built here yet):** a *word-scan* model -- per observed word, ``x ~ lm_token``,
``o ~ obs_dist(x, cand_x, cand_l)``, candidates from :func:`noise_word.word_sub_candidates`
passed as **scanned inputs**, plus a custom data-driven proposal ``q`` (local posterior over
candidates). Inference runs as a hand-rolled SMC outer loop calling ``model.importance`` with
``q``-sampled intended tokens (the lean option from ``MIGRATION_PLAN.md`` §6.1), keeping ``@gen``
traces + ``Rejuvenate`` while sidestepping the GenSP ``SampleDistribution`` API surface. The M1
gate is posterior parity with the hand-rolled unified filter on the substitution suite.
"""

import math

import jax
import jax.numpy as jnp
import genjax

from . import lm_penzai as L
from . import noise as N
from .lm_genjax import lm_token, lm_logp

try:
    from genjax import exact_density
except ImportError:
    from genjax._src.generative_functions.distributions.distribution import exact_density


# --- 1. Autoregressive intended-sentence generator (LM prior) -------------------------------

def make_lm_scan_model(T):
    """Return ``(model, M)``: a ``@gen`` ``Scan`` over ``T`` steps and the buffer width ``M``.

    The carry is ``(buf [M], i_len)``; each step samples ``tok ~ lm_token(buf, i_len) @ "tok"``,
    appends it, and advances ``i_len``. Run with::

        model, M = make_lm_scan_model(T)
        init = (jnp.full(M, L.EOS_ID, jnp.int32), jnp.array(1, jnp.int32))
        tr = model.simulate(key, (init, None))          # trace records "tok" per step
        # or, constrained to an observed sentence (LM chain-rule weight):
        vchm = ChoiceMap.empty().at[:, "tok"].set(obs_ids)
        tr, w = model.importance(key, vchm, (init, None))
    """
    M = T + 2

    @genjax.gen
    def kernel(carry, _):
        buf, i_len = carry
        tok = lm_token(buf, i_len) @ "tok"
        buf = buf.at[i_len].set(tok.astype(jnp.int32))
        return (buf, i_len + 1), tok

    return kernel.scan(n=T), M


# --- 2. Substitution noisy channel (table-lookup likelihood) --------------------------------

# Per-token channel split: with probability COPY_P the observed token equals the intended one;
# otherwise it is a form-substitution drawn from the intended token's edit neighborhood. These
# mirror the spike; the word-level filter uses noise_word's word_sub_loglik instead, which M1
# wires in at word granularity.
COPY_LP, SUB_LP = math.log(0.95), math.log(0.05)
KPAD = 8  # fixed candidate-table width (copy + up to KPAD-1 substitutions), -inf padded


def token_candidates(o_id, kpad=KPAD):
    """Fixed-size candidate table for an observed token ``o_id``.

    Returns ``(cand_x [kpad], cand_l [kpad])`` where ``cand_x`` are intended token ids that
    could produce ``o_id`` (copy first, then edit-1 substitutions) and ``cand_l[i] =
    log P(o_id | x = cand_x[i])``. Padded with a dummy id ``0`` and ``-inf`` loglik.
    """
    cand = [(int(o_id), COPY_LP)] + [(x, SUB_LP + ll) for x, ll in N.sub_candidates(int(o_id))]
    cand = cand[:kpad]
    xs = [x for x, _ in cand] + [0] * (kpad - len(cand))
    ls = [l for _, l in cand] + [-jnp.inf] * (kpad - len(cand))
    return jnp.asarray(xs, jnp.int32), jnp.asarray(ls, jnp.float32)


def _obs_sample(key, x, cand_x, cand_l):
    return cand_x[0]  # never used -- the observed token is always data (constrained)


def _obs_logpdf(o, x, cand_x, cand_l):
    """log P(o | x): look x up in the candidate table; -inf if x cannot produce o."""
    match = cand_x == x
    return jnp.where(jnp.any(match), jnp.max(jnp.where(match, cand_l, -jnp.inf)), -jnp.inf)


# P(observed token | intended token), candidates passed as args. Use inside @gen as:
#   o = obs_dist(x, cand_x, cand_l) @ "o"
obs_dist = exact_density(_obs_sample, _obs_logpdf, "obs")


# --- 3. Word model: per-word Switch over candidates (the N:1 substitution representation) -----
#
# A word is N:1 -- COPY of a multi-token observed word emits n intended tokens, SUB emits 1.
# We model each observed word as a Switch over candidate branches:
#   branch 0      = COPY: emits the word's n tokens as live lm_token choices "t0".."t{n-1}"
#   branch 1..S   = SUB : emits 1 intended token "t0"
# Each branch ends with a deterministic channel factor "ch" (action prior + word_sub_loglik for
# subs). The branch index is the addressable action choice -- this is exactly what R1
# rejuvenation (substitution flip) edits. Branches return the threaded (buf, i_len), so words
# compose into a word-scan. Verified in tests/test_word_model.py (importance == manual joint).

# Inject a deterministic log-weight into a trace: observe value 0.0 with logpdf = the weight.
factor = exact_density(lambda key, lw: jnp.float32(0.0), lambda v, lw: lw, "factor")


def _make_copy_branch(n):
    @genjax.gen
    def copy_branch(buf, i_len, ch_lw):
        b, il = buf, i_len
        for j in range(n):
            t = lm_token(b, il) @ f"t{j}"
            b = b.at[il].set(t.astype(jnp.int32))
            il = il + 1
        _ = factor(ch_lw) @ "ch"
        return (b, il)
    return copy_branch


@genjax.gen
def _sub_branch(buf, i_len, ch_lw):
    t = lm_token(buf, i_len) @ "t0"
    b = buf.at[i_len].set(t.astype(jnp.int32))
    _ = factor(ch_lw) @ "ch"
    return (b, i_len + 1)


def make_word_model(n, n_sub):
    """A per-word ``Switch`` generative function: ``[copy(n)] + n_sub * [sub(1)]``.

    Returns the switch gen fn. Call as ``model.importance(key, chm, (idx, copy_args, *sub_args))``
    where each branch's args are ``(buf, i_len, channel_loglik)`` and ``chm`` constrains the
    branch's emitted tokens (``"t0".."t{n-1}"`` for copy, ``"t0"`` for sub) plus ``"ch": 0.0``.
    ``idx`` selects the action (0=copy, k=the k-th substitution). Branch importance weight equals
    the LM chain-rule over the emitted tokens plus the channel log-weight.
    """
    copy_branch = _make_copy_branch(n)
    return copy_branch.switch(*([_sub_branch] * n_sub))


def word_constraints(idx, copy_token_ids, sub_token_id):
    """Choicemap constraining the chosen branch's emitted tokens + the channel factor.

    For ``idx == 0`` (copy) constrain ``t0..t{n-1}`` to ``copy_token_ids``; otherwise constrain
    ``t0`` to ``sub_token_id``. ``"ch"`` is always constrained to ``0.0`` (the factor's support).
    """
    import genjax as _g
    if idx == 0:
        d = {f"t{j}": jnp.int32(t) for j, t in enumerate(copy_token_ids)}
    else:
        d = {"t0": jnp.int32(sub_token_id)}
    d["ch"] = jnp.float32(0.0)
    return _g.ChoiceMap.d(d)
