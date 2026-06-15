"""M5 bridge: rejuvenation on the filtering-sweep's particles, vectorized over particles.

The filtering sweep (:func:`smc_substitution.run_smc_substitution`) carries particles as plain JAX
buffers; the rejuvenation move (:mod:`rejuvenation`) needs an addressable genjax trace. This module
bridges them: it *materializes* a chain trace per particle, runs the R1 MH substitution-flip move,
and writes the (possibly reanalysed) tokens back -- all under a single ``jax.vmap`` over the P
particle axis, so the penzai forward batches over particles (the ~6x-not-Px win that is the point of
the port; Phase 0 spike measured P=64 at 4.3x of P=1). The per-particle Python loop is gone.

Two entry points share one vectorized primitive (:func:`vmapped_window_move`):

- :func:`run_smc_rejuv` -- **post-sweep** (second-pass): run the full sweep, then one windowed move
  over the whole sentence. (Bridge v1.)
- :func:`run_smc_conditional_rejuv` -- **interleaved** (the real SMC rejuvenation): inside the
  every-word loop, after each resample, a per-particle surprisal gate drives a windowed move over a
  lookback window, so early commitments are corrected mid-sweep before the next resample. Mirrors the
  Gen.jl reference's conditional reanalysis (``src/gen_inference.jl``), but vectorized.

**Scope (v1):** single-token observed words, substitution-only (``max_deletions=0``,
``allow_insertion=False``) -- the regime where every particle's chain trace is structurally identical
(addresses ``x0..x{W-1}``, ``o0..o{W-1}``, scalar), so the batched trace is rectangular and vmap
applies. Trans-dimensional add/delete (R2) needs masked/padded ragged traces (Phase 2).

**Graceful degradation (never less capable than the filter).** Rejuvenation is an enhancement on top
of filtering, so an out-of-scope sentence (a multi-token word) is NOT an error: the entry points fall
back to the plain substitution filter and skip the move (``accept_rate = 0.0``). The plain filter
still corrects multi-token substitutions like experimemt->experiment (that is M1, not deletion). Only
the rejuvenation *move* is unavailable for those sentences in v1.

**Correctness keystone.** The move must target the *same* posterior the sweep does. For a single-token
word the sweep's per-branch evidence is ``log_action_prior[COPY] + lm(obs0|ctx)`` (copy) and
``log_action_prior[SUB] + lm(sub_x|ctx) + word_sub_loglik(d)`` (sub) -- exactly the chain model's
``x ~ lm_token`` plus ``o ~ obs_dist(x, cand_xs, cand_ls)`` when the candidate tables are built from
the sweep's word candidates and the particle's action prior (:func:`_word_candidate_tables`). Verified
in ``tests/test_rejuv_bridge.py::test_chain_importance_matches_sweep_evidence``.
"""

import functools
import math
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from genjax import ChoiceMap as C

from . import lm_penzai as L
from . import noise_word as NW
from .tokenizer import decode
from .model import COPY, SUB
from .smc_substitution import run_smc_substitution
from .rejuvenation import make_chain_model, rejuv_step


def _single_token_words(obs_ids):
    """Segment ``obs_ids``, asserting every observed word is a single BPE token (v1 homogeneity).

    Returns ``(words, obs)`` (``noise_word.segment_words`` output + single observed token ids). This
    is the *internal* contract for the homogeneous vmapped primitives; callers gate on
    :func:`_all_single_token` first and fall back to plain filtering, so this should not raise in
    practice. Raises ``ValueError`` if it ever does (a guard against silently producing a ragged
    batched trace).
    """
    words = NW.segment_words([int(i) for i in obs_ids])
    multi = [w for span_ids, w in words if len(span_ids) != 1]
    if multi:
        raise ValueError(f"non-single-token words reached a homogeneous primitive: {multi}")
    return words, [int(span_ids[0]) for span_ids, _ in words]


def _all_single_token(obs_ids):
    """``(all_single_token, n_words)`` -- whether every observed word is a single BPE token.

    The v1 rejuvenation move needs a homogeneous (rectangular) batched trace, which holds only when
    every particle has the same intended-token count. A multi-token observed word breaks that (its
    COPY reading emits n tokens, its SUB reading 1), so such sentences fall back to plain filtering.
    """
    words = NW.segment_words([int(i) for i in obs_ids])
    return all(len(s) == 1 for s, _ in words), len(words)


# Out-of-v1-scope sentences degrade to the plain substitution filter (NEVER an error): rejuvenation
# is an enhancement, so it must never be less capable than `--filter native`. The plain filter still
# handles multi-token substitutions like experimemt->experiment (that is M1, not deletion/insertion).
_SKIP_MSG = ("rejuvenation skipped (outside v1 scope -- single-token words only): {why}. Running the "
             "plain substitution filter instead (still corrects substitutions like "
             "'experimemt'->'experiment'); multi-token rejuvenation is Phase 2 / R2.")


def _plain_fallback(key, obs_ids, num_particles, max_dist, why, kw):
    warnings.warn(_SKIP_MSG.format(why=why), stacklevel=3)
    sents, lm, ess = run_smc_substitution(
        key, obs_ids, num_particles=num_particles, max_dist=max_dist,
        max_deletions=0, allow_insertion=False, **kw)
    return sents, lm, ess, 0.0


def _word_candidate_tables(words, log_action_prior, max_dist):
    """Per-word candidate tables matching the sweep's evidence: ``(cand_xs [W,K], cand_ls [P,W,K])``.

    Column 0 is COPY (the observed token, log-weight ``action_prior[COPY]``); columns ``1..`` are the
    ``word_sub_candidates`` substitutions (log-weight ``action_prior[SUB] + word_sub_loglik(d)``).
    ``cand_xs`` depends only on the words (shared across particles); ``cand_ls`` carries each
    particle's action prior. Padded with dummy id 0 / ``-inf`` (the ``-inf`` makes ``obs_dist`` ignore
    unused slots).
    """
    ap = np.asarray(log_action_prior)                       # [P, 3]
    P = ap.shape[0]
    W = len(words)
    sub_lists = [NW.word_sub_candidates(word_str, max_dist=max_dist) for _, word_str in words]
    K = 1 + max((len(s) for s in sub_lists), default=0)
    cand_xs = np.zeros((W, K), np.int32)
    cand_ls = np.full((P, W, K), -np.inf, np.float32)
    for t, ((span_ids, _), subs) in enumerate(zip(words, sub_lists)):
        cand_xs[t, 0] = int(span_ids[0])
        cand_ls[:, t, 0] = ap[:, COPY]
        for k, (x, d) in enumerate(subs):
            cand_xs[t, 1 + k] = int(x)
            cand_ls[:, t, 1 + k] = ap[:, SUB] + float(NW.word_sub_loglik(d))
    return jnp.asarray(cand_xs), jnp.asarray(cand_ls)


def _positions(w, order):
    """Window-local sweep order. BACKWARD (most-recent-first) is the reanalysis default."""
    base = list(range(w))
    if order == "BACKWARD":
        base = base[::-1]
    elif order != "FORWARD":
        raise ValueError(f"order must be 'FORWARD' or 'BACKWARD' (got {order!r}); "
                         "per-particle SHUFFLE is not vectorized in v1.")
    return tuple(base)


@functools.lru_cache(maxsize=None)
def _window_move_fn(w, n_sweeps, positions):
    """Cached jitted vmapped MH substitution-flip move over a length-``w`` window.

    Returns ``fn(keys[P], x[P,w], buf0[P,M], ilen0, cand_xs[w,K], cand_ls[P,w,K], obs[w]) ->
    (new_x[P,w], accepts[P])``. ``w``/``n_sweeps``/``positions`` are static (baked into the trace);
    everything else is a runtime arg. One compile per distinct window length.
    """
    model = make_chain_model(w)

    def per_particle(key, x_ids, buf0, ilen0, cand_xs, cand_ls, obs):
        ik, sk = jax.random.split(key)
        chm = C.d({**{f"x{t}": x_ids[t] for t in range(w)},
                   **{f"o{t}": obs[t] for t in range(w)}})
        tr, _ = model.importance(ik, chm, (buf0, ilen0, cand_xs, cand_ls))
        acc = jnp.int32(0)
        for _ in range(n_sweeps):
            for k in positions:
                sk, ssk = jax.random.split(sk)
                tr, _, a = rejuv_step(ssk, tr, k, buf0, ilen0, cand_xs, cand_ls)
                acc = acc + a.astype(jnp.int32)
        new_x = jnp.stack([tr.get_choices()[f"x{t}"] for t in range(w)]).astype(jnp.int32)
        return new_x, acc

    vmapped = jax.vmap(per_particle, in_axes=(0, 0, 0, None, None, 0, None))
    return jax.jit(vmapped)


def vmapped_window_move(key, x_window, obs_window, buf0_prefix, ilen0, cand_xs_win, cand_ls_win,
                        positions, n_sweeps):
    """Run the vectorized windowed move. ``positions`` is a tuple of window-local indices.

    ``buf0_prefix [P,M]`` is each particle's buffer seeded with the words *before* the window (rest
    EOS); ``ilen0`` is ``1 + window_start``. Returns ``(new_x [P,w], accepts [P])``.
    """
    P = x_window.shape[0]
    fn = _window_move_fn(int(x_window.shape[1]), int(n_sweeps), tuple(positions))
    keys = jax.random.split(key, P)
    return fn(keys, x_window, buf0_prefix, jnp.int32(ilen0), cand_xs_win, cand_ls_win, obs_window)


def rejuvenate_particles(key, intended_buf, log_action_prior, obs_ids, max_dist=2, n_sweeps=1,
                         order="BACKWARD"):
    """Post-sweep reanalysis of every particle over the whole sentence. Returns
    ``(new_intended_buf, accept_rate)``. One vectorized windowed move (window = full sentence)."""
    words, obs = _single_token_words(obs_ids)
    W = len(words)
    P, M = int(intended_buf.shape[0]), int(intended_buf.shape[1])
    cand_xs, cand_ls = _word_candidate_tables(words, log_action_prior, max_dist)
    buf0_prefix = jnp.full((P, M), L.EOS_ID, jnp.int32)            # nothing committed before word 0
    x_window = intended_buf[:, 1:1 + W]
    obs_window = jnp.asarray(obs, jnp.int32)
    new_x, accs = vmapped_window_move(key, x_window, obs_window, buf0_prefix, 1, cand_xs, cand_ls,
                                      _positions(W, order), n_sweeps)
    new_buf = intended_buf.at[:, 1:1 + W].set(new_x)
    total = P * n_sweeps * W
    return new_buf, (float(jnp.sum(accs)) / total if total else 0.0)


def run_smc_rejuv(key, obs_ids, num_particles=64, max_dist=2, n_sweeps=1, order="BACKWARD", **kw):
    """Filtering sweep (substitution-only) followed by one post-sweep (second-pass) rejuvenation.

    Returns ``(sentences, log_marginal, min_ess, accept_rate)``. ``log_marginal``/``min_ess`` are the
    sweep's (rejuvenation re-decides intended tokens at fixed evidence; it does not re-estimate the
    marginal). A sentence with multi-token words is **not** rejected -- it falls back to the plain
    substitution filter (``accept_rate = 0.0``), never less capable than ``--filter native``. ``**kw``
    flows to ``run_smc_substitution``; do not pass ``max_deletions`` / ``allow_insertion`` (off, v1).
    """
    all_single, W = _all_single_token(obs_ids)
    if not all_single:
        return _plain_fallback(key, obs_ids, num_particles, max_dist, "multi-token word(s)", kw)
    key, sweep_key, rejuv_key = jax.random.split(key, 3)
    _, log_marginal, min_ess, (intended_buf, _, lap) = run_smc_substitution(
        sweep_key, obs_ids, num_particles=num_particles, max_dist=max_dist,
        max_deletions=0, allow_insertion=False, return_state=True, **kw)
    new_buf, accept_rate = rejuvenate_particles(
        rejuv_key, intended_buf, lap, obs_ids, max_dist=max_dist, n_sweeps=n_sweeps, order=order)
    sentences = [decode(new_buf[p, 1:1 + W]).strip() for p in range(num_particles)]
    return sentences, log_marginal, min_ess, accept_rate


def custom_sigmoid(x, center, spread):
    """Gate probability ``sigmoid(spread * (x - center))`` (matches ``gen_inference.jl`` custom_sigmoid).

    Overflow-safe: for large |z| the naive ``1/(1+exp(-z))`` raises on ``exp`` of a huge magnitude.
    """
    z = spread * (x - center)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _make_rejuv_hook(words, obs, cand_xs, max_dist, lookback, center, spread, n_sweeps, order, stats):
    """Build the ``post_resample_hook`` for interleaved conditional rejuvenation.

    Per word ``t``: a per-particle Bernoulli gate (prob ``custom_sigmoid(surprisal, ...)``) selects
    which particles attempt a windowed move over ``[t-lookback, t]``; gated-out / MH-rejected
    particles are unchanged. Accumulates accept stats into ``stats``.
    """
    def hook(t, key, intended_buf, i_len, lap, surprisal):
        P, M = int(intended_buf.shape[0]), int(intended_buf.shape[1])
        p_fire = custom_sigmoid(surprisal, center, spread)
        key, gk = jax.random.split(key)
        gate = jax.random.uniform(gk, (P,)) < p_fire
        n_gate = int(jnp.sum(gate))
        if n_gate == 0:
            return key, intended_buf
        s = max(0, t - lookback)
        w = t - s + 1
        _, cand_ls = _word_candidate_tables(words, lap, max_dist)          # cand_xs is precomputed
        cand_xs_win = cand_xs[s:t + 1]
        cand_ls_win = cand_ls[:, s:t + 1, :]
        col = jnp.arange(M)
        buf0_prefix = jnp.where(col[None, :] < (1 + s), intended_buf, jnp.int32(L.EOS_ID))
        x_window = intended_buf[:, 1 + s:1 + t + 1]
        obs_window = jnp.asarray(obs[s:t + 1], jnp.int32)
        key, mk = jax.random.split(key)
        new_x, accs = vmapped_window_move(mk, x_window, obs_window, buf0_prefix, 1 + s,
                                          cand_xs_win, cand_ls_win, _positions(w, order), n_sweeps)
        new_x = jnp.where(gate[:, None], new_x, x_window)                  # only gated particles move
        cols = 1 + s + jnp.arange(w)
        intended_buf = intended_buf.at[:, cols].set(new_x)
        stats["accepts"] += int(jnp.sum(jnp.where(gate, accs, 0)))
        stats["attempts"] += n_gate * n_sweeps * w
        return key, intended_buf
    return hook


def run_smc_conditional_rejuv(key, obs_ids, num_particles=64, max_dist=2, lookback=4,
                              logprob_thresh=5.0, logprob_spread=1.0, n_sweeps=1,
                              order="BACKWARD", **kw):
    """Filtering sweep with interleaved, surprisal-gated rejuvenation (the real SMC rejuvenation).

    After each word's resample, particles whose Bernoulli gate fires (prob rising with the word's
    surprisal via ``custom_sigmoid(surprisal, logprob_thresh, logprob_spread)``) run a windowed MH
    reanalysis over the last ``lookback`` words, vectorized over particles. Returns
    ``(sentences, log_marginal, min_ess, accept_rate)``. A sentence with multi-token words is **not**
    rejected -- it falls back to the plain substitution filter (``accept_rate = 0.0``), never less
    capable than ``--filter native``. ``**kw`` flows to ``run_smc_substitution``; do not pass
    ``max_deletions`` / ``allow_insertion`` (forced off, v1 scope).
    """
    all_single, _ = _all_single_token(obs_ids)
    if not all_single:
        return _plain_fallback(key, obs_ids, num_particles, max_dist, "multi-token word(s)", kw)
    words, obs = _single_token_words(obs_ids)
    cand_xs, _ = _word_candidate_tables(words, jnp.zeros((1, 3), jnp.float32), max_dist)  # ids only
    stats = {"accepts": 0, "attempts": 0}
    hook = _make_rejuv_hook(words, obs, cand_xs, max_dist, lookback,
                            logprob_thresh, logprob_spread, n_sweeps, order, stats)
    sentences, log_marginal, min_ess = run_smc_substitution(
        key, obs_ids, num_particles=num_particles, max_dist=max_dist,
        max_deletions=0, allow_insertion=False, post_resample_hook=hook, **kw)
    rate = stats["accepts"] / stats["attempts"] if stats["attempts"] else 0.0
    return sentences, log_marginal, min_ess, rate
