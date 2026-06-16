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
import genjax
from genjax import ChoiceMap as C

from jax.scipy.special import logsumexp

from . import lm_penzai as L
from . import noise_word as NW
from .unigram import unigram_surprisal
from .tokenizer import decode
from .lm_genjax import lm_logp
from .model import COPY, SUB
from .smc_substitution import run_smc_substitution
from .rejuvenation import make_chain_model, rejuv_step
from .rejuvenation_r2 import (
    make_gap_chain, gap_chain_inputs, add_delete_step, sub_flip_step, _q_logits,
)
from .config import MAX_DELETIONS, P_DELETE_PRIOR, LOOKAHEAD_K


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


# --- R2: post-sweep add/delete (trans-dimensional) reanalysis ---------------------------------
#
# The R1 bridge above edits intended tokens in place at a FIXED alignment (substitution is
# dimension-preserving), so its window writeback is trivial. The R2 add/delete move changes each
# particle's word count, so the buffer<->observed-word alignment becomes per-particle. The
# *post-sweep, whole-sentence* form sidesteps that: the window is the entire sentence (it starts at
# buffer position 1 for every particle), and the output is decoded strings -- so per-particle length
# variation needs no flat-buffer surgery. This mirrors run_smc_rejuv (the R1 post-sweep), and is the
# R2 analog of bridge v1. (Interleaved, mid-sentence windows -- where the start position is
# per-particle once earlier reanalyses have added words -- are the next step; see planning/R2_PLAN.md.)


# NB: do NOT fuse the whole sweep into one jit -- the add/delete step is heavy (a K-row lookahead
# forward + a full-suffix re-score), so unrolling W*n_sweeps of them in a single graph OOMs (XLA holds
# every intermediate). Instead jit *one step per position* (compiles W times, reused across sweeps) and
# loop in Python so memory frees between steps; materialize the batched trace once.

@functools.lru_cache(maxsize=None)
def _materialize_fn(W, p_del):
    """Jitted vmapped materialization of a per-particle gap-chain trace (all gaps off)."""
    model = make_gap_chain(W, p_del)

    def one(key, x_row, buf0, ilen0, cand_xs, cand_ls, obs):
        chm = C.d({**{f"del{t}": jnp.bool_(False) for t in range(W)},
                   **{f"x{t}": x_row[t].astype(jnp.int32) for t in range(W)},
                   **{f"o{t}": obs[t].astype(jnp.int32) for t in range(W)}})
        tr, _ = model.importance(key, chm, (buf0, ilen0, cand_xs, cand_ls))
        return tr

    return jax.jit(jax.vmap(one, in_axes=(0, 0, None, None, None, None, None)))


# The per-position step must compile ONCE and be reused for every word -- else each of the W
# positions compiles a separate heavy graph (the chain re-score + the K-row lookahead, doubled by the
# sub-flip), which on a 7-word sentence is ~150s of pure compile (the static-k step baked `range(k)`
# into the graph, so the graph differed per position). The fix: pass `k` as a RUNTIME value and do the
# context replay + the edit over a fixed `range(W)` masked/selected by `k`. The edit constrains all W
# addresses with only position `k` changed (re-asserted addresses contribute 0 to the update weight,
# and re-asserting the suffix is exactly what lets the trans-dimensional shift re-score it). This is
# the manual SMCP3 of add_delete_step / sub_flip_step, made position-independent. Validated against the
# static-k moves in tests/test_rejuv_bridge.py::test_dyn_step_matches_static.

@functools.lru_cache(maxsize=None)
def _dyn_step_fn(W, lookahead_k, p_del, do_sub):
    """Jitted vmapped per-position move with ``k`` as a runtime arg -- ONE compile for all positions.

    Does an add/delete step at gap ``k`` then (if ``do_sub``) a substitution-flip on word ``k``, on the
    one gap-chain trace. Returns a callable ``(k, keys, trs, buf0, ilen0, cand_xs, cand_ls, obs) ->
    (trs, accepts [P])``."""
    def per_particle(k, key, tr, buf0, ilen0, cand_xs, cand_ls, obs):
        model, args = tr.get_gen_fn(), tr.get_args()   # the trace's OWN gen fn (make_gap_chain makes a
                                                        # fresh closure each call; using another breaks
                                                        # tree_map of the edited vs original trace)
        chm = tr.get_choices()
        dels = jnp.stack([chm[f"del{t}"] for t in range(W)])
        gaps = jnp.stack([chm[f"gap{t}", "xd"].value for t in range(W)]).astype(jnp.int32)
        xs = jnp.stack([chm[f"x{t}"] for t in range(W)]).astype(jnp.int32)

        # context (buf, il) just before gap k: replay words t<k (masked over the fixed range)
        buf, il = buf0, ilen0
        for t in range(W):
            before = t < k
            dt = dels[t] & before
            buf = jnp.where(dt, buf.at[il].set(gaps[t]), buf)
            il = il + dt.astype(jnp.int32)
            buf = jnp.where(before, buf.at[il].set(xs[t]), buf)
            il = il + before.astype(jnp.int32)
        obs_k = obs[k]

        # ---- add/delete at gap k (toggle del{k}; full-trace update selected by t==k) ----
        key, ka, ke, ku = jax.random.split(key, 4)
        del_k, gap_k = dels[k], gaps[k]
        adding = jnp.logical_not(del_k)
        cand_ids, q_logits, logZ = _q_logits(buf, il, obs_k, lookahead_k)   # ONE forward (incl lookahead)
        xprop = cand_ids[jax.random.categorical(ka, q_logits)].astype(jnp.int32)

        def _qlp(x):
            m = cand_ids == x
            return jnp.where(jnp.any(m), jnp.max(jnp.where(m, q_logits, -jnp.inf)) - logZ, -jnp.inf)

        new_gap_k = jnp.where(adding, xprop, gap_k)
        eq = jnp.arange(W) == k
        upd = C.d({**{f"del{t}": jnp.where(eq[t], jnp.logical_not(dels[t]), dels[t]) for t in range(W)},
                   **{f"gap{t}": C.d({"xd": jnp.where(eq[t], new_gap_k, gaps[t])}) for t in range(W)}})
        new_tr, w_upd, _, _ = model.edit(ke, tr, genjax.Update(upd), genjax.Diff.no_change(args))
        s_fwd = jnp.where(adding, _qlp(xprop), 0.0)
        s_bwd = jnp.where(adding, 0.0, _qlp(gap_k))
        acc = jnp.log(jax.random.uniform(ku)) < (w_upd + s_bwd - s_fwd)
        tr = jax.tree_util.tree_map(lambda a, b: jnp.where(acc, a, b), new_tr, tr)
        accepts = acc.astype(jnp.int32)

        if do_sub:
            # re-read (only position k changed; the t<k context above is still valid)
            chm = tr.get_choices()
            dels = jnp.stack([chm[f"del{t}"] for t in range(W)])
            gaps = jnp.stack([chm[f"gap{t}", "xd"].value for t in range(W)]).astype(jnp.int32)
            xs = jnp.stack([chm[f"x{t}"] for t in range(W)]).astype(jnp.int32)
            d_k = dels[k]
            buf_w = jnp.where(d_k, buf.at[il].set(gaps[k]), buf)
            il_w = il + d_k.astype(jnp.int32)
            cx_k, cl_k = cand_xs[k], cand_ls[k]
            key, kp, kse, ksu = jax.random.split(key, 4)
            x_cur = xs[k]
            sc = lm_logp(buf_w, il_w)[cx_k] + cl_k      # ONE forward; q(x) prop LM(x|ctx)*channel
            lz = logsumexp(sc)
            x_new = cx_k[jax.random.categorical(kp, sc)].astype(jnp.int32)

            def _plp(x):
                m = cx_k == x
                return jnp.where(jnp.any(m), jnp.max(jnp.where(m, sc, -jnp.inf)) - lz, -jnp.inf)

            upd_s = C.d({f"x{t}": jnp.where(eq[t], x_new, xs[t]) for t in range(W)})
            ntr2, w_upd2, _, _ = model.edit(kse, tr, genjax.Update(upd_s), genjax.Diff.no_change(args))
            acc2 = jnp.log(jax.random.uniform(ksu)) < (w_upd2 + _plp(x_cur) - _plp(x_new))
            tr = jax.tree_util.tree_map(lambda a, b: jnp.where(acc2, a, b), ntr2, tr)
            accepts = accepts + acc2.astype(jnp.int32)

        return tr, accepts

    vm = jax.vmap(per_particle, in_axes=(None, 0, 0, None, None, None, None, None))
    return jax.jit(vm)


def _gap_choices(trs, W):
    """Read final ``(dels [P,W], gaps [P,W], xs [P,W])`` from a batched gap-chain trace."""
    chm = trs.get_choices()
    dels = jnp.stack([chm[f"del{t}"] for t in range(W)], axis=1)
    gaps = jnp.stack([chm[f"gap{t}", "xd"].value for t in range(W)], axis=1).astype(jnp.int32)
    xs = jnp.stack([chm[f"x{t}"] for t in range(W)], axis=1).astype(jnp.int32)
    return dels, gaps, xs


def _decode_gap_row(dels, gaps, xs, W):
    """Decode one particle's gap chain to a sentence (omitted gap tokens spliced before their word)."""
    ids = []
    for t in range(W):
        if bool(dels[t]):
            ids.append(int(gaps[t]))
        ids.append(int(xs[t]))
    return decode(ids).strip()


def run_smc_add_delete(key, obs_ids, num_particles=64, max_dist=2, n_sweeps=2, order="BACKWARD",
                       lookahead_k=LOOKAHEAD_K, sub_flip=False, **kw):
    """Substitution-only filtering sweep, then a post-sweep add/delete (R2) reanalysis pass.

    REFERENCE / oracle path (``run.py --filter native --add_delete``), NOT the production rejuvenation.
    Per the 2026-06-16 pivot, add/delete reanalysis is deprioritized (the forward filter already does
    add/delete) and this routes the move through the ``@gen`` gap chain (``rejuvenation_r2``), which is
    correct but heavy (W LM forwards per edit). The production rejuvenation is the manual, aligned,
    substitution-only ``run_smc_conditional_rejuv_aligned``. Kept for validation / future revival.

    The forward sweep stays substitution-only; the trans-dimensional move is the *sole* mechanism for
    positing omitted words, but now with the whole sentence as context -- so a dropped word only
    disambiguated late (which the forward 1-step-lookahead deletion gap can miss) is recovered here.
    With ``sub_flip=True`` the post-sweep pass ALSO runs the R1 substitution-flip on each word, so one
    reanalysis pass revises both substitutions and add/deletes (the maximal post-sweep move).
    Returns ``(sentences, log_marginal, min_ess, accept_rate)`` (the marginal/ess are the sweep's; the
    move re-decides the alignment at fixed evidence). A multi-token-word sentence is **not** rejected
    -- it falls back to the native filter (forward deletions + insertion on), never less capable.
    ``**kw`` flows to ``run_smc_substitution`` (do not pass ``max_deletions``/``allow_insertion``).
    """
    all_single, W = _all_single_token(obs_ids)
    if not all_single:
        warnings.warn(_SKIP_MSG.format(why="multi-token word(s)"), stacklevel=2)
        sents, lm, ess = run_smc_substitution(
            key, obs_ids, num_particles=num_particles, max_dist=max_dist,
            max_deletions=MAX_DELETIONS, allow_insertion=True, **kw)
        return sents, lm, ess, 0.0
    key, sweep_key, move_key = jax.random.split(key, 3)
    _, log_marginal, min_ess, (intended_buf, _, _) = run_smc_substitution(
        sweep_key, obs_ids, num_particles=num_particles, max_dist=max_dist,
        max_deletions=0, allow_insertion=False, return_state=True, **kw)
    _, obs, buf0, ilen0, cand_xs, cand_ls = gap_chain_inputs(obs_ids)
    obs_arr = jnp.asarray(obs, jnp.int32)
    x_rows = intended_buf[:, 1:1 + W].astype(jnp.int32)
    p_del = float(P_DELETE_PRIOR)

    move_key, mat_key = jax.random.split(move_key)
    mat_keys = jax.random.split(mat_key, num_particles)
    trs = _materialize_fn(W, p_del)(mat_keys, x_rows, buf0, ilen0, cand_xs, cand_ls, obs_arr)

    step = _dyn_step_fn(W, int(lookahead_k), p_del, bool(sub_flip))  # ONE compile, reused per position
    accepts = 0
    for _ in range(n_sweeps):                                 # Python loop: memory frees per step
        for k in _positions(W, order):
            move_key, sk = jax.random.split(move_key)
            keys = jax.random.split(sk, num_particles)
            trs, acc = step(jnp.int32(k), keys, trs, buf0, ilen0, cand_xs, cand_ls, obs_arr)
            accepts += int(jnp.sum(acc))
    dels, gaps, xs = _gap_choices(trs, W)
    sentences = [_decode_gap_row(dels[p], gaps[p], xs[p], W) for p in range(num_particles)]
    moves_per_pos = 2 if sub_flip else 1                       # add/delete (+ optional sub-flip)
    total = num_particles * n_sweeps * W * moves_per_pos
    accept_rate = accepts / total if total else 0.0
    return sentences, float(log_marginal), min_ess, accept_rate


# --- manual (buffer-based) substitution-flip move: alignment-robust, no @gen trace -------------
#
# This is the R1 substitution-flip done directly on the [P, M] buffer (one forward gives every
# position's logits, like the filtering sweep), addressing a word by its per-particle buffer position
# rather than a fixed 1:1 slot. That is what lets it run AFTER the forward filter has done add/delete
# (which shifts each particle's alignment): the word's token is at ``pos = align[p, t]``, wherever the
# deletions/insertions put it. The MH weight is identical to the @gen move (docs/model.tex Thm 2);
# validated by detailed balance in tests/test_rejuv_bridge.py::test_manual_subflip_detailed_balance.

def manual_subflip_move(key, buf, i_len, pos, cand_x, cand_l, gate):
    """One MH substitution-flip on the token at per-particle position ``pos`` (vectorized over P).

    ``buf [P,M]``, ``i_len [P]``, ``pos [P]`` (the word's token position; ``<1`` or ``>=i_len`` ==
    no token / out of range -> skipped), ``cand_x [K]`` candidate ids, ``cand_l [P,K]`` channel
    logliks (incl. action prior), ``gate [P]`` which particles attempt the move. Returns
    ``(buf, accepted [P])``. Two LM forwards (current + proposed buffer); the prefix cancels."""
    P, M = buf.shape
    rows = jnp.arange(P)
    kq, ku = jax.random.split(key)
    posc = jnp.clip(pos, 1, M - 1)
    valid = gate & (pos >= 1) & (pos < i_len)
    idx = jnp.arange(M)

    def chain_from_pos(lp, b):
        # token at position i is scored by the logits at i-1; sum over the suffix [pos, i_len)
        tok_lp = lp[rows[:, None], idx[None, :] - 1, b]                 # [P, M]
        mask = (idx[None, :] >= posc[:, None]) & (idx[None, :] < i_len[:, None])
        return jnp.sum(jnp.where(mask, tok_lp, 0.0), axis=1)           # [P]

    def chan(x):
        m = cand_x[None, :] == x[:, None]
        return jnp.where(jnp.any(m, 1), jnp.max(jnp.where(m, cand_l, -jnp.inf), 1), -jnp.inf)

    logp_old = jax.nn.log_softmax(L._raw_logits(buf), axis=-1)         # [P, M, V]  (one forward)
    lm_at = logp_old[rows, posc - 1]                                   # [P, V]  dist for the token at pos
    q_logits = lm_at[rows[:, None], cand_x[None, :]] + cand_l          # [P, K]
    logZ = logsumexp(q_logits, axis=1)
    x_cur = buf[rows, posc]
    x_new = cand_x[jax.random.categorical(kq, q_logits)].astype(buf.dtype)

    def qlp(x):
        m = cand_x[None, :] == x[:, None]
        return jnp.where(jnp.any(m, 1), jnp.max(jnp.where(m, q_logits, -jnp.inf), 1) - logZ, -jnp.inf)

    chain_old = chain_from_pos(logp_old, buf)
    buf_new = buf.at[rows, posc].set(x_new)
    logp_new = jax.nn.log_softmax(L._raw_logits(buf_new), axis=-1)     # [P, M, V]  (one forward)
    chain_new = chain_from_pos(logp_new, buf_new)

    w = (chain_new + chan(x_new)) - (chain_old + chan(x_cur)) + qlp(x_cur) - qlp(x_new)
    accept = valid & (jnp.log(jax.random.uniform(ku, (P,))) < w)
    buf = buf.at[rows, posc].set(jnp.where(accept, x_new, x_cur))
    return buf, accept


@functools.lru_cache(maxsize=None)
def _aligned_window_move_fn(nwin, n_sweeps):
    """Cached jitted fused windowed manual sub-flip move (the alignment-robust analog of
    :func:`vmapped_window_move`, but on the flat ``[P, M]`` buffer with per-particle positions).

    Returns ``fn(key, buf[P,M], i_len[P], pos_win[P,nwin], cand_x_win[nwin,K], cand_l_win[P,nwin,K],
    gate[P]) -> (key, buf[P,M], accepts[nwin])`` where ``accepts[j]`` counts accepted moves at window
    column ``j`` summed over gated particles and sweeps (so ``sum(accepts)`` is the total accepted
    moves, the quantity the aggregate accept-rate uses, and the per-column breakdown feeds the
    structured ``rejuv_events`` log; column ``j`` corresponds to the caller's ``win[j]``).
    The ``n_sweeps x nwin`` single sub-flips run as ONE compiled graph (static unrolled loop), so the
    old per-word Python loop -- whose ~2 un-fused eager forwards per move made the path exec-bound --
    is gone; XLA fuses the gathers and the suffix re-score across the window. Columns of
    ``pos_win``/``cand_*_win`` are pre-ordered in sweep order, so the ``range(nwin)`` walk is in order.
    ``key`` is split per move and returned, exactly replicating the old per-word loop's RNG stream (so
    this is a pure perf change). One compile per distinct ``(nwin, n_sweeps)`` (K folded in via shapes)."""
    def body(key, buf, i_len, pos_win, cand_x_win, cand_l_win, gate):
        accs = jnp.zeros((nwin,), jnp.int32)
        for _ in range(n_sweeps):
            for j in range(nwin):
                key, mk = jax.random.split(key)
                buf, acc = manual_subflip_move(
                    mk, buf, i_len, pos_win[:, j], cand_x_win[j], cand_l_win[:, j], gate)
                accs = accs.at[j].add(jnp.sum(jnp.where(gate, acc, 0)))
        return key, buf, accs

    return jax.jit(body)


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

    Per word ``t``: a per-particle Bernoulli gate (prob
    ``custom_sigmoid(surprisal - unigram_surp[t], ...)``) selects which particles attempt a windowed
    move over ``[t-lookback, t]``; gated-out / MH-rejected particles are unchanged. The gate input is
    contextual surprisal minus the word's unigram surprisal, so it fires on words that are more
    surprising in context than out of it, not on merely rare words. Accumulates accept stats into
    ``stats``.
    """
    unigram_surp = [unigram_surprisal(surf) for _, surf in words]  # unigram-relative gate (Gen.jl)

    def hook(t, key, intended_buf, i_len, align, lap, surprisal):  # align unused (sub-only-forward v1)
        P, M = int(intended_buf.shape[0]), int(intended_buf.shape[1])
        p_fire = custom_sigmoid(surprisal - unigram_surp[t], center, spread)
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


def run_smc_conditional_rejuv(key, obs_ids, num_particles=64, max_dist=2, lookback=2,
                              logprob_thresh=0.0, logprob_spread=1.0, n_sweeps=1,
                              order="BACKWARD", **kw):
    """Filtering sweep with interleaved, surprisal-gated rejuvenation (the real SMC rejuvenation).

    After each word's resample, particles whose Bernoulli gate fires (prob rising with the word's
    unigram-relative surprisal via
    ``custom_sigmoid(surprisal - unigram_surp, logprob_thresh, logprob_spread)``) run a windowed MH
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


# --- aligned conditional rejuvenation: forward filter does sub+add/delete, interleaved SUB-flip -----
#
# Unlike run_smc_conditional_rejuv (which forces the forward sweep sub-only so its @gen window move
# can assume position == 1 + word index), this version lets the forward filter do deletions AND
# insertions, and rejuvenates substitutions via the alignment-robust manual_subflip_move: each word's
# token is found at its per-particle align[p, t], so deletions/insertions shifting the buffer don't
# break it. The move is shape-preserving (it only re-decides existing tokens), so no trans-dimensional
# machinery is needed -- the add/delete capability lives entirely in the forward filter. Multi-token
# words are simply skipped by the rejuvenation (the forward filter still handles them).

def _make_aligned_subflip_hook(words, max_dist, lookback, center, spread, n_sweeps, order, stats,
                               record=None):
    """``post_resample_hook`` running surprisal-gated manual sub-flips over a lookback window, using
    the per-particle ``align`` to locate each (single-token) observed word's token in the buffer.

    The gate fires on contextual surprisal *minus* the word's unigram surprisal (Gen.jl), so a
    legitimately-rare literal is not reanalysed just for being rare -- only words more surprising in
    context than their base rate predicts. When ``record`` is given, each firing event appends one
    ``record["rejuv_events"]`` entry with the gate fire-prob, the count of gated particles, and the
    per-target-word attempts/accepts (the structured ``--output_json`` rejuvenation log)."""
    single = [len(span) == 1 for span, _ in words]
    # Gate on contextual surprisal RELATIVE to the word's unigram surprisal (Gen.jl): fire only when
    # a word is more surprising in context than its base rate predicts. Per-word scalar, precomputed.
    unigram_surp = [unigram_surprisal(surf) for _, surf in words]

    def hook(t, key, intended_buf, i_len, align, lap, surprisal):
        P = int(intended_buf.shape[0])
        p_fire = custom_sigmoid(surprisal - unigram_surp[t], center, spread)
        key, gk = jax.random.split(key)
        gate = jax.random.uniform(gk, (P,)) < p_fire
        if int(jnp.sum(gate)) == 0:
            return key, intended_buf
        s = max(0, t - lookback)
        win = [w for w in range(s, t + 1) if single[w]]                    # only single-token words
        if order == "BACKWARD":
            win = win[::-1]
        if not win:
            return key, intended_buf
        cand_xs, cand_ls = _word_candidate_tables(words, lap, max_dist)    # [W,K], [P,W,K]
        win_idx = jnp.asarray(win, jnp.int32)
        fn = _aligned_window_move_fn(len(win), n_sweeps)
        key, intended_buf, accs = fn(key, intended_buf, i_len, align[:, win_idx],
                                     cand_xs[win_idx], cand_ls[:, win_idx], gate)
        accs = np.asarray(accs)                                            # accs[j] <-> win[j]
        n_gate = int(jnp.sum(gate))
        stats["accepts"] += int(accs.sum())
        stats["attempts"] += n_gate * n_sweeps * len(win)
        if record is not None:
            record["rejuv_events"].append({
                "t": t, "gate_p": float(p_fire), "fired_particles": n_gate,
                "targets": [{"word": int(win[j]), "attempts": n_gate * n_sweeps,
                             "accepts": int(accs[j])} for j in range(len(win))],
            })
        return key, intended_buf

    return hook


def run_smc_conditional_rejuv_aligned(key, obs_ids, num_particles=64, max_dist=2, lookback=2,
                                      logprob_thresh=0.0, logprob_spread=1.0, n_sweeps=1,
                                      order="BACKWARD", max_deletions=MAX_DELETIONS,
                                      allow_insertion=True, record=None, record_topk=5, **kw):
    """Forward filter with substitution + add/delete, plus interleaved surprisal-gated SUBSTITUTION
    rejuvenation (alignment-robust, vectorized over particles).

    The forward sweep does copy/substitution/deletion/insertion as usual; after each word's resample,
    gated particles run manual sub-flip MH moves over a lookback window, locating each word's token via
    the per-particle alignment so the deletions/insertions don't misalign the move. The rejuvenation is
    substitution-only (shape-preserving) -- add/delete reanalysis is intentionally not done here; that
    capability is the forward filter's. Works on any sentence (multi-token words are handled by the
    forward filter and skipped by the rejuvenation). Returns
    ``(sentences, log_marginal, min_ess, accept_rate)``. ``**kw`` flows to ``run_smc_substitution``.
    ``record`` (default ``None``): when given, fills per-word diagnostics (via ``run_smc_substitution``)
    and one ``rejuv_events`` entry per firing event for the ``--output_json`` artifact."""
    words = NW.segment_words([int(i) for i in obs_ids])
    stats = {"accepts": 0, "attempts": 0}
    hook = _make_aligned_subflip_hook(words, max_dist, lookback, logprob_thresh, logprob_spread,
                                      n_sweeps, order, stats, record=record)
    sentences, log_marginal, min_ess = run_smc_substitution(
        key, obs_ids, num_particles=num_particles, max_dist=max_dist,
        max_deletions=max_deletions, allow_insertion=allow_insertion, post_resample_hook=hook,
        record=record, record_topk=record_topk, **kw)
    rate = stats["accepts"] / stats["attempts"] if stats["attempts"] else 0.0
    return sentences, log_marginal, min_ess, rate
