"""R1: capacity-parametric Gibbs rejuvenation sweep over intended words, with genjax SMCP3 weights.

ADDITIVE to (never edits) the certified ``pairhmm_smc`` forward filter -- it imports the same
building blocks (``_word_row_update``, the model's ``lm_fn`` / channel / candidate injections) and
re-derives the channel table from ``(observed, model)``, so the certified path is untouched.

A sweep resamples one intended *word* at a time from its full conditional

    P(word_w = x | other words, observed)
        ∝  LM_prior(sentence with x at w)  ·  channel_marginal(observed | sentence with x at w)

over a candidate set (toy: the whole vocab -> a true Gibbs step; Pythia: COPY + SymSpell pool). The
two factors are scored by exactly the pieces the exact-enumeration test uses -- the autoregressive
``lm_fn`` for the LM prior, the forward DP ``_word_row_update`` -> ``alpha[M]`` for the channel
marginal -- so sampling from the per-candidate scores IS the conditional. Applied to an EQUALLY-
WEIGHTED cloud (post-resample), this is a Gibbs move: it leaves the posterior invariant and restores
the diversity that resampling collapses (the ``cat/mat`` impoverishment from planning/kv_cache_spikes/).

Two things are NEW vs. R0 (plan REJUV_KV_REDESIGN_PLAN.md, phase R1):

 1. **Capacity-parametric representation (`T_max`)**, per plan §0.1. A word is a bounded *token span*
    (capacity ``T_max``), not a single token. The state is unpacked into per-word slots
    ``word_tok [P, Wmax, T_max]`` / ``word_len [P, Wmax]``; a move replaces a word's span by a
    fixed-shape slot ``.set`` and the flat LM buffer is rebuilt by a **cumsum/scatter re-pack**
    (:func:`_pack`) that handles unequal spans. The LM prior loops over *token* positions (chain-rule
    product over a word's tokens); the channel DP loops over *word* slots (one row per word, surface-
    based -- token-count-agnostic). We RUN at ``T_max = 1`` (the filter is single-token until Phase D
    of PAIRHMM_RBSMC_PLAN.md), but nothing hard-codes a single-index swap: ``T_max = 1`` falls out of
    the general code (``word_tok[..., 0]``, ``_pack`` with all lengths 1, one token per LM step).

 2. **genjax SMCP3 weights.** The move's reweight is produced by genjax via the ``Rejuvenate`` SMCP3
    recipe (propose from the full conditional, ``Update`` the trace, assess the reverse), inlined as
    :func:`_smcp3_move` so per-particle proposal scores can be threaded (the ``Rejuvenate`` class's
    ``argument_mapping(chm)``-only signature can't, and its proposal would re-address the selected
    address -- confirmed in planning/kv_cache_spikes/rejuv_smcp3_spike.py). For a full-conditional
    proposal the SMCP3 weight ``w + bwd − fwd`` is ≈ 0 (asserted as a built-it-right check); for an
    asymmetric candidate set it carries real mass into the next resample (REJUV_GOAL3). ``gibbs_sweep``
    returns ``(ctx_buf, move_logw)``; the caller folds ``move_logw`` into ``log_w`` BEFORE resampling.

Still deferred to R3: the KV-cache (the LM prior is a full O(T) re-score of the sentence per
candidate here -- correctness/shape first). Certified on the toy by exact enumeration in
``tests/test_pairhmm_exact.py``.
"""

import functools
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import genjax
from genjax import ChoiceMap, Update, Diff

from genjax_port.genjax_factor import factor
from genjax_port.word_dp import _word_row_update, channel_carry


@dataclass
class RejuvCtx:
    """Everything a sweep needs, derived once from (observed, model). Mirrors the setup inside
    ``pairhmm_smc.run`` but lives here so the certified filter is not touched."""
    model: object
    emit_full: jnp.ndarray   # (M, Vc) channel logpdf of each candidate surface vs each observed word
    a0: jnp.ndarray          # (M+1,) initial forward vector (leading spurious words)
    M: int                   # number of observed words
    seed_len: int            # context-seed length (toy 0; Pythia [EOS]+prime)
    Wmax: int                # max intended words = M + slack
    wdel: float
    wins: object             # spurious-word log-cost: scalar (uniform) OR (M,) per-observed-word vector
    band: Optional[int]      # |k - t| <= band; None = no band (matches exact enumeration)
    t_max: int = 1           # capacity: max tokens per intended word (run at 1 until Phase D)
    lm_temp: float = 1.0     # LM-prior temperature lambda: the move targets P_LM^lm_temp * P_channel
    #                          (multiplies the LM suffix `chain` only, not the channel marginal `chan`).
    #                          1.0 = untempered (certified); < 1 flattens the LM's over-confident
    #                          preferences so plausible inputs are read more literally (less over-editing).


def make_rejuv_ctx(observed, model, wdel, wins, band=None, slack=3, t_max=1, lm_temp=1.0):
    obs_words = model.obs_words(observed)
    M = len(obs_words)
    obs_char = jnp.stack([jnp.asarray(model.char_ids(w)[0], jnp.int32) for w in obs_words])
    emit_full = jax.vmap(jax.vmap(model.channel_logpdf, in_axes=(None, 0, 0)),
                         in_axes=(0, None, None))(obs_char, model.vocab_char, model.vocab_clen)
    ks = jnp.arange(M + 1)
    wins_arr = jnp.asarray(wins)                                    # scalar (uniform) or (M,) per-word
    if wins_arr.ndim == 0:
        a0 = jnp.where(ks == 0, 0.0, ks * wins)
    else:                                                          # a0[k] = first-k insertion-cost sum
        a0 = jnp.concatenate([jnp.zeros((1,), wins_arr.dtype), jnp.cumsum(wins_arr)])
    if band is not None:                                            # match the filter's band_mask(.,0)
        a0 = jnp.where(jnp.abs(ks) <= band, a0, -jnp.inf)
    return RejuvCtx(model, emit_full, a0, M, len(model.seed_ids), M + slack, wdel, wins, band, t_max,
                    lm_temp)


def _band_mask_a(band, M, alpha, t):
    if band is None:
        return alpha
    ks = jnp.arange(M + 1)
    return jnp.where(jnp.abs(ks - t) <= band, alpha, -jnp.inf)


def _band_mask(ctx, alpha, t):
    return _band_mask_a(ctx.band, ctx.M, alpha, t)


# --------------------------------------------------------------------------------------------------
# Capacity-parametric buffer machinery (T_max). _pack is the cumsum-scatter re-pack of variable
# per-word token spans into a contiguous LM buffer; at T_max=1 (all lengths 1) it is the identity
# map word w -> position seed_len + w, but the SAME code handles unequal spans (Phase D / R4).
# --------------------------------------------------------------------------------------------------
def _pack(word_tok, word_len, n_out):
    """Pack per-word token spans into a contiguous prefix. ``word_tok`` (N, Wmax, T_max), ``word_len``
    (N, Wmax) token count per word. Returns ``(packed [N, n_out], total [N])`` where ``packed[:, p]``
    is the p-th token of ``concat(word w's first word_len[w] tokens for w)`` and ``total`` the token
    count. Positions past ``total`` are left as ``word_tok``'s pad (caller masks). Gather-by-cumsum,
    collision-free and fixed-shape; reduces to ``packed[:, w] = word_tok[:, w, 0]`` when all lens 1."""
    N, Wmax, T = word_tok.shape
    cum = jnp.cumsum(word_len, axis=1)                              # (N, Wmax) inclusive token counts
    start = cum - word_len                                          # (N, Wmax) exclusive start per word
    total = cum[:, -1]                                             # (N,)
    rel = jnp.arange(n_out)                                        # (n_out,) output token positions
    w = jnp.clip(jnp.sum(cum[:, None, :] <= rel[None, :, None], axis=2), 0, Wmax - 1)  # owning word
    j = jnp.clip(rel[None, :] - jnp.take_along_axis(start, w, axis=1), 0, T - 1)       # offset in word
    tok = jnp.take_along_axis(word_tok.reshape(N, Wmax * T), w * T + j, axis=1)        # (N, n_out)
    return tok, total


def _unpack(ctx_buf, word_len, sl, Wmax, t_max):
    """Gather per-word token spans from the flat buffer using the forward filter's ``word_len``
    boundaries (R4). A multi-token buffer has no recoverable word boundaries on its own, so the
    forward state carries ``word_len`` (token count per word) and ``word_surf`` (channel surface id
    per word); this reconstructs ``word_tok [P, Wmax, t_max]`` (pad 0) by cumsum boundaries and
    ``n_words [P]`` (count of non-empty words). At ``t_max == 1`` with all lengths 1 it reduces to the
    old single-token unpack (word ``w`` == token ``ctx_buf[:, sl+w]``)."""
    P, LCTX = ctx_buf.shape
    cum = jnp.cumsum(word_len, axis=1)                            # (P, Wmax) inclusive token counts
    start = cum - word_len                                        # (P, Wmax) exclusive start (rel to sl)
    j = jnp.arange(t_max)
    pos = jnp.clip(sl + start[:, :, None] + j[None, None, :], 0, LCTX - 1)   # (P, Wmax, t_max)
    tok = jnp.take_along_axis(ctx_buf, pos.reshape(P, Wmax * t_max), axis=1).reshape(P, Wmax, t_max)
    word_tok = jnp.where(j[None, None, :] < word_len[:, :, None], tok, 0)
    n_words = jnp.sum(word_len > 0, axis=1)
    return word_tok, n_words


def _flat_buffer_a(eos_id, seed_ids, sl, word_tok, word_len, LCTX, n_out):
    """Pack per-word spans (+ seed) into a flat LM buffer [N, LCTX]. Returns (bufs, total_tokens).
    ``seed_ids`` is the (sl,) seed-token array (empty when sl==0); ``eos_id``/``sl`` are static."""
    N = word_tok.shape[0]
    packed, total = _pack(word_tok, word_len, n_out)              # (N, n_out)
    bufs = jnp.full((N, LCTX), eos_id, jnp.int32)
    if sl:
        bufs = bufs.at[:, :sl].set(seed_ids.astype(jnp.int32))
    valid = jnp.arange(n_out)[None, :] < total[:, None]
    bufs = bufs.at[:, sl:sl + n_out].set(jnp.where(valid, packed, eos_id).astype(jnp.int32))
    return bufs, total


def _flat_buffer(ctx, word_tok, word_len, LCTX, n_out):
    sl = ctx.seed_len
    seed_ids = jnp.asarray(ctx.model.seed_ids, jnp.int32) if sl else jnp.zeros((0,), jnp.int32)
    return _flat_buffer_a(ctx.model.eos_id, seed_ids, sl, word_tok, word_len, LCTX, n_out)


def _lm_logprior(ctx, bufs, total_tokens, add_eos):
    """(N,) LM prior = sum_t log P(token_t | seed + tokens[:t]) [+ log P(EOS | all tokens) where
    ``add_eos``]. Loops over TOKEN positions (chain-rule), so a multi-token word contributes the
    product over its tokens; at T_max=1 this is one term per word. ``add_eos [N]`` is the EOS term --
    on for DONE (complete-hypothesis) particles, off for partial mid-loop ones (no EOS yet). Same
    quantity ``_joint_batch``'s ``lm`` computes, driven through the injected ``lm_fn``."""
    model = ctx.model
    sl, LCTX, N = ctx.seed_len, bufs.shape[1], bufs.shape[0]
    n_out = ctx.Wmax * ctx.t_max
    total = jnp.zeros(N)
    for t in range(n_out + 1):
        lp = model.lm_fn(bufs, jnp.full((N,), sl + t, jnp.int32))  # (N, vocab) next-token logprobs
        tok = bufs[:, min(sl + t, LCTX - 1)]
        tok_lp = jnp.take_along_axis(lp, tok[:, None], axis=1)[:, 0]
        eos_lp = lp[:, model.eos_id]
        total = total + jnp.where(t < total_tokens, tok_lp,
                                  jnp.where((t == total_tokens) & add_eos, eos_lp, 0.0))
    return total


# The channel forward carry is ``word_dp.channel_carry`` (one source of truth shared with the forward
# filter's theta-refresh). Both the sweep's per-candidate scorer (``_chan_scores``) and its final
# ``log_alpha`` recompute call it with the per-particle word-action costs; the char-copy / OFF path
# passes the zero-action parameterization (``lp_copy=lp_sub=0``, global ``wdel``/``wins``, all-zero
# ``copy_mask``) the sweep builds from ``ctx`` -- bit-identical to the pre-word-action carry.


def _tail_chain_uncached(lm_fn, ctx_bufs, ctx_lens, tails, tail_lens):
    """Generic uncached chain-rule ``log P(tail | ctx[:ctx_len])`` for ``[B, K]`` candidates (tail =
    the continuation after ctx), via the injected single-position ``lm_fn``. Returns ``[B, K]``. This
    is the toy / default scorer and the correctness reference; Pythia injects a KV-cached equivalent
    (``lm_penzai.batch_tail_logprobs``) that shares the prefill across candidates -- the R3 perf win."""
    B, K, w = tails.shape
    L = ctx_bufs.shape[1]
    flat_ctx = jnp.repeat(ctx_bufs, K, axis=0)                     # (B*K, L)
    flat_ilen = jnp.repeat(ctx_lens, K)
    flat_tails = tails.reshape(B * K, w)
    flat_lens = tail_lens.reshape(B * K)
    rows = jnp.arange(B * K)

    def one_step(i, carry):
        sc, buf, ilen = carry
        active = i < flat_lens
        lp = lm_fn(buf, ilen)                                      # (B*K, vocab)
        tok = flat_tails[:, i]
        sc = sc + jnp.where(active, jnp.take_along_axis(lp, tok[:, None], axis=1)[:, 0], 0.0)
        wpos = jnp.clip(ilen, 0, L - 1)
        buf = buf.at[rows, wpos].set(jnp.where(active, tok, buf[rows, wpos]))
        ilen = ilen + active.astype(jnp.int32)
        return sc, buf, ilen

    sc, _, _ = jax.lax.fori_loop(0, w, one_step, (jnp.zeros(B * K), flat_ctx, flat_ilen))
    return sc.reshape(B, K)


# --------------------------------------------------------------------------------------------------
# genjax SMCP3 move. Inlines ``Rejuvenate.edit``'s recipe (propose from the full conditional, Update
# the trace, assess the reverse) over a one-variable model whose log-density is the candidate target
# ``target_lp[s]``. For a full-conditional proposal the weight ``w + bwd − fwd`` is ≈ 0; genjax owns
# the bookkeeping so we never hand-derive the ratio. Vmapped over P (per-particle ``target_lp``).
# --------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _slot_gf(K):
    """A one-variable model + its full-conditional proposal over K candidate slots. The model's joint
    log-density at (slot=s, ev=0) is ``target_lp[s]`` up to the constant ``-log K`` -- so an Update
    of ``slot`` reweights by ``target_lp[s'] − target_lp[s]`` and the SMCP3 weight collapses to 0
    when the proposal is the exact conditional ``softmax(target_lp)``."""
    @genjax.gen
    def slot_model(target_lp):
        s = genjax.categorical(jnp.zeros((K,))) @ "slot"
        _ = factor(target_lp[s]) @ "ev"
        return s

    @genjax.gen
    def slot_proposal(target_lp):
        s = genjax.categorical(target_lp) @ "slot"
        return s

    return slot_model, slot_proposal


def _smcp3_move(slot_model, slot_proposal, keys, target_lp, s_cur):
    """Per-particle SMCP3 (Rejuvenate) move. Returns (s_new [P], weight [P]). ``weight = w + bwd − fwd``
    per genjax: w = model density ratio from ``Update``, fwd/bwd = proposal logprob of the new/old slot."""
    def one(key, tlp, sc):
        k1, k2, k3 = jax.random.split(key, 3)
        tr, _ = slot_model.importance(k1, ChoiceMap.d({"slot": sc, "ev": jnp.float32(0.0)}), (tlp,))
        proposed, fwd, _ = slot_proposal.propose(k2, (tlp,))
        new_tr, w, _, bwd_req = Update(proposed).edit(k3, tr, (Diff.no_change(tlp),))
        bwd, _ = slot_proposal.assess(bwd_req.constraint, (tlp,))
        return new_tr.get_choices()["slot"], w + bwd - fwd

    return jax.vmap(one)(keys, target_lp, s_cur)


# --------------------------------------------------------------------------------------------------
# Per-word move, factored into pure (un-jitted) helpers so the FUSED step (tail_fn inside the jit) and
# the SPLIT dedup steps (tail_fn lifted to the host) share IDENTICAL logic -- no drift between paths.
# The split exists for R3 item 1b: post-resample the cloud is ~93% duplicate buffers, so the LM
# tail_fn (KV prefill + K tails) is run on the unique buffers only and the [P,Kt] scores scattered
# back; the per-particle SMCP3 sample still runs on all P (duplicates must DIVERGE -- that is the
# diversification the dedup must never collapse). See planning/REJUV_KV_REDESIGN_PLAN.md R3 item (1).
# --------------------------------------------------------------------------------------------------
def _candidates(w, word_tok, word_len, word_surf, pool_tok, pool_len, pool_surf, K, T):
    """[COPY (current word at slot w)] ++ pool[w] -> (cand_tok [P,Kt,T], cand_len [P,Kt],
    cand_surf [P,Kt], valid [P,Kt]). Cheap (no LM); pool pads are invalid and pool entries equal to
    COPY are de-duplicated (COPY kept at index 0). The pool carries each candidate's real channel
    surface id (``pool_surf``, R4) -- NOT the first token -- so a multi-token candidate's channel
    column is the surface of its whole word, and the COPY uses the current word's stored surface id."""
    P = word_tok.shape[0]
    cur_tok, cur_len, cur_surf = word_tok[:, w, :], word_len[:, w], word_surf[:, w]
    pt, pl, ps = pool_tok[w], pool_len[w], pool_surf[w]                    # (K,T),(K,),(K,)
    cand_tok = jnp.concatenate(
        [cur_tok[:, None, :], jnp.broadcast_to(pt[None], (P, K, T))], axis=1)       # (P,Kt,T)
    cand_len = jnp.concatenate([cur_len[:, None], jnp.broadcast_to(pl[None], (P, K))], axis=1)
    cand_surf = jnp.concatenate([cur_surf[:, None], jnp.broadcast_to(ps[None], (P, K))], axis=1)
    dup = ps[None, :] == cur_surf[:, None]                                          # (P,K)
    valid = jnp.concatenate([jnp.ones((P, 1), bool), (pl[None] > 0) & ~dup], axis=1)
    return cand_tok, cand_len, cand_surf, valid


def _chan_scores(w, word_len, word_surf, cand_len, cand_surf, done,
                 a0p, emit_full, wdel_p, wins_p, band, M, Wmax, lp_copy, lp_sub, copy_mask):
    """Done-aware channel marginal per candidate (P,Kt): splice each candidate into slot w, run the
    word forward DP with the PER-PARTICLE word-action costs. DONE particles read the terminal
    ``alpha[M]``; mid-loop ones the partial forward mass ``logsumexp(alpha)``.

    The per-particle costs (``a0p`` (P,M+1), ``lp_copy``/``lp_sub``/``wdel_p`` (P,), ``wins_p`` (P,M))
    are ``jnp.repeat``-ed by ``Kt`` to align with the P*Kt spliced rows and handed to the SHARED
    ``word_dp.channel_carry`` -- the same carry the forward filter's theta-refresh uses, so the sweep
    scores every candidate against the live per-particle channel (theta). The char-copy / OFF path
    passes the zero-action parameterization -> bit-identical to the pre-word-action carry."""
    P, Kt = cand_surf.shape
    wl = jnp.broadcast_to(word_len[:, None], (P, Kt, Wmax)).at[:, :, w].set(cand_len)
    ws = jnp.broadcast_to(word_surf[:, None], (P, Kt, Wmax)).at[:, :, w].set(cand_surf)
    N = P * Kt
    rep = lambda a: jnp.repeat(a, Kt, axis=0)                       # align (P, ...) -> (P*Kt, ...)
    carry = channel_carry(rep(a0p), emit_full, band, M, ws.reshape(N, Wmax), wl.reshape(N, Wmax),
                          rep(lp_copy), rep(lp_sub), rep(wdel_p), rep(wins_p), copy_mask)   # (N, M+1)
    return jnp.where(jnp.repeat(done, Kt), carry[:, M], logsumexp(carry, axis=1)).reshape(P, Kt)


def _tail_inputs(w, word_tok, word_len, cand_tok, cand_len, done, n_words,
                 sl, Wmax, T, mt, eos_id, seed_ids):
    """``tail_fn`` inputs (ctx_bufs [P,LCTX], ctx_lens [P], tail [P,Kt,mt], tail_len [P,Kt]). ``tail_fn``
    prefills the prefix (words < w; cancels across candidates) and scores the candidate-dependent
    ``tail = [candidate span, suffix word spans w+1.., EOS?]`` in TOKENS (R4 multi-token). Built by
    splicing each candidate's span into slot w, packing the whole buffer (``_pack``), and slicing the
    suffix tokens starting at the prefix length; an EOS is appended for DONE particles. At ``T == 1``
    with unit lengths this reduces to the old one-token-per-word tail."""
    P, Kt = cand_len.shape
    n_out = Wmax * T
    LCTX = sl + n_out + 1
    cum = jnp.cumsum(word_len, axis=1)                                              # (P,Wmax) inclusive
    prefix_tok = jnp.where(w > 0, cum[:, jnp.clip(w - 1, 0, Wmax - 1)], 0)          # tokens of words < w
    ctx_bufs, _ = _flat_buffer_a(eos_id, seed_ids, sl, word_tok, word_len, LCTX, n_out)
    ctx_lens = jnp.full((P,), sl, jnp.int32) + prefix_tok.astype(jnp.int32)          # prefix end

    wt = jnp.broadcast_to(word_tok[:, None], (P, Kt, Wmax, T)).at[:, :, w, :].set(cand_tok)
    wl = jnp.broadcast_to(word_len[:, None], (P, Kt, Wmax)).at[:, :, w].set(cand_len)
    packed, total = _pack(wt.reshape(P * Kt, Wmax, T), wl.reshape(P * Kt, Wmax), n_out)
    packed = packed.reshape(P, Kt, n_out)
    total = total.reshape(P, Kt)
    rel = jnp.arange(mt)
    gpos = jnp.clip(prefix_tok[:, None, None] + rel[None, None, :], 0, n_out - 1)    # (P,1,mt)
    tail = jnp.take_along_axis(packed, jnp.broadcast_to(gpos, (P, Kt, mt)), axis=2)  # (P,Kt,mt)
    tail_tok = (total - prefix_tok[:, None]).astype(jnp.int32)                       # [cand+suffix] tokens
    eos_at = jnp.clip(tail_tok, 0, mt - 1)
    p_idx, k_idx = jnp.arange(P)[:, None], jnp.arange(Kt)[None, :]
    tail = tail.at[p_idx, k_idx, eos_at].set(                                        # EOS for done particles
        jnp.where(done[:, None], jnp.int32(eos_id), tail[p_idx, k_idx, eos_at]))
    tail_len = jnp.clip(tail_tok + done[:, None].astype(jnp.int32), 0, mt)
    return ctx_bufs, ctx_lens, tail, tail_len


def _apply_move(key, target, w, cand_tok, cand_len, cand_surf, word_tok, word_len, word_surf,
                move_logw, n_words, slot_model, slot_proposal):
    """Sample the per-particle SMCP3 move from ``target`` (P,Kt) and splice the chosen candidate into
    slot w; accumulate the SMCP3 weight. The ``split(key, P)`` makes duplicate particles DIVERGE -- the
    diversification dedup must leave untouched (it only dedups the deterministic ``target`` upstream)."""
    P = word_tok.shape[0]
    cur_tok, cur_len, cur_surf = word_tok[:, w, :], word_len[:, w], word_surf[:, w]
    keys = jax.random.split(key, P)
    s_new, weight = _smcp3_move(slot_model, slot_proposal, keys, target, jnp.zeros(P, jnp.int32))
    weight = jnp.where(jnp.isnan(weight), 0.0, weight)
    gather = lambda a: jnp.take_along_axis(a, s_new[:, None], axis=1)[:, 0]
    new_tok = jnp.take_along_axis(cand_tok, s_new[:, None, None], axis=1)[:, 0, :]      # (P,T)
    new_len, new_surf = gather(cand_len), gather(cand_surf)
    active = w < n_words                                                                # (P,)
    word_tok = word_tok.at[:, w, :].set(jnp.where(active[:, None], new_tok, cur_tok))
    word_len = word_len.at[:, w].set(jnp.where(active, new_len, cur_len))
    word_surf = word_surf.at[:, w].set(jnp.where(active, new_surf, cur_surf))
    move_logw = move_logw + jnp.where(active, weight, 0.0)
    return word_tok, word_len, word_surf, move_logw


@functools.lru_cache(maxsize=64)
def _build_step(sl, Wmax, T, M, K, mt, eos_id, Vc, band, tail_fn):
    """Build (and memoize on the static structure + ``tail_fn``) the FUSED jitted per-word Gibbs/SMCP3
    step (``tail_fn`` runs inside the jit, over all P). This is the non-dedup path + the toy default.

    Memoizing on the structural signature is what makes the sweep **compile once across runs**: the
    per-run-varying data (``emit_full``/``a0``/pool spans + ``wdel``/``wins``/``seed_ids``) is threaded
    as TRACED ARGS rather than baked into the closure, so two ``run`` calls of the SAME shape (e.g. one
    per seed, or per same-length sentence) get the identical ``step`` object and hit JAX's jit cache --
    no recompile. Only a new sentence *length* (new ``M``/``Wmax`` -> new key) recompiles. ``tail_fn``
    is part of the key and is stable per model (the Pythia KV scorer comes from the lru_cached
    ``_pythia_model``; the toy uncached fallback is rebuilt per ctx, which is cheap)."""
    Kt = 1 + K
    slot_model, slot_proposal = _slot_gf(Kt)

    @jax.jit
    def step(key, w, word_tok, word_len, word_surf, move_logw, done, n_words,
             emit_full, a0p, pool_tok, pool_len, pool_surf, wdel_p, wins_p, seed_ids, lm_temp,
             lp_copy, lp_sub, copy_mask):
        cand_tok, cand_len, cand_surf, valid = _candidates(
            w, word_tok, word_len, word_surf, pool_tok, pool_len, pool_surf, K, T)
        chan = _chan_scores(w, word_len, word_surf, cand_len, cand_surf, done,
                            a0p, emit_full, wdel_p, wins_p, band, M, Wmax, lp_copy, lp_sub, copy_mask)
        ctx_bufs, ctx_lens, tail, tail_len = _tail_inputs(
            w, word_tok, word_len, cand_tok, cand_len, done, n_words, sl, Wmax, T, mt, eos_id, seed_ids)
        chain = tail_fn(ctx_bufs, ctx_lens, tail, tail_len)                         # (P, Kt) -- LM forward
        target = jnp.where(valid, lm_temp * chain + chan, -jnp.inf)
        return _apply_move(key, target, w, cand_tok, cand_len, cand_surf,
                           word_tok, word_len, word_surf, move_logw, n_words, slot_model, slot_proposal)

    return step


@functools.lru_cache(maxsize=64)
def _build_dedup_steps(sl, Wmax, T, M, K, mt, eos_id, Vc, band):
    """R3 item 1b: the per-word step SPLIT at the LM-forward seam so ``tail_fn`` can run on the HOST over
    UNIQUE buffers (:func:`_dedup_tail`) instead of inside the jit over all P. ``emit_inputs`` (jitted,
    no LM) builds the ``tail_fn`` inputs; ``move`` (jitted) recomputes the cheap candidates + channel and
    finishes the SMCP3 sample from the passed-in ``chain``. Memoized on the structure -- and crucially
    with NO ``tail_fn`` in the key (it is host-side now, not traced)."""
    Kt = 1 + K
    slot_model, slot_proposal = _slot_gf(Kt)

    @jax.jit
    def emit_inputs(w, word_tok, word_len, word_surf, done, n_words, pool_tok, pool_len, pool_surf,
                    seed_ids):
        cand_tok, cand_len, _cs, _v = _candidates(w, word_tok, word_len, word_surf, pool_tok,
                                                  pool_len, pool_surf, K, T)
        return _tail_inputs(w, word_tok, word_len, cand_tok, cand_len, done, n_words,
                            sl, Wmax, T, mt, eos_id, seed_ids)

    @jax.jit
    def move(key, chain, w, word_tok, word_len, word_surf, move_logw, done, n_words,
             emit_full, a0p, pool_tok, pool_len, pool_surf, wdel_p, wins_p, lm_temp,
             lp_copy, lp_sub, copy_mask):
        cand_tok, cand_len, cand_surf, valid = _candidates(
            w, word_tok, word_len, word_surf, pool_tok, pool_len, pool_surf, K, T)
        chan = _chan_scores(w, word_len, word_surf, cand_len, cand_surf, done,
                            a0p, emit_full, wdel_p, wins_p, band, M, Wmax, lp_copy, lp_sub, copy_mask)
        target = jnp.where(valid, lm_temp * chain + chan, -jnp.inf)
        return _apply_move(key, target, w, cand_tok, cand_len, cand_surf,
                           word_tok, word_len, word_surf, move_logw, n_words, slot_model, slot_proposal)

    return emit_inputs, move


def _dedup_tail(tail_fn, ctx_bufs, ctx_lens, tail, tail_len, stats=None):
    """Score only the UNIQUE ``tail_fn``-input rows and scatter the [P,Kt] chain back (R3 item 1b). Key
    per row = exactly the bytes ``tail_fn`` reads -- ``ctx_bufs[:ctx_len]`` (the prefix it prefills) ++
    ``tail`` ++ ``tail_len`` -- so duplicate post-resample buffers (uniq/P ~ 0.07) share ONE prefill.
    EXACT: the per-particle SMCP3 sample (in ``move``) is unchanged, so the sweep is bit-identical given
    the same RNG; only redundant LM forwards are removed. Host-side (un-jitted dict over bytes); unique
    rows padded to a fixed bucket ladder so the jitted ``tail_fn`` recompiles at only a few batch sizes."""
    from genjax_port import cache_dedup
    cb = np.asarray(ctx_bufs)
    cl = np.asarray(ctx_lens).astype(np.int64)
    tl = np.asarray(tail)
    tll = np.asarray(tail_len)
    P = cb.shape[0]
    slot_of, reps, inverse = {}, [], np.empty(P, np.int64)
    for r in range(P):
        key = cb[r, :cl[r]].tobytes() + b"|" + tl[r].tobytes() + b"|" + tll[r].tobytes()
        slot = slot_of.get(key)
        if slot is None:
            slot = len(reps)
            slot_of[key] = slot
            reps.append(r)
        inverse[r] = slot
    U = len(reps)
    Ub = cache_dedup._bucket_size(U, P)                                  # pad to a fixed rung (compiles)
    rep_idx = np.array(reps + [reps[0]] * (Ub - U), np.int64)            # pad with a valid row (no NaNs)
    chain = tail_fn(jnp.asarray(cb[rep_idx]), jnp.asarray(cl[rep_idx]),
                    jnp.asarray(tl[rep_idx]), jnp.asarray(tll[rep_idx]))  # [Ub, Kt]
    if stats is not None:
        stats.calls += 1
        stats.rows_in += P
        stats.rows_computed += Ub
    return chain[jnp.asarray(inverse)]                                   # [P, Kt]


def make_sweep(ctx, pool_tok, pool_len, pool_surf=None, max_tail=None, dedup=False):
    """Build a reusable ``sweep(key, ctx_buf, ctx_len, word_len, word_surf, positions, done,
    dedup_stats) -> (ctx_buf, ctx_len, word_len, word_surf, log_alpha, move_logw)``. ``word_len`` /
    ``word_surf`` (per-word token counts + channel surface ids) come from the forward filter state
    (R4 multi-token); when omitted they default to the single-token reading of ``ctx_buf`` (toy /
    ``gibbs_sweep``). ``pool_surf`` is each pool candidate's surface id (defaults to the first token --
    the single-token case). The per-word step is built by a memoized factory (:func:`_build_step` /
    :func:`_build_dedup_steps`) so it is jitted ONCE PER STRUCTURE and reused across the many resample
    events in one run AND across separate ``run`` calls of the same shape (per-run arrays are passed as
    args, not baked in -- the fix for the per-run recompile that dominated R3 wall-clock). ``done`` is a
    per-word step ARG so one compiled step serves every call; ``positions`` only sets the loop length.

    **LM scoring (R3): only the SUFFIX is scored.** The conditional ``q(x) ∝ LM(x|prefix) +
    LM(suffix|prefix,x) + channel(x)`` -- the prefix LM is identical for every candidate (the move
    only touches word ``w``), so it CANCELS in the softmax / SMCP3 weight and is never computed. We
    score the candidate-dependent tail ``[x, suffix words, EOS?]`` via ``ctx.model.tail_logprobs``
    (Pythia: a KV-cached scorer that prefills the prefix once and shares it across candidates; toy /
    default: a generic uncached chain-rule). ``max_tail`` bounds the suffix width (default ``Wmax+1``;
    the filter passes ``rejuv_lookback+1`` for a windowed sweep).

    **``dedup=True`` (R3 item 1b)** runs ``tail_fn`` on the HOST over the unique post-resample buffers
    (:func:`_dedup_tail`) instead of inside the jit over all P -- the sweep's dominant cost (its KV
    prefills scale ~linearly with P) drops to scale with the unique count. EXACT (bit-identical given the
    same RNG): the per-word key split and per-particle sample are unchanged; only the deterministic tail
    scores are deduped. ``sweep(..., dedup_stats=DedupStats())`` collects the rows-saved ratio."""
    sl, Wmax, T, M = ctx.seed_len, ctx.Wmax, ctx.t_max, ctx.M
    n_out = Wmax * T
    K = pool_tok.shape[1]
    mt = (n_out + 1) if max_tail is None else max_tail   # suffix-tail budget in TOKENS
    eos_id, Vc = ctx.model.eos_id, ctx.model.emit_vocab
    # tail_fn: Pythia's KV scorer (stable, from the lru_cached _pythia_model) or the toy uncached
    # fallback. In the dedup path it is called HOST-SIDE (not a _build_* cache key); in the fused path
    # it is the _build_step key and must be stable across runs for the compile to be reused.
    tail_fn = ctx.model.tail_logprobs or functools.partial(_tail_chain_uncached, ctx.model.lm_fn)
    if dedup:
        emit_inputs, move = _build_dedup_steps(sl, Wmax, T, M, K, mt, eos_id, Vc, ctx.band)
    else:
        step = _build_step(sl, Wmax, T, M, K, mt, eos_id, Vc, ctx.band, tail_fn)

    # Per-run data threaded as TRACED args (not baked into the step) so same-shape runs reuse the compile.
    emit_full = ctx.emit_full
    Vc_aug = emit_full.shape[1]                               # augmented channel width (copy_mask columns)
    wdel0 = jnp.float32(ctx.wdel)                             # global delete cost (OFF / char-copy default)
    wins0 = jnp.asarray(ctx.wins, jnp.float32)               # scalar or (M,) global insertion cost (OFF)
    a00 = jnp.asarray(ctx.a0, jnp.float32)                   # (M+1,) leading-spurious init (OFF)
    lm_temp = jnp.float32(ctx.lm_temp)                        # LM-prior temperature (traced; see RejuvCtx)
    seed_ids = jnp.asarray(ctx.model.seed_ids, jnp.int32) if sl else jnp.zeros((0,), jnp.int32)
    if pool_surf is None:
        pool_surf = jnp.asarray(pool_tok)[:, :, 0]                # single-token default: surface == token
    pool_tok, pool_len, pool_surf = jnp.asarray(pool_tok), jnp.asarray(pool_len), jnp.asarray(pool_surf)

    def sweep(key, ctx_buf, ctx_len, word_len=None, word_surf=None, positions=None, done=None,
              dedup_stats=None, theta_costs=None):
        """``theta_costs`` (None = char-copy / OFF) is the per-particle WORD-ACTION cost tuple
        ``(lp_copy (P,), lp_sub (P,), wdel_p (P,), wins_p (P,M), a0p (P,M+1), copy_mask (M,Vc_aug))`` --
        the SAME costs the forward filter carries from each particle's theta. Absent, the zero-action
        char-copy parameterization is built from ``ctx`` (``lp_copy=lp_sub=0``, global ``wdel``/``wins``,
        ``ctx.a0``, all-zero ``copy_mask``) -> bit-identical to the pre-word-action sweep."""
        P, LCTX = ctx_buf.shape
        done = jnp.ones(P, bool) if done is None else done
        if word_len is None:                                     # single-token default (toy / gibbs_sweep)
            n0 = ctx_len - sl
            word_len = (jnp.arange(Wmax)[None, :] < n0[:, None]).astype(jnp.int32)
        if word_surf is None:
            word_surf = ctx_buf[:, sl:sl + Wmax]
        if theta_costs is None:                                  # char-copy / OFF: zero action offset
            lp_copy, lp_sub = jnp.zeros(P), jnp.zeros(P)
            wdel_p = jnp.broadcast_to(wdel0, (P,))
            wins_p = jnp.broadcast_to(wins0, (P, M))
            a0p = jnp.broadcast_to(a00, (P, M + 1))
            copy_mask = jnp.zeros((M, Vc_aug), jnp.float32)
        else:
            lp_copy, lp_sub, wdel_p, wins_p, a0p, copy_mask = theta_costs
        word_tok, n_words = _unpack(ctx_buf, word_len, sl, Wmax, T)
        move_logw = jnp.zeros(P)
        for w in (range(Wmax) if positions is None else positions):
            key, sub = jax.random.split(key)                            # same split order both paths
            wi = jnp.int32(w)
            if dedup:
                ci = emit_inputs(wi, word_tok, word_len, word_surf, done, n_words,
                                 pool_tok, pool_len, pool_surf, seed_ids)  # (ctx_bufs,ctx_lens,tail,tail_len)
                chain = _dedup_tail(tail_fn, *ci, stats=dedup_stats)    # host: unique tail_fn -> [P,Kt]
                word_tok, word_len, word_surf, move_logw = move(
                    sub, chain, wi, word_tok, word_len, word_surf, move_logw, done, n_words,
                    emit_full, a0p, pool_tok, pool_len, pool_surf, wdel_p, wins_p, lm_temp,
                    lp_copy, lp_sub, copy_mask)
            else:
                word_tok, word_len, word_surf, move_logw = step(
                    sub, wi, word_tok, word_len, word_surf, move_logw, done, n_words,
                    emit_full, a0p, pool_tok, pool_len, pool_surf, wdel_p, wins_p, seed_ids, lm_temp,
                    lp_copy, lp_sub, copy_mask)
        bufs, total = _flat_buffer(ctx, word_tok, word_len, LCTX, n_out)
        ctx_len2 = sl + total.astype(jnp.int32)                  # word lengths may have changed (multi-token)
        log_alpha = channel_carry(a0p, emit_full, ctx.band, M, word_surf, word_len,  # (P, M+1) for filter
                                  lp_copy, lp_sub, wdel_p, wins_p, copy_mask)
        return bufs, ctx_len2, word_len, word_surf, log_alpha, move_logw

    return sweep


def gibbs_sweep(key, ctx_buf, ctx_len, ctx, pool_tok, pool_len, positions=None, done=None):
    """One Gibbs/SMCP3 pass resampling each word in ``positions`` from its full conditional over the
    candidate set [COPY (current word)] + ``pool`` (the SymSpell / vocab pool for that slot). Thin
    wrapper over :func:`make_sweep` (rebuilds the jitted step each call -- fine for one-off / toy use;
    the in-loop filter uses ``make_sweep`` once and reuses it).

    EQUALLY-WEIGHTED cloud in; ``(ctx_buf, log_alpha, move_logw)`` out -- ``log_alpha`` is the
    recomputed forward carry consistent with the swept words (so the filter can continue), ``move_logw``
    the per-particle SMCP3 weight to fold into ``log_w`` BEFORE the next resample. ``pool_tok``
    (Wmax, K, T_max) / ``pool_len`` (Wmax, K) are the shared candidate spans per word slot
    (-1 / 0 = pad); the per-particle COPY is prepended (index 0) so the move can keep the current word
    and the reverse proposal is always scorable, and pool entries equal to COPY are de-duplicated.

    ``done [P]`` selects each particle's target: DONE (complete hypothesis) scores the terminal channel
    marginal ``alpha[M]`` + an EOS LM term; not-done (mid-loop) scores the partial forward mass
    ``logsumexp(alpha)`` with no EOS. ``done=None`` => all terminal (end-of-sequence sweep; the R1/toy
    default, which reproduces the certified terminal scoring exactly). Returns the toy's 3-tuple
    ``(ctx_buf, log_alpha, move_logw)`` (word_len/word_surf default to the single-token reading)."""
    cb, _cl, _wl, _ws, la, mlw = make_sweep(ctx, pool_tok, pool_len)(
        key, ctx_buf, ctx_len, positions=positions, done=done)
    return cb, la, mlw


def pool_from_table(cand_table, t_max=1):
    """Adapt a flat (Wmax, K) candidate-token table (single-token, the toy whole-vocab Gibbs set or a
    SymSpell table) into (pool_tok [Wmax, K, T_max], pool_len [Wmax, K]). -1 entries are pads."""
    Wmax, K = cand_table.shape
    pool_tok = jnp.full((Wmax, K, t_max), -1, jnp.int32).at[:, :, 0].set(cand_table)
    pool_len = jnp.where(cand_table >= 0, 1, 0).astype(jnp.int32)
    return pool_tok, pool_len


def build_pool(observed, model, max_dist, Ke, Wmax, t_max=1):
    """Per-intended-slot candidate pool from the model's ``candidate_words`` (COPY + SymSpell). Slot
    ``i`` uses the candidates of observed word ``min(i, M-1)`` (1:1 alignment, clipped past the
    observed length -- those slots are inactive for an M-word parse anyway). Returns
    (pool_tok [Wmax, Ke, T_max], pool_len [Wmax, Ke])."""
    obs_words = model.obs_words(observed)
    obs_spans = (model.obs_spans(observed) if getattr(model, "obs_spans", None) is not None
                 else [None] * len(obs_words))
    M = len(obs_words)
    rows = []
    for i in range(Wmax):
        oi = min(i, M - 1)
        cands = list(model.candidate_words(obs_words[oi], obs_spans[oi], max_dist, Ke))[:Ke]
        ids = [int(span[0]) for span, _surf in cands]   # single-token pool (R4: full multi-token spans)
        rows.append(ids + [-1] * (Ke - len(ids)))
    return pool_from_table(jnp.array(rows, jnp.int32), t_max=t_max)


def decode_counts(ctx_buf, ctx_len, model, seed_len, top=50):
    """Posterior over sentences from an EQUALLY-WEIGHTED cloud (uniform count, no log_w)."""
    P = ctx_buf.shape[0]
    trajs = [tuple(int(t) for t in ctx_buf[p][seed_len:int(ctx_len[p])]) for p in range(P)]
    c = Counter(trajs)
    return {model.decode_ids(t): n / P for t, n in c.most_common(top)}


# ==================================================================================================
# Birth/death involution core -- REJUV_BIRTH_DEATH_PLAN.md Phase 0.
#
# A single SYMMETRIC move that adds OR removes one intended word, built as an involution: ``_insert_word``
# (birth) and ``_delete_word`` (death) are mutual inverses at the same position ``w``, with the deleted
# word as the dimension-matching auxiliary ``x``. ``_involve`` is the one move -- it flips direction and
# is its OWN INVERSE (``_involve ∘ _involve = id``). Everything here is PURE array surgery on the per-word
# buffers (no LM / channel scoring); the SMCP3 weight + the proposals (q_ins / q_del / κ) land in Phase 1.
#
# All ops are vectorized over P with PER-PARTICLE ``(w, x)``. Canonical state convention (matches the
# forward filter / ``_unpack``): a word slot is ACTIVE iff ``word_len > 0``; active words fill the prefix
# ``[0, n_words)`` and pad slots carry ``(word_tok=0, word_len=0, word_surf=_PAD_SURF)``. Both ops preserve
# this canonical form, so the round-trip is bit-exact on the full buffers (tests/test_rejuv_birth_death.py).
# ==================================================================================================
_PAD_SURF = -1


def _gather_slots(arr, idx):
    """Gather along the word axis (axis=1) for a 2-D ``(P, Wmax)`` or 3-D ``(P, Wmax, T)`` per-word buffer,
    broadcasting the ``(P, Wmax)`` index over the trailing token axis when present."""
    if arr.ndim == 3:
        return jnp.take_along_axis(arr, jnp.broadcast_to(idx[:, :, None], arr.shape), axis=1)
    return jnp.take_along_axis(arr, idx, axis=1)


def _insert_word(word_tok, word_len, word_surf, n_words, w, x_tok, x_len, x_surf):
    """Birth: insert word ``x = (x_tok [P,T], x_len [P], x_surf [P])`` at gap ``w [P]`` (``0 <= w <= n_words``).
    Words at positions ``>= w`` shift right one slot; ``n_words += 1``. Inverse of :func:`_delete_word` at the
    same ``w``. The caller's birth boundary guarantees ``n_words < Wmax`` (else the last word would fall off
    the end -- never happens under the move). Output stays canonical."""
    P, Wmax, T = word_tok.shape
    ar = jnp.arange(Wmax)[None, :]                                   # (1, Wmax) output slot indices
    at = ar == w[:, None]                                            # (P, Wmax) the slot that receives x
    src = jnp.where(ar <= w[:, None], ar, ar - 1)                    # i<=w -> i (i==w overridden); i>w -> i-1
    srcc = jnp.clip(src, 0, Wmax - 1)
    new_tok = jnp.where(at[:, :, None], x_tok[:, None, :], _gather_slots(word_tok, srcc))
    new_len = jnp.where(at, x_len[:, None], _gather_slots(word_len, srcc))
    new_surf = jnp.where(at, x_surf[:, None], _gather_slots(word_surf, srcc))
    return new_tok, new_len, new_surf, n_words + 1


def _delete_word(word_tok, word_len, word_surf, n_words, w):
    """Death: remove the word at position ``w [P]`` (``0 <= w < n_words``). Words at positions ``> w`` shift
    left one slot; ``n_words -= 1``; the freed tail slot is forced to canonical pad. Returns
    ``((word_tok', word_len', word_surf', n_words'), (x_tok, x_len, x_surf))`` where ``x`` is the removed
    word -- the dimension-matching auxiliary that :func:`_insert_word` at the same ``w`` re-inserts. Inverse
    of :func:`_insert_word`."""
    P, Wmax, T = word_tok.shape
    ar = jnp.arange(Wmax)[None, :]
    idx = ar + (ar >= w[:, None]).astype(jnp.int32)                  # pull from i+1 at/after w (shift left)
    overflow = idx >= Wmax                                           # last slot has nothing to pull -> pad
    idxc = jnp.clip(idx, 0, Wmax - 1)
    new_tok = jnp.where(overflow[:, :, None], 0, _gather_slots(word_tok, idxc))
    new_len = jnp.where(overflow, 0, _gather_slots(word_len, idxc))
    new_surf = jnp.where(overflow, _PAD_SURF, _gather_slots(word_surf, idxc))
    wc = jnp.clip(w, 0, Wmax - 1)
    x_tok = _gather_slots(word_tok, jnp.broadcast_to(wc[:, None], (P, Wmax)))[:, 0, :]
    x_len = _gather_slots(word_len, wc[:, None])[:, 0]
    x_surf = _gather_slots(word_surf, wc[:, None])[:, 0]
    return (new_tok, new_len, new_surf, n_words - 1), (x_tok, x_len, x_surf)


def _involve(word_tok, word_len, word_surf, n_words, d_birth, w, x_tok, x_len, x_surf):
    """The single symmetric birth/death move as an INVOLUTION. ``d_birth [P]`` selects per particle: True =
    insert ``x`` at ``w`` (birth), False = delete the word at ``w`` (death). Returns the new augmented tuple
    ``(word_tok, word_len, word_surf, n_words, d_birth, w, x_tok, x_len, x_surf)`` with the direction FLIPPED
    and the auxiliary carried so that ``_involve ∘ _involve = id``:

      * birth -> death: ``x`` is the inserted word, which is exactly the word now at slot ``w`` -- so the
        re-applied death recovers it and undoes the insert.
      * death -> birth: ``x`` is the recovered (removed) word -- so the re-applied birth re-inserts it.

    Self-inverse holds on the move's support: a death's input ``x`` slot is the word at ``w`` (which death
    ignores but carries), birth's ``x`` is the word to insert. The position ``w`` is unchanged by φ."""
    ins = _insert_word(word_tok, word_len, word_surf, n_words, w, x_tok, x_len, x_surf)
    (dlt, (rx_tok, rx_len, rx_surf)) = _delete_word(word_tok, word_len, word_surf, n_words, w)
    sel, sel3 = d_birth[:, None], d_birth[:, None, None]
    out_tok = jnp.where(sel3, ins[0], dlt[0])
    out_len = jnp.where(sel, ins[1], dlt[1])
    out_surf = jnp.where(sel, ins[2], dlt[2])
    out_nw = jnp.where(d_birth, ins[3], dlt[3])
    out_x_tok = jnp.where(sel, x_tok, rx_tok)                       # birth carries x; death carries removed
    out_x_len = jnp.where(d_birth, x_len, rx_len)
    out_x_surf = jnp.where(d_birth, x_surf, rx_surf)
    return out_tok, out_len, out_surf, out_nw, ~d_birth, w, out_x_tok, out_x_len, out_x_surf


# ==================================================================================================
# Birth/death move + reversible-jump weight -- REJUV_BIRTH_DEATH_PLAN.md Phase 1.
#
# ``birth_death_move`` performs one involutive add/remove-a-word move per particle and RETURNS the
# trans-dimensional SMCP3 weight to fold into ``log_w`` before resampling (the move ALWAYS applies --
# no accept/reject; the weight does the correction, exactly as the substitution sweep folds ``move_logw``).
# Proposals are NEAR-CONDITIONAL (Phase 2), both directions: a DEATH samples its position from q_del(w) ∝
# softmax over deletable positions of logπ(y \ word w) (``_del_logq``), concentrating on the removable
# duplicate; a BIRTH keeps the gap uniform but samples its word from q_ins(x) ∝ softmax over the pool of
# logπ(y + x @ gap) (``_ins_logq``). Balancing q_ins against q_del is what keeps the trans-dim weight
# low-variance -- informed births stop proposing easy-to-undo spurious words (the Phase-1 uniform asymmetry
# was the logZ-depressing variance source). A word is "deletable" iff its surface is in the pool -- so the
# pool is exactly the set of words a birth can create, which makes every death reversible (its reverse birth
# has positive density). ``score_fn`` is injected (the toy test passes a synthetic target; production passes
# an LM+channel closure). The weight ``_bd_log_weight`` is PROPOSAL-AGNOSTIC -- ``birth_death_move`` computes
# the ACTUAL forward/reverse densities (q_del/q_ins at y for the chosen move, and re-evaluated at y' for the
# reverse) and feeds them in -- so the RISKY part (densities + weight) is certified END-TO-END by the exact
# transition-sum invariance tests (tests/test_rejuv_birth_death.py), independent of any scoring.
# ==================================================================================================
def _in_pool(word_surf, cand_surf):
    """(P, Wmax) bool: is each slot's surface one of the ``cand_surf`` (Kc,) pool surfaces?"""
    return jnp.any(word_surf[:, :, None] == cand_surf[None, None, :], axis=2)


def _deletable_count(word_len, word_surf, cand_surf):
    """(P,) number of ACTIVE slots whose surface is in the pool -- the death proposal's support size D."""
    return jnp.sum((word_len > 0) & _in_pool(word_surf, cand_surf), axis=1)


def _bd_log_weight(logp_y, logp_yp, log_qfwd, log_qbwd):
    """Trans-dim reversible-jump / SMCP3 log-weight for one birth/death move (plan §2), Jacobian 1 (discrete),
    **proposal-agnostic**: the caller supplies the actual forward proposal log-density ``log_qfwd`` (the density
    of the move that produced y' from y) and the reverse log-density ``log_qbwd`` (the move that undoes it from
    y'). The involutive-MCMC weight is then just

        W = [logπ(y') − logπ(y)] + log_qbwd − log_qfwd.

    Phase-1's uniform proposals (direction·gap·word for a birth, direction·position for a death) are the special
    case where the caller passes the uniform densities; Phase-2 informed proposals only change those two scalars
    in :func:`birth_death_move`, never this formula. Vectorized over P; only the SELECTED move's densities enter
    per particle (the caller zeroes no-op particles). The selected branch's reverse density is always positive
    (a birth leaves a deletable pool word; a death leaves births feasible), so ``W`` is finite there."""
    return (logp_yp - logp_y) + log_qbwd - log_qfwd


def _del_logq(score_fn, word_tok, word_len, word_surf, n_words, done, cand_surf):
    """(P, Wmax) NEAR-CONDITIONAL deletion log-distribution (Phase 2): for each position ``w``, the log-softmax
    -- over the DELETABLE positions of that particle (active and surface-in-pool) -- of ``logπ(y with word w
    removed)``. Non-deletable positions are masked to ``−inf`` (≈ zero probability); a particle with no
    deletable position gets an all-``−inf`` (→ NaN after log_softmax) row, which is harmless because its death
    branch is never selected and birth-reverse only ever indexes the inserted (always-deletable) slot.

    Scores the ``Wmax`` candidate removals with the injected ``score_fn`` via ``lax.map`` (sequential over
    positions -> O(Wmax) score_fn calls, gentle on memory; each score_fn is itself ~O(Wmax) LM steps, so this
    is the O(Wmax^2) cost the Phase-2/3 suffix-tail KV sharing will cut). The malformed states produced by
    "removing" a pad position are scored too but masked out, so any NaN/inf there never reaches the softmax."""
    P, Wmax, T = word_tok.shape

    def score_remove_at(i):
        wcol = jnp.full((P,), i, jnp.int32)
        (dt, dl, ds, dnw), _ = _delete_word(word_tok, word_len, word_surf, n_words, wcol)
        return score_fn(dt, dl, ds, dnw, done)                       # (P,) logπ(y \ word i)

    scores = jax.lax.map(score_remove_at, jnp.arange(Wmax)).T         # (P, Wmax)
    deletable = (jnp.arange(Wmax)[None, :] < n_words[:, None]) & _in_pool(word_surf, cand_surf)
    logits = jnp.where(deletable, scores, -jnp.inf)                  # mask drops any garbage/NaN at pad slots
    return jax.nn.log_softmax(logits, axis=1)


def _ins_logq(score_fn, word_tok, word_len, word_surf, n_words, done, cand_tok, cand_len, cand_surf, gap):
    """(P, Kc) NEAR-CONDITIONAL insertion-word log-distribution at a GIVEN per-particle ``gap [P]``: for each
    pool candidate ``c``, the log-softmax over the WHOLE pool of ``logπ(y with candidate c inserted at gap)``.
    The pool is the full candidate set (every entry insertable), so the row is always finite and normalized.
    This is the birth's WORD proposal q_ins(·|gap, y); re-evaluated at y' with gap = w_d it is the reverse
    density of a death (the birth that re-inserts the removed word). Balancing q_ins against q_del is what
    keeps the trans-dim weight low-variance: informed births stop proposing easy-to-undo spurious words.
    Scores Kc insertions via score_fn (lax.map over the pool). Full particles (no headroom) get garbage rows
    that the caller discards (their birth branch is never selected)."""
    P = word_tok.shape[0]
    Kc = cand_surf.shape[0]

    def score_ins(c):
        it, il, is_, inw = _insert_word(word_tok, word_len, word_surf, n_words, gap,
                                        jnp.broadcast_to(cand_tok[c], (P,) + cand_tok.shape[1:]),
                                        jnp.broadcast_to(cand_len[c], (P,)),
                                        jnp.broadcast_to(cand_surf[c], (P,)))
        return score_fn(it, il, is_, inw, done)                      # (P,) logπ(y + cand c @ gap)

    scores = jax.lax.map(score_ins, jnp.arange(Kc)).T                # (P, Kc)
    return jax.nn.log_softmax(scores, axis=1)


def birth_death_move(key, word_tok, word_len, word_surf, n_words, done,
                     score_fn, cand_tok, cand_len, cand_surf):
    """One SMCP3-weighted birth/death involution move per particle (plan §1-2). ``score_fn(word_tok,
    word_len, word_surf, n_words, done) -> logπ [P]`` is the injected target (lm_temp*LM + channel).
    ``cand_*`` is the fixed candidate pool: ``cand_tok [Kc, T]`` / ``cand_len [Kc]`` / ``cand_surf [Kc]``.
    Returns ``((word_tok', word_len', word_surf', n_words'), move_logw [P])`` -- the move always applies;
    fold ``move_logw`` into ``log_w`` before the next resample."""
    P, Wmax, T = word_tok.shape
    Kc = cand_surf.shape[0]
    parange = jnp.arange(P)
    kdir, kbpos, kbword, kdpos = jax.random.split(key, 4)

    D_y = _deletable_count(word_len, word_surf, cand_surf)
    feas_birth = n_words < Wmax
    feas_death = D_y > 0
    both = feas_birth & feas_death
    none = (~feas_birth) & (~feas_death)
    p_birth = jnp.where(both, 0.5, jnp.where(feas_birth, 1.0, 0.0))          # 0 if only death feasible
    d_birth = jax.random.bernoulli(kdir, p_birth) & (~none)

    # birth proposal: gap w_b ~ U{0..n}; word x ~ q_ins, the NEAR-CONDITIONAL softmax over the pool at w_b
    # (_ins_logq). Informed word choice is what balances q_ins against q_del so the trans-dim weight stays
    # low-variance. ``ins_logq_y`` re-evaluated at y'/w_d below is also the reverse density of a death.
    nf = n_words.astype(jnp.float32)
    w_b = jnp.clip(jnp.floor(jax.random.uniform(kbpos, (P,)) * (nf + 1.0)).astype(jnp.int32), 0, n_words)
    ins_logq_y = _ins_logq(score_fn, word_tok, word_len, word_surf, n_words, done,
                           cand_tok, cand_len, cand_surf, w_b)              # (P, Kc)
    ci = jax.random.categorical(kbword, ins_logq_y)
    bstate = _insert_word(word_tok, word_len, word_surf, n_words, w_b,
                          cand_tok[ci], cand_len[ci], cand_surf[ci])

    # death proposal: position w_d ~ q_del, the NEAR-CONDITIONAL softmax over deletable positions (_del_logq).
    # The same distribution re-evaluated at y' is the reverse density of a birth (computed below).
    del_logq_y = _del_logq(score_fn, word_tok, word_len, word_surf, n_words, done, cand_surf)
    cat_logits = jnp.where(feas_death[:, None], del_logq_y, 0.0)     # D=0 rows -> uniform (sampled, discarded)
    w_d = jax.random.categorical(kdpos, cat_logits)
    dstate, _aux = _delete_word(word_tok, word_len, word_surf, n_words, w_d)

    # select branch; no-op particles (neither feasible) keep their state
    sel, sel3 = d_birth[:, None], d_birth[:, None, None]
    keep, keep2, keep3 = none, none[:, None], none[:, None, None]
    new_tok = jnp.where(keep3, word_tok, jnp.where(sel3, bstate[0], dstate[0]))
    new_len = jnp.where(keep2, word_len, jnp.where(sel, bstate[1], dstate[1]))
    new_surf = jnp.where(keep2, word_surf, jnp.where(sel, bstate[2], dstate[2]))
    new_nw = jnp.where(keep, n_words, jnp.where(d_birth, bstate[3], dstate[3]))

    logp_y = score_fn(word_tok, word_len, word_surf, n_words, done)
    logp_yp = score_fn(new_tok, new_len, new_surf, new_nw, done)
    D_yp = _deletable_count(new_len, new_surf, cand_surf)

    # Forward/reverse proposal log-densities for the SELECTED move, fed to the proposal-agnostic weight
    # (_bd_log_weight). A birth = direction · gap(uniform 1/(n+1)) · q_ins(near-conditional word); a death =
    # direction · q_del(near-conditional position). The REVERSE of a birth is a death at y' -- its density is
    # q_del RE-EVALUATED at y' (``del_logq_yp``) at the inserted slot w_b; the reverse of a death is a birth at
    # y' re-inserting the removed word at gap w_d -- density gap(1/(n'+1)) · q_ins RE-EVALUATED at y'/w_d.
    nnwf = new_nw.astype(jnp.float32)
    qdel_fwd = del_logq_y[parange, w_d]                                            # death fwd: log q_del(w_d|y)
    qins_fwd = ins_logq_y[parange, ci]                                             # birth fwd: log q_ins(x|w_b,y)
    del_logq_yp = _del_logq(score_fn, new_tok, new_len, new_surf, new_nw, done, cand_surf)
    qdel_rev = del_logq_yp[parange, w_b]                                           # birth rev: log q_del(w_b|y')
    ins_logq_yp = _ins_logq(score_fn, new_tok, new_len, new_surf, new_nw, done,
                            cand_tok, cand_len, cand_surf, w_d)                     # q_ins at y', gap = w_d
    rem_surf = word_surf[parange, w_d]                                             # surface the death removed
    rem_ci = jnp.argmax(cand_surf[None, :] == rem_surf[:, None], axis=1)           # its pool index (in-pool)
    qins_rev = ins_logq_yp[parange, rem_ci]                                        # death rev: log q_ins(rem|w_d,y')
    # direction probabilities: p_dir_fwd = prob this direction was sampled at y; p_dir_rev = prob the move's
    # own direction rule picks the REVERSE direction at y'. Recompute the rule at y' (feasibility there).
    p_dir_fwd = jnp.where(d_birth, p_birth, 1.0 - p_birth)
    fb_yp, fd_yp = new_nw < Wmax, D_yp > 0
    both_yp = fb_yp & fd_yp
    p_birth_yp = jnp.where(both_yp, 0.5, jnp.where(fb_yp, 1.0, 0.0))
    p_death_yp = jnp.where(both_yp, 0.5, jnp.where(fd_yp, 1.0, 0.0))
    p_dir_rev = jnp.where(d_birth, p_death_yp, p_birth_yp)        # undo a birth = death; undo a death = birth
    log_qfwd = jnp.log(p_dir_fwd) + jnp.where(d_birth, -jnp.log(nf + 1.0) + qins_fwd, qdel_fwd)
    log_qbwd = jnp.log(p_dir_rev) + jnp.where(d_birth, qdel_rev, -jnp.log(nnwf + 1.0) + qins_rev)
    W = _bd_log_weight(logp_y, logp_yp, log_qfwd, log_qbwd)
    # A band-limited channel can make a proposed (or source) parse impossible (logπ = −inf); then the target
    # ratio is −inf−(−inf) = NaN, and a single NaN poisons logZ and every downstream softmax. Such a move is
    # degenerate -> send its weight to −inf so resampling discards that particle. Clean −inf rejections (a
    # finite source moved to an impossible y') and all finite weights pass through unchanged; the toy
    # (band=None) never triggers this, so its results are bit-identical.
    W = jnp.where(jnp.isnan(W) | (W == jnp.inf), -jnp.inf, W)
    move_logw = jnp.where(none, 0.0, W)
    return (new_tok, new_len, new_surf, new_nw), move_logw


# --------------------------------------------------------------------------------------------------
# Production scoring + the in-filter sweep (plan §4). ``_make_bd_score_fn`` reproduces the SAME target the
# filter uses for a complete hypothesis -- ``lm_temp * LM + channel_marginal`` -- via the shared
# ``channel_carry`` / ``_lm_logprior`` (no suffix-tail KV cancellation yet; that is the Phase-2 perf win).
# Only the RATIO logπ(y')−logπ(y) enters the weight, so any constant offset (e.g. the leading-spurious a0)
# cancels. ``make_bd_sweep`` wraps ``birth_death_move`` into a filter-shaped sweep: unpack the flat buffer
# to per-word slots, run N done-ONLY moves (mid-construction particles are left untouched so the forward
# filter's n_words counter never desyncs), repack, and recompute ``log_alpha`` consistent with the move.
# --------------------------------------------------------------------------------------------------
def _bd_costs(ctx, P, theta_costs):
    """The per-particle word-action cost 6-tuple ``(lp_copy, lp_sub, wdel_p, wins_p, a0p, copy_mask)`` --
    the LIVE theta costs when ``theta_costs`` is given (channel ON), else the zero-action char-copy / OFF
    parameterization built from ``ctx`` (bit-identical to the pre-word-action carry), matching ``make_sweep``."""
    if theta_costs is not None:
        return theta_costs
    M, Vc_aug = ctx.M, ctx.emit_full.shape[1]
    return (jnp.zeros(P), jnp.zeros(P),
            jnp.broadcast_to(jnp.float32(ctx.wdel), (P,)),
            jnp.broadcast_to(jnp.asarray(ctx.wins, jnp.float32), (P, M)),
            jnp.broadcast_to(jnp.asarray(ctx.a0, jnp.float32), (P, M + 1)),
            jnp.zeros((M, Vc_aug), jnp.float32))


def _make_bd_score_fn(ctx):
    """``score(word_tok, word_len, word_surf, n_words, done, theta_costs=None) -> logπ [P]`` =
    ``lm_temp * LM(sentence) + channel`` for a batch of states (DONE reads ``alpha[M]``; mid-loop the partial
    forward mass), the same quantity the sweep/filter score. Injected into :func:`birth_death_move`."""
    sl, Wmax, T, M = ctx.seed_len, ctx.Wmax, ctx.t_max, ctx.M
    n_out = Wmax * T

    def score(word_tok, word_len, word_surf, n_words, done, theta_costs=None):
        P = word_tok.shape[0]
        lp_copy, lp_sub, wdel_p, wins_p, a0p, copy_mask = _bd_costs(ctx, P, theta_costs)
        carry = channel_carry(a0p, ctx.emit_full, ctx.band, M, word_surf, word_len,
                              lp_copy, lp_sub, wdel_p, wins_p, copy_mask)              # (P, M+1)
        chan = jnp.where(done, carry[:, M], logsumexp(carry, axis=1))
        bufs, total = _flat_buffer(ctx, word_tok, word_len, sl + n_out + 1, n_out)
        lm = _lm_logprior(ctx, bufs, total, done)
        return ctx.lm_temp * lm + chan

    return score


def make_bd_sweep(ctx, cand_tok, cand_len, cand_surf, n_attempts=2):
    """Build a reusable birth/death rejuvenation sweep over a fixed candidate pool ``cand_*`` (Phase 1 =
    observed surfaces). ``sweep(key, ctx_buf, ctx_len, word_len, word_surf, done, theta_costs) ->
    (state', move_logw)`` where ``state'`` is the filter's 7-tuple ``(ctx_buf, ctx_len, n_words, word_len,
    word_surf, log_alpha, done)``. Runs ``n_attempts`` moves (each adds/removes <=1 word) on DONE particles
    only; folds every move's SMCP3 weight into ``move_logw`` for the caller to add to ``log_w``."""
    sl, Wmax, T, M = ctx.seed_len, ctx.Wmax, ctx.t_max, ctx.M
    n_out = Wmax * T
    score = _make_bd_score_fn(ctx)

    def sweep(key, ctx_buf, ctx_len, word_len, word_surf, done, theta_costs=None):
        P, LCTX = ctx_buf.shape
        wt, nw = _unpack(ctx_buf, word_len, sl, Wmax, T)
        wl, ws = word_len, word_surf
        sfn = lambda a, b, c, d, e: score(a, b, c, d, e, theta_costs)
        move_logw = jnp.zeros(P)
        for _ in range(n_attempts):
            key, sub = jax.random.split(key)
            (nt, nl, ns, nnw), mlw = birth_death_move(sub, wt, wl, ws, nw, done,
                                                      sfn, cand_tok, cand_len, cand_surf)
            m, m2, m3 = done, done[:, None], done[:, None, None]            # done-only gate
            wt = jnp.where(m3, nt, wt); wl = jnp.where(m2, nl, wl)
            ws = jnp.where(m2, ns, ws); nw = jnp.where(m, nnw, nw)
            move_logw = move_logw + jnp.where(m, mlw, 0.0)
        bufs, total = _flat_buffer(ctx, wt, wl, LCTX, n_out)
        ctx_len2 = sl + total.astype(jnp.int32)
        lp_copy, lp_sub, wdel_p, wins_p, a0p, copy_mask = _bd_costs(ctx, P, theta_costs)
        log_alpha = channel_carry(a0p, ctx.emit_full, ctx.band, M, ws, wl,
                                  lp_copy, lp_sub, wdel_p, wins_p, copy_mask)
        return (bufs, ctx_len2, nw, wl, ws, log_alpha, done), move_logw

    return sweep
