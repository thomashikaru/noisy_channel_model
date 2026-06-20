"""The one RB-SMC pair-HMM noisy-channel filter (toy bigram and Pythia are two configs of it).

This is Phase A's unified filter (see ``planning/PAIRHMM_RBSMC_PLAN.md`` A1). The only sampled
latent is the intended sentence, generated left-to-right from an injected LM; the edit-alignment
between intended and observed words is summed out by a nested pair-HMM (per-particle carry
``log_alpha[k]`` = log P(intended prefix, exactly k observed words consumed)). SMC weights are the
forward-mass increment; the proposal is the channel-aware (fully-adapted) one, so weights are
near-zero-variance and small particle counts suffice.

Everything model-specific is injected through a :class:`PairHMMModel`, so the toy bigram (fast,
no LM load, certified against brute-force enumeration by ``tests/test_pairhmm_exact.py``) and
Pythia (``pythia_word_caprop.py``) run *identical* inference code. Correctness proven on the toy
therefore transfers to Pythia by construction.

The proposal kernel is GenJAX ``@gen`` (``genjax.categorical @ "action"`` +
``factor(logsumexp_C) @ "ev"``, driven by ``kernel.importance``); ``proposal="bootstrap"`` selects
the LM-prior baseline (used only for the variance contrast in the exact test). ``poc_word_indel*``
stay as frozen reference PoCs; this module is what everything imports.

**Multi-token intended words (Phase D).** A candidate intended word is a **(token span, surface
id)** pair, not a single token: its LM score is the chain-rule product over the span's tokens and
its channel column is indexed by a ``surface_id`` (decoupled from the token id). Single-token
candidates keep ``surface_id == token_id`` and the cheap one-forward LM score, so **with no
multi-token candidates (``n_mt == 0``, ``T_max == 1``) every code path and value is bit-identical to
the certified single-token filter** (the exact-enumeration gates guard this). Multi-token candidates
(COPY of a >=2-token observed word, M:N substitution neighbours) get their chain-rule from the
injected ``tail_logprobs`` scorer and their channel column from the per-sentence-augmented emission
table. The kernel appends the chosen candidate's whole token span and advances by its length; an
explicit per-particle ``n_words`` counter (not ``ctx_len``) drives the band, since token count and
word count diverge once words are multi-token. See ``planning/PAIRHMM_RBSMC_PLAN.md`` Phase D.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

import genjax
from genjax import ChoiceMap

from genjax_port.genjax_factor import factor
from genjax_port.word_dp import _word_row_update, _ess, channel_carry


@dataclass
class PairHMMModel:
    """The model-specific injections that turn the generic filter into a concrete model.

    All inference (the kernel, the forward DP, resampling, the terminal correction) is generic;
    only these inputs differ between the toy bigram and Pythia.
    """

    lm_fn: Callable            # BATCHED (ctx_buf [P,LCTX], ctx_len [P]) -> logprobs [P, vocab]
    eos_id: int                # index of EOS in the lm_fn output row
    emit_vocab: int            # number of single-token candidate columns (Vc) in the emission table
    vocab_char: jnp.ndarray    # (Vc, Lchar) char ids of every single-token candidate word
    vocab_clen: jnp.ndarray    # (Vc,) char lengths
    channel_logpdf: Callable   # (obs_char_ids, intended_char_ids, n_x) -> channel logpdf
    char_ids: Callable         # word_str -> (list[int] padded char ids, n)
    candidate_words: Callable   # (word_str, obs_span, max_dist, Ke) -> list[(token-span tuple, surf_str)].
    #                             The COPY (FIRST) is the unit's actual observed token span ``obs_span``
    #                             (verbatim, so punctuation/case are faithful; ``obs_span=None`` re-encodes),
    #                             then substitution neighbours. A single-token candidate is ``((tok,), surf)``;
    #                             a multi-token one ``((t0, t1, ...), surf)``. Surfaces drive the channel
    #                             column; spans drive the LM chain-rule + buffer splice.
    obs_words: Callable        # observed_str -> list[word_str]
    decode_ids: Callable       # tuple[int] -> str (intended ids back to a sentence)
    tail_logprobs: Callable = None  # (ctx_bufs, ctx_lens, tails [B,K,w], tail_lens) -> [B,K] chain-rule
    #                                 log P(tail | ctx). Scores multi-token candidate spans in the FORWARD
    #                                 step (Phase D) and the candidate SUFFIX in the rejuvenation sweep
    #                                 (R3). Pythia injects the KV scorer; None => a generic uncached
    #                                 fallback built from lm_fn (pairhmm_rejuv._tail_chain_uncached).
    seed_ids: Sequence[int] = ()  # context seed (toy: (); Pythia: [EOS] + prime tokens)
    word_mask: jnp.ndarray = None  # (emit_vocab,) bool: True where the token is a lexical word; the
    #                                top-J LM candidate pool is restricted to it so the prior cannot
    #                                emit non-word tokens (\n, '#', '****') as intended words. None =
    #                                no restriction (the toy, whose vocab is already all words).
    obs_spans: Callable = None  # observed_str -> list[token-span tuple], parallel to obs_words: each
    #                             unit's actual observed token ids, threaded into candidate_words for a
    #                             faithful COPY. None => candidate_words re-encodes (the toy / generic).
    channel_form: Callable = None  # the base-rate-DECOUPLED FORM channel (COPY_LP=0): scores only the
    #                                substitution FORM, the word-action redesign's emission table (plan
    #                                WORD_ACTION_CHANNEL_PLAN sec 2/3). Used iff run(action_alpha=...) is
    #                                set; the per-word action cost (log p_copy / log p_sub) is added on
    #                                the column. None => word-action unsupported (the OFF / certified path).


def _build_candidates(model, obs_words, obs_spans, obs_char, emit_full, max_dist, Ke, channel_fn=None):
    """Assemble the per-observed-word candidate inventory (Phase D), generalizing the old
    single-token ``_emit_table``. Each candidate is a (token span, surface) from
    ``model.candidate_words``. Returns:

      * ``emit_first`` (M, Ke): the candidate's FIRST token id (-1 pad) -- its single-token LM logprob
        is ``lmlog[emit_first]``.
      * ``emit_surf`` (M, Ke): the candidate's surface id, indexing the (augmented) emission table.
        Single-token candidates keep ``surf == token id`` (an existing column); multi-token surfaces
        get appended columns ``>= Vc``.
      * ``emit_mtidx`` (M, Ke): index into the multi-token inventory, or -1 for single-token.
      * ``mt_span`` (n_mt_pad, T_max) / ``mt_len`` (n_mt_pad): the distinct multi-token candidate
        spans (padded to >= 1 row so the gather is always well-shaped; the dummy row is never selected
        because all ``emit_mtidx`` are -1 when n_mt == 0).
      * ``emit_full_aug`` (M, Vc + n_app): ``emit_full`` with appended channel columns for the
        multi-token surfaces. When there are none it IS ``emit_full`` (bit-identical single-token path).
      * ``T_max`` (>=1), ``n_mt`` (>=0).
    """
    Vc = emit_full.shape[1]
    M = len(obs_words)
    word_cands = [list(model.candidate_words(w, sp, max_dist, Ke))[:Ke]
                  for w, sp in zip(obs_words, obs_spans)]
    T_max = max([1] + [len(span) for cands in word_cands for span, _surf in cands])

    surf_col, extra_surfaces = {}, []          # multi-token surface -> appended column index
    mt_index, mt_list = {}, []                 # span tuple -> mt-inventory index; (span, surf_id)

    def surf_id_for(span, surf):
        if len(span) == 1:
            return int(span[0])                # existing single-token column (bit-identical path)
        if surf not in surf_col:
            surf_col[surf] = Vc + len(extra_surfaces)
            extra_surfaces.append(surf)
        return surf_col[surf]

    first = np.full((M, Ke), -1, np.int32)
    surf = np.full((M, Ke), -1, np.int32)
    mtidx = np.full((M, Ke), -1, np.int32)
    for i, cands in enumerate(word_cands):
        for j, (span, sstr) in enumerate(cands):
            sid = surf_id_for(span, sstr)
            first[i, j] = int(span[0])
            surf[i, j] = sid
            if len(span) > 1:
                key = tuple(int(t) for t in span)
                if key not in mt_index:
                    mt_index[key] = len(mt_list)
                    mt_list.append((key, sid))
                mtidx[i, j] = mt_index[key]
    n_mt = len(mt_list)

    channel_fn = channel_fn or model.channel_logpdf       # match the fn that built emit_full (form vs base)
    if extra_surfaces:
        extra_char = jnp.stack([jnp.asarray(model.char_ids(s)[0], jnp.int32) for s in extra_surfaces])
        extra_clen = jnp.asarray([model.char_ids(s)[1] for s in extra_surfaces], jnp.int32)
        extra_cols = jax.vmap(jax.vmap(channel_fn, in_axes=(None, 0, 0)),
                              in_axes=(0, None, None))(obs_char, extra_char, extra_clen)  # (M, n_app)
        emit_full_aug = jnp.concatenate([emit_full, extra_cols], axis=1)
    else:
        emit_full_aug = emit_full

    n_pad = max(n_mt, 1)
    mt_span = np.zeros((n_pad, T_max), np.int32)
    mt_len = np.ones((n_pad,), np.int32)
    for k, (span, _sid) in enumerate(mt_list):
        mt_span[k, :len(span)] = span
        mt_len[k] = len(span)

    # Case-INSENSITIVE copy mask (M, Vc_aug): copy_mask[m, s] = surface s is a COPY of observed word m,
    # i.e. its (lowercased) char row equals the COPY surface's. This is the word-action analogue of the
    # OFF channel's case-INSENSITIVE char DP: a sentence-initial / orthographic capitalization ('she'->
    # 'She') is a faithful copy, not a substitution. Using ``vocab_char`` (already lowercased by
    # ``char_ids``) makes ' She' and ' she' share a char row, so both match the COPY -- restoring the
    # case-insensitivity the word-action emission offset previously lost by comparing surface IDs.
    vocab_char_aug = (jnp.concatenate([model.vocab_char, extra_char], axis=0)
                      if extra_surfaces else model.vocab_char)             # (Vc_aug, LC)
    copy_chars = vocab_char_aug[jnp.asarray(surf[:, 0])]                   # (M, LC) each slot's COPY chars
    copy_mask = jnp.all(vocab_char_aug[None] == copy_chars[:, None], axis=2)  # (M, Vc_aug) bool
    return (jnp.asarray(first), jnp.asarray(surf), jnp.asarray(mtidx),
            jnp.asarray(mt_span), jnp.asarray(mt_len), emit_full_aug, copy_mask, T_max, n_mt)


def _rejuv_pool_from_inventory(emit_first, emit_surf, emit_mtidx, mt_span, mt_len, M, Wmax, T_max):
    """Per-intended-slot candidate pool for the rejuvenation move (R4), built from the SAME inventory
    the forward filter uses (so the pool's surface ids index the SAME augmented ``emit_full``). Slot
    ``i`` reuses observed word ``min(i, M-1)``'s candidates (1:1 alignment, clipped past the observed
    length). Returns (pool_tok [Wmax, Ke, T_max], pool_len [Wmax, Ke], pool_surf [Wmax, Ke]); -1 / 0 =
    pad. Multi-token candidates carry their full span + augmented surface id (the R4 generalization of
    the old single-token ``build_pool``)."""
    ef, es, em = np.asarray(emit_first), np.asarray(emit_surf), np.asarray(emit_mtidx)
    msp, mln = np.asarray(mt_span), np.asarray(mt_len)
    Ke = ef.shape[1]
    pool_tok = np.full((Wmax, Ke, T_max), -1, np.int32)
    pool_len = np.zeros((Wmax, Ke), np.int32)
    pool_surf = np.full((Wmax, Ke), -1, np.int32)
    for i in range(Wmax):
        oi = min(i, M - 1)
        for j in range(Ke):
            if em[oi, j] >= 0:                                  # multi-token candidate
                k = em[oi, j]
                pool_tok[i, j, :mln[k]] = msp[k, :mln[k]]
                pool_len[i, j] = mln[k]
                pool_surf[i, j] = es[oi, j]
            elif ef[oi, j] >= 0:                                # single-token candidate
                pool_tok[i, j, 0] = ef[oi, j]
                pool_len[i, j] = 1
                pool_surf[i, j] = es[oi, j]
    return jnp.asarray(pool_tok), jnp.asarray(pool_len), jnp.asarray(pool_surf)


def _caprop_scores(log_alpha, lmlog, mt_chain, emit_first, emit_surf, emit_mtidx, mt_span, mt_len,
                   emit_full, offs, J, M, band_mask, t_new, eos_id, emit_vocab, wdel, wins,
                   word_mask, lm_temp, T_max, n_mt, lp_copy, lp_sub, copy_mask):
    """Channel-aware (fully-adapted) candidate scores: [word(Kw), EOS(1)], plus the chosen word's
    token span for the kernel to splice.

    Candidate set C = emission candidates in a window around the alignment frontier (channel-
    compatible) + top-J LM words (fluency/deletion bridges), deduped by SURFACE id. Each word's score
    is ``lm_temp * lm_logprob + dZ`` where ``dZ`` is the forward-mass increment of one
    ``_word_row_update`` row (using the candidate's surface channel column ``emit_full[:, surf]`` with
    the per-particle word-action offset ``lp_sub + (lp_copy-lp_sub)*copy_mask[m, surf]`` added, and
    the per-particle ``wdel``/``wins`` -- all trivial when the word-action channel is OFF). The LM
    log-prob is ``lmlog[first_token]`` for a single-token candidate (the cheap one-forward path) or the
    precomputed chain-rule ``mt_chain[mt]`` for a multi-token one. The incremental importance weight is
    ``logsumexp(scores)`` (independent of the draw). EOS reads the terminal full-consumption mass
    ``alpha[M]``.

    Returns ``(cand_span [Kw, T_max], cand_len [Kw], emit_cols [M, Kw], scores [Kw+1])``. With
    ``n_mt == 0`` and ``T_max == 1`` this reduces exactly to the certified single-token scoring (the
    span is just ``[first_token]``, len 1, ``surf == token``), so the exact-enumeration gates hold.

    ``lm_temp`` (lambda) tempers the LM PRIOR (target ``P_LM^lm_temp * P_channel``); ``word_mask``
    restricts the top-J LM bridge pool to lexical words. There is no explicit INSERT action -- a
    spurious observed word is a channel event marginalized inside ``_word_row_update``'s insertion
    sweep (cost ``WINS``) with the band for reach (see ``planning/PAIRHMM_RBSMC_PLAN.md`` A3).
    """
    Z = logsumexp(log_alpha)
    fpos = jnp.clip(jnp.argmax(log_alpha), 0, M - 1)                       # alignment frontier
    widx = jnp.clip(fpos + offs, 0, M - 1)                                # window observed positions
    win_first = emit_first[widx].reshape(-1)
    win_surf = emit_surf[widx].reshape(-1)
    win_mt = emit_mtidx[widx].reshape(-1)
    lm_word = jnp.where(jnp.arange(lmlog.shape[0]) < emit_vocab, lmlog, -jnp.inf)
    lm_word = lm_word.at[eos_id].set(-jnp.inf)                             # EOS handled separately
    if word_mask is not None:                                             # restrict bridges to words
        lm_word = lm_word.at[:emit_vocab].set(
            jnp.where(word_mask, lm_word[:emit_vocab], -jnp.inf))
    top_j = jax.lax.top_k(lm_word, J)[1]
    cand_first = jnp.concatenate([win_first, top_j])                      # (Kw,) first token of each cand
    cand_surf = jnp.concatenate([win_surf, top_j])                       # top-J: surface == token id
    cand_mt = jnp.concatenate([win_mt, jnp.full((J,), -1, win_mt.dtype)])
    kw = cand_first.shape[0]
    valid = cand_first >= 0
    cand_surf_c = jnp.clip(cand_surf, 0, emit_full.shape[1] - 1)
    earlier_eq = (cand_surf[:, None] == cand_surf[None, :]) & valid[None, :] & jnp.tril(
        jnp.ones((kw, kw), bool), -1)
    valid = valid & ~jnp.any(earlier_eq, axis=1)                          # keep first occurrence

    emit_cols = emit_full[:, cand_surf_c]                                 # (M, Kw)
    # Word-action offset (plan WORD_ACTION_CHANNEL_PLAN sec 3.1): a cell (obs word m, candidate surf)
    # is a COPY iff surf matches m's observed word UP TO CASE (``copy_mask[m, surf]``), paying
    # ``log p_copy``; otherwise a substitution, paying ``log p_sub``. Case-insensitivity is essential --
    # a sentence-initial 'she'->'She' is a copy, not a sub (the OFF char DP is case-insensitive too).
    # The form (edited-char cost) is already in ``emit_full`` (the COPY_LP=0 form table when ON). OFF:
    # lp_copy=lp_sub=0 -> offset is identically 0 -> bit-identical to the certified path.
    is_copy = copy_mask[jnp.arange(M)[:, None], cand_surf_c[None, :]]     # (M, Kw)
    emit_cols = emit_cols + lp_sub + (lp_copy - lp_sub) * is_copy

    def cand_dZ(col):
        return logsumexp(band_mask(_word_row_update(log_alpha, col, wdel, wins), t_new)) - Z

    dZ = jax.vmap(cand_dZ, in_axes=1)(emit_cols)
    cand_first_c = jnp.clip(cand_first, 0, emit_vocab - 1)
    if n_mt > 0:
        lm_part = jnp.where(cand_mt >= 0, mt_chain[jnp.clip(cand_mt, 0, n_mt - 1)],
                            lmlog[cand_first_c])
    else:
        lm_part = lmlog[cand_first_c]
    score_word = jnp.where(valid, lm_temp * lm_part + dZ, -jnp.inf)
    score_eos = lm_temp * lmlog[eos_id] + (log_alpha[M] - Z)
    scores = jnp.concatenate([score_word, score_eos[None]])

    single_span = jnp.zeros((kw, T_max), jnp.int32).at[:, 0].set(cand_first_c.astype(jnp.int32))
    single_len = jnp.ones((kw,), jnp.int32)
    if n_mt > 0:
        mt_sel = jnp.clip(cand_mt, 0, n_mt - 1)
        cand_span = jnp.where((cand_mt >= 0)[:, None], mt_span[mt_sel], single_span)
        cand_len = jnp.where(cand_mt >= 0, mt_len[mt_sel], single_len)
    else:
        cand_span, cand_len = single_span, single_len
    cand_surf_out = cand_surf_c.astype(jnp.int32)        # surface id of each candidate (for word_surf)
    return cand_span, cand_len, cand_surf_out, emit_cols, scores


def _make_kernel(seed_len, M, band, T_max, LCTX, Wmax):
    ks = jnp.arange(M + 1)

    def band_mask(log_alpha, t):
        if band is None:                                                 # toy: no band (identity)
            return log_alpha
        return jnp.where(jnp.abs(ks - t) <= band, log_alpha, -jnp.inf)

    @genjax.gen
    def kernel(state, cand_span, cand_len, cand_surf, emit_cols, scores, wdel, wins):
        ctx_buf, ctx_len, n_words, word_len, word_surf, log_alpha, done = state
        action = genjax.categorical(scores) @ "action"
        incr = logsumexp(scores)
        incr = jnp.where(done, 0.0, incr)
        incr = jnp.where(jnp.isnan(incr), -jnp.inf, incr)                # -inf - -inf (dead) -> -inf
        _ = factor(incr) @ "ev"

        kw = scores.shape[0] - 1
        chose_eos = action == kw
        ci = jnp.clip(action, 0, kw - 1)
        span = cand_span[ci]                                             # (T_max,) chosen token span
        span_len = cand_len[ci]
        surf = cand_surf[ci]                                            # chosen word's channel surface id
        col = emit_cols[:, ci]

        advance = (~done) & (~chose_eos)
        t_after = n_words + 1                                            # band uses WORD count, not tokens
        new_alpha = band_mask(_word_row_update(log_alpha, col, wdel, wins), t_after)
        pos = jnp.clip(ctx_len + jnp.arange(T_max), 0, LCTX - 1)         # splice the span's tokens
        write = advance & (jnp.arange(T_max) < span_len)
        ctx_buf2 = ctx_buf.at[pos].set(jnp.where(write, span.astype(jnp.int32), ctx_buf[pos]))
        wpos = jnp.clip(n_words, 0, Wmax - 1)
        word_len2 = word_len.at[wpos].set(jnp.where(advance, span_len.astype(jnp.int32),
                                                    word_len[wpos]))
        word_surf2 = word_surf.at[wpos].set(jnp.where(advance, surf.astype(jnp.int32),
                                                      word_surf[wpos]))   # for the rejuv move (R4)
        return (ctx_buf2,
                jnp.where(advance, ctx_len + span_len, ctx_len),
                jnp.where(advance, n_words + 1, n_words),
                word_len2, word_surf2,
                jnp.where(advance, new_alpha, log_alpha),
                done | chose_eos)

    return kernel, band_mask


def _uniq_frac(ctx_buf, ctx_len, seed_len):
    """Cloud degeneracy: fraction of DISTINCT intended trajectories among P particles (host-side; the
    GOAL1 Step-0 metric -- low => degenerate => dedup helps). Only called when rejuv_stats requested."""
    P = ctx_buf.shape[0]
    rows = {tuple(int(t) for t in ctx_buf[p][seed_len:int(ctx_len[p])]) for p in range(P)}
    return len(rows) / P


def _record_step(model, state, log_w, seed_len, t, ess, resampled, logZ, rejuv, final=False,
                 topk=8, max_particles=512):
    """One host-side snapshot of the particle cloud for the step-trace viewer (only built when a
    ``trace`` list is passed to :func:`run` -- it reads state, never alters the certified math). See
    ``planning/TRACE_SCHEMA.md`` for the field contract. ``dist``/``frontier`` are EXACT over all P;
    ``particles`` is the heaviest ``max_particles`` (the long tail is the residual)."""
    ctx_buf, ctx_len, _n_words, _word_len, _word_surf, log_alpha, done = state
    P = ctx_buf.shape[0]
    w = np.asarray(jax.nn.softmax(log_w))                            # filtering weights (post-step)
    cl = np.asarray(ctx_len)
    dn = np.asarray(done)
    frontier = np.asarray(jnp.argmax(log_alpha, axis=1))            # observed words consumed per particle
    cb = np.asarray(ctx_buf)
    prefixes = [model.decode_ids(tuple(int(x) for x in cb[p][seed_len:cl[p]])) for p in range(P)]

    dw, dc = defaultdict(float), defaultdict(int)                   # distinct-latent distribution
    fw = defaultdict(float)                                         # alignment-frontier histogram
    for p in range(P):
        dw[prefixes[p]] += float(w[p]); dc[prefixes[p]] += 1
        fw[int(frontier[p])] += float(w[p])
    items = sorted(dw, key=lambda s: -dw[s])
    dist = [[s, dw[s], dc[s]] for s in items[:topk]]
    res = items[topk:]
    dist_residual = [sum(dw[s] for s in res), sum(dc[s] for s in res)]
    frontier_hist = sorted(([k, v] for k, v in fw.items()), key=lambda kv: -kv[1])

    order = np.argsort(-w)[:max_particles]                          # heaviest particles for the dump
    particles = [{"p": int(p), "weight": float(w[p]), "k": int(frontier[p]),
                  "done": bool(dn[p]), "prefix": prefixes[p]} for p in order]
    return {"t": int(t), "ess": float(ess), "resampled": bool(resampled), "final": bool(final),
            "logZ": float(logZ), "n_done": int(dn.sum()), "n_unique": len(dw),
            "dist": dist, "dist_residual": dist_residual, "frontier": frontier_hist,
            "particles": particles, "rejuv": rejuv}


def _theta_to_costs(theta, enable_indel, wins_vec):
    """``theta`` (P,4) over the word action ``(copy, sub, insert, delete)`` -> the per-particle channel
    action costs ``(lp_copy (P,), lp_sub (P,), wdel_p (P,), wins_p (P,M))``. ``wins_vec`` (M,) is the
    per-observed-word CONTENT cost (``-unigram_surprisal``); the insertion RATE is ``log p_insert`` (so
    ``wins_p = log p_insert + content``). ``enable_indel=False`` masks delete/insert to -inf."""
    P = theta.shape[0]
    neg = jnp.float32(-jnp.inf)
    lp = jnp.log(theta)
    lp_copy, lp_sub = lp[:, 0], lp[:, 1]
    wdel_p = jnp.where(enable_indel, lp[:, 3], neg) * jnp.ones(P)                  # log p_delete (P,)
    wins_p = jnp.where(enable_indel, lp[:, 2][:, None] + wins_vec[None, :], neg)  # log p_insert + content
    return lp_copy, lp_sub, wdel_p, wins_p


def _action_counts(word_surf, word_len, copy_mask, M):
    """Per-particle word-action counts ``(n_copy, n_sub, n_ins, n_del)`` (P,4) from a POSITIONAL 1:1
    alignment (intended word i <-> observed word i). EXACT for substitution-dominated, near-deterministic
    alignments -- the calibration battery's one-edit items (plan WORD_ACTION_CHANNEL_PLAN sec 3.3) --
    and approximate when an insertion/deletion shifts the alignment mid-sentence (a documented
    general-text refinement: the true counts need the sampled/MAP alignment). Drives the Dirichlet-
    conjugate theta refresh (sec 5.4). Copy vs sub is CASE-INSENSITIVE via ``copy_mask`` (a capitalization
    is a copy, not a substitution), consistent with the emission offset."""
    Wmax = word_surf.shape[1]
    idx = jnp.arange(Wmax)[None, :]
    n_words = jnp.sum(word_len > 0, axis=1)                         # (P,) intended word count
    active = idx < n_words[:, None]
    aligned = idx < M                                              # positional slot maps to an observed word
    slot = jnp.clip(jnp.arange(Wmax), 0, M - 1)                    # (Wmax,) observed word each slot maps to
    ws_c = jnp.clip(word_surf, 0, copy_mask.shape[1] - 1)
    is_cp = copy_mask[slot[None, :], ws_c]                         # (P,Wmax) word_surf is a (case-insens) COPY
    is_copy = active & aligned & is_cp
    is_sub = active & aligned & (~is_cp)
    is_del = active & (~aligned)                                   # intended word past the observed length
    n_copy = jnp.sum(is_copy, axis=1)
    n_sub = jnp.sum(is_sub, axis=1)
    n_del = jnp.sum(is_del, axis=1)
    n_ins = jnp.maximum(0, M - n_words)                           # observed words past the intended length
    return jnp.stack([n_copy, n_sub, n_ins, n_del], axis=1).astype(jnp.float32)


# The per-particle channel forward carry now lives in word_dp.channel_carry (one source of truth shared
# with the rejuvenation sweep); the filter's theta-refresh calls it directly below.


def run(observed, key, model, P=4000, wdel=jnp.log(0.1), wins=jnp.log(0.05), slack=3, band=None,
        max_dist=2, Ke=6, J=4, cwin=1, proposal="caprop", enable_indel=True,
        rejuv="off", rejuv_pool=None, rejuv_lookback=3, rejuv_stats=None, trace=None, rejuv_dedup=False,
        lm_temp=1.0, action_alpha=None, channel=None):
    """Sequential RB-SMC over intended words; the word alignment ``alpha`` is marginalized.

    Returns ``(state, log_w, logZ, seed_len)``. ``proposal="caprop"`` is the fully-adapted kernel;
    ``"bootstrap"`` proposes from the LM prior (baseline). ``band=None`` disables the band mask
    (the toy); Pythia passes an integer band, which also gives insertions/deletions their reach
    (consumption ``k`` may run up to ``band`` ahead of / behind the emission count).

    ``channel`` names the noise model: ``"word_action"`` is THE model (the per-word Dirichlet action
    channel below); ``"char_copy"`` is the deprecated bundled char channel, kept ONLY as the
    exact-enumeration **certification anchor** -- ``test_pairhmm_exact`` proves the SMC/DP machinery is
    bit-identical to brute-force enumeration through it, and it is the concentrated-alpha limit, not a
    deployment option. ``channel=None`` (default) infers the channel from ``action_alpha`` (set ->
    ``"word_action"``, ``None`` -> ``"char_copy"``), a back-compat shim for the retired ``ON`` boolean.

    ``action_alpha`` (default ``None`` -- the OFF / char-copy anchor path) carries the **word-action
    channel**'s (plan WORD_ACTION_CHANNEL_PLAN) length-4 Dirichlet concentration over the per-word action
    ``(copy, sub, insert, delete)``. When set, a latent ``theta ~ Dirichlet(action_alpha)`` is drawn per
    particle and the channel score factors into a word-level action cost + a form cost: the emission
    table is built from the base-rate-DECOUPLED FORM channel (``model.channel_form``, COPY_LP=0) and
    each emission cell pays ``log p_copy`` (verbatim copy of that observed word) or ``log p_sub`` (a
    substitution); deletion pays ``log p_delete`` (the WDEL arc) and insertion ``log p_insert + wins``
    (``wins`` is then the per-word CONTENT cost ``-unigram_surprisal``, the rate coming from theta).
    With ``action_alpha=None`` every value is bit-identical to the certified char-copy filter (lp=0
    offset, the bundled ``channel_logpdf``, global ``WDEL/WINS``). theta is carried per particle and
    refreshed by the (Dirichlet-conjugate) rejuvenation move when ``rejuv="gibbs"``.

    Intended words may span multiple BPE tokens (Phase D): ``_build_candidates`` builds the
    candidate inventory (a candidate = a token span + a surface id) and ``T_max`` = the max span
    length; the per-step ``mt_chain`` scores multi-token candidate spans by chain-rule via the
    injected ``tail_logprobs``. With no multi-token candidates (``n_mt == 0``, ``T_max == 1``) the
    run is bit-identical to the certified single-token filter.

    All three edit types are handled without any explicit edit "action": substitution and deletion
    by ``_word_row_update``'s diag/up terms, and *insertion* (a spurious observed word) by that same
    row's WINS-costed insertion sweep -- a channel event marginalized inside the DP, not an LM
    action (see ``_caprop_scores``). Every SMC step is thus a clean LM word/EOS choice.

    ``rejuv`` (default ``"off"`` -- the certified path; the exact-enumeration gates run with it off)
    enables a flag-gated Gibbs/SMCP3 rejuvenation sweep (REJUV_KV_REDESIGN_PLAN.md R2). ``"gibbs"``
    runs a windowed full-conditional sweep (``pairhmm_rejuv.make_sweep``) over the last
    ``rejuv_lookback`` words **after each resample** on the equal-weight cloud, recomputes ``log_alpha``,
    and folds the move's SMCP3 weight into ``log_w`` BEFORE the next resample so mass can flow
    (REJUV_GOAL3). ``rejuv_pool`` is ``(pool_tok, pool_len)`` (model-specific candidate spans, e.g.
    ``pairhmm_rejuv.build_pool``). ``rejuv_stats`` (optional dict) accumulates LM-forward + degeneracy
    counters for the R2 cost measurement. The forward filter itself is untouched -- rejuvenation is
    additive and only runs inside the ``rejuv == "gibbs"`` branch. Multi-token candidates are handled
    (R4). On the WORD-ACTION channel (``action_alpha`` set) the sweep is theta-aware -- it scores
    candidates against the live per-particle channel via ``theta_costs`` -- and each resample event runs
    the word sweep THEN a Dirichlet-conjugate ``theta`` refresh on the corrected parse (plan
    WORD_ACTION_REJUV_PLAN sec 3); on the char-copy channel it is the certified zero-action sweep.

    ``lm_temp`` (lambda) tempers the LM PRIOR in BOTH the caprop step and the rejuvenation move: the
    target posterior is ``P_LM^lm_temp * P_channel`` (see :func:`_caprop_scores`). ``1.0`` = untempered
    (the certified path); ``< 1`` flattens the LM's over-confident preferences so plausible inputs are
    read more literally (curbs over-editing). Applies to ``proposal="caprop"`` (production); the
    ``"bootstrap"`` baseline samples from the raw LM and is meaningful only at ``lm_temp=1.0``.
    """
    # Channel selector (plan WORD_ACTION_REJUV_PLAN Phase 3): ``"word_action"`` is THE model -- the
    # per-word Dirichlet action channel; ``"char_copy"`` is the deprecated bundled char channel, kept
    # ONLY as the exact-enumeration certification anchor (``test_pairhmm_exact``). ``channel=None`` infers
    # it from ``action_alpha`` -- a pure rename of the retired ``ON = action_alpha is not None`` boolean,
    # so every existing caller is bit-identical with no edit.
    if channel is None:
        channel = "word_action" if action_alpha is not None else "char_copy"
    if channel not in ("word_action", "char_copy"):
        raise ValueError(f"channel must be 'word_action' or 'char_copy', got {channel!r}")
    if channel == "word_action" and action_alpha is None:
        raise ValueError("channel='word_action' requires action_alpha (the Dirichlet action concentration)")
    if channel == "char_copy" and action_alpha is not None:
        raise ValueError("channel='char_copy' is the zero-action anchor; do not pass action_alpha")
    ON = channel == "word_action"                                       # word-action channel?
    rj_theta = ON and rejuv != "off"                                    # Dirichlet-conjugate theta refresh (5.4)
    seed_ids = list(model.seed_ids)
    seed_len = len(seed_ids)
    obs_words = model.obs_words(observed)
    M = len(obs_words)
    obs_spans = model.obs_spans(observed) if model.obs_spans is not None else [None] * M
    obs_char = jnp.stack([jnp.asarray(model.char_ids(w)[0], jnp.int32) for w in obs_words])  # (M,Lc)
    chan_fn = model.channel_form if ON else model.channel_logpdf        # FORM table (COPY_LP=0) when ON
    emit_full = jax.vmap(jax.vmap(chan_fn, in_axes=(None, 0, 0)),
                         in_axes=(0, None, None))(obs_char, model.vocab_char, model.vocab_clen)
    # copy_mask (M, Vc_aug): the CASE-INSENSITIVE copy classifier for the word-action emission offset
    # (a capitalization 'she'->'She' is a copy, not a sub -- the bug the word-action path silently had).
    (emit_first, emit_surf, emit_mtidx, mt_span, mt_len, emit_full, copy_mask, T_max, n_mt) = _build_candidates(
        model, obs_words, obs_spans, obs_char, emit_full, max_dist, Ke, channel_fn=chan_fn)
    WDEL = wdel if enable_indel else -jnp.inf
    WINS = wins if enable_indel else -jnp.inf
    offs = jnp.arange(-cwin, cwin + 1)
    eos_id, Vc = model.eos_id, model.emit_vocab
    Wmax = M + slack
    LCTX = seed_len + Wmax * T_max + 1

    # Per-particle action costs (plan sec 3.2). OFF: lp_copy=lp_sub=0 (zero emission offset) and the
    # global WDEL/WINS broadcast over P -> bit-identical to the certified path. ON: draw theta ~
    # Dirichlet(action_alpha) per particle and read off log p_copy/p_sub/p_insert/p_delete; the insertion
    # rate replaces ins_rate (so ``wins`` carries only the per-word content cost -unigram_surprisal).
    wins_vec = jnp.broadcast_to(jnp.asarray(WINS, jnp.float32), (M,))    # per-observed-word insertion cost
    if ON:
        key, tkey = jax.random.split(key)
        action_alpha = jnp.asarray(action_alpha, jnp.float32)
        theta = jax.random.dirichlet(tkey, action_alpha, shape=(P,))                 # (P,4)
        lp_copy, lp_sub, wdel_p, wins_p = _theta_to_costs(theta, enable_indel, wins_vec)
    else:
        theta = None
        lp_copy = jnp.zeros(P); lp_sub = jnp.zeros(P)
        wdel_p = jnp.broadcast_to(jnp.asarray(WDEL, jnp.float32), (P,))
        wins_p = jnp.broadcast_to(wins_vec, (P, M))

    kernel, band_mask = _make_kernel(seed_len, M, band, T_max, LCTX, Wmax)
    constraint = ChoiceMap.d({"ev": jnp.float32(0.0)})
    # a0 per particle (leading spurious words): a0[k] = sum of the first k observed-word insertion costs.
    # OFF: every particle's wins_p row is identical -> a0 matches the old broadcast a0 exactly.
    a0p = jax.vmap(lambda wn: band_mask(
        jnp.concatenate([jnp.zeros((1,), wn.dtype), jnp.cumsum(wn)]), 0))(wins_p)       # (P, M+1)
    a0 = a0p[0]                                                          # representative row (rejuv ctx)
    ctx0 = jnp.full((P, LCTX), eos_id, jnp.int32)
    if seed_len:
        ctx0 = ctx0.at[:, :seed_len].set(jnp.array(seed_ids, jnp.int32))
    state = (ctx0, jnp.full(P, seed_len, jnp.int32), jnp.zeros(P, jnp.int32),
             jnp.zeros((P, Wmax), jnp.int32), jnp.zeros((P, Wmax), jnp.int32),
             a0p, jnp.zeros(P, bool))   # +word_len, +word_surf (R4)
    log_w = jnp.zeros(P)
    logZ = 0.0

    # Multi-token LM scorer (Phase D): the injected KV/uncached chain-rule over a candidate's tokens.
    tail_fn = None
    if n_mt > 0:
        from genjax_port import pairhmm_rejuv as RJ
        tail_fn = model.tail_logprobs or (
            lambda cb, cl, t, tl: RJ._tail_chain_uncached(model.lm_fn, cb, cl, t, tl))
        mt_span_b = jnp.broadcast_to(mt_span[None], (P, n_mt, T_max))
        mt_len_b = jnp.broadcast_to(mt_len[None], (P, n_mt))

    # Flag-gated rejuvenation (R2): build the sweep ONCE (jitted step reused across resample events).
    # rejuv_dedup (R3 item 1b) runs the sweep's tail_fn on the unique buffers only (host-side) -- EXACT,
    # cuts the sweep's dominant per-particle prefill cost (~linear in P). A DedupStats tracks rows saved.
    rj_sweep, rj_dedup_stats = None, None
    if rejuv == "gibbs":                  # word sweep for BOTH channels: theta-aware via theta_costs when ON,
        #                                   char-copy / zero-action (theta_costs=None) when OFF (Phase 2 de-fork)
        from genjax_port import pairhmm_rejuv as RJ
        # R4: t_max = the forward's T_max; the pool carries multi-token spans + surface ids, built from
        # the SAME candidate inventory (so its surfaces index the SAME augmented emit_full).
        rj_pool = _rejuv_pool_from_inventory(emit_first, emit_surf, emit_mtidx, mt_span, mt_len,
                                             M, Wmax, T_max)
        rj_ctx = RJ.RejuvCtx(model, emit_full, a0, M, seed_len, Wmax, WDEL, WINS, band, T_max, lm_temp)
        mt_tokens = (rejuv_lookback + 1) * T_max + 1                     # suffix-tail budget in TOKENS
        rj_sweep = RJ.make_sweep(rj_ctx, *rj_pool, max_tail=mt_tokens, dedup=rejuv_dedup)
        if rejuv_dedup:
            from genjax_port import cache_dedup
            rj_dedup_stats = cache_dedup.DedupStats()
        if rejuv_stats is not None:
            rejuv_stats.update(P=P, Kt=rj_pool[0].shape[1] + 1, max_tail=mt_tokens,
                               filter_lm_calls=0, sweep_prefills=0, sweep_tail_steps=0, uniq_frac=[],
                               dedup_rows_in=0, dedup_rows_computed=0)

    word_mask = model.word_mask
    def _assemble(n_words, log_alpha, lmlog, mt_chain, lp_copy, lp_sub, wdel_p, wins_p):
        return jax.vmap(
            lambda la, lm, mtc, nw, lpc, lps, wd, wn: _caprop_scores(
                la, lm, mtc, emit_first, emit_surf, emit_mtidx, mt_span, mt_len, emit_full, offs, J, M,
                band_mask, nw + 1, eos_id, Vc, wd, wn, word_mask, lm_temp, T_max, n_mt,
                lpc, lps, copy_mask),
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0))(log_alpha, lmlog, mt_chain, n_words,
                                              lp_copy, lp_sub, wdel_p, wins_p)

    @jax.jit
    def extend_caprop(keys, ctx_buf, ctx_len, n_words, word_len, word_surf, log_alpha, done,
                      cand_span, cand_len, cand_surf, emit_cols, scores, wdel_p, wins_p):
        def one(k, cb, cl, nw, wl, ws_, la, dn, csp, cln, csf, ec, sc, wd, wn):
            tr, w = kernel.importance(k, constraint,
                                      ((cb, cl, nw, wl, ws_, la, dn), csp, cln, csf, ec, sc, wd, wn))
            rv = tr.get_retval()
            return rv[0], rv[1], rv[2], rv[3], rv[4], rv[5], rv[6], w

        cb, cl, nw, wl, ws2, la, dn, ws = jax.vmap(one)(
            keys, ctx_buf, ctx_len, n_words, word_len, word_surf, log_alpha, done,
            cand_span, cand_len, cand_surf, emit_cols, scores, wdel_p, wins_p)
        return (cb, cl, nw, wl, ws2, la, dn), ws

    @jax.jit
    def extend_bootstrap(keys, ctx_buf, ctx_len, n_words, word_len, word_surf, log_alpha, done, lmlog,
                         lp_copy, lp_sub, wdel_p, wins_p):
        def one(k, cb, cl, nw, wl, wsf, la, dn, lm, lpc, lps, wd, wn):
            Z = logsumexp(la)
            s = jax.random.categorical(k, lm)
            chose_eos = s == eos_id
            w_id = jnp.where(chose_eos, 0, s)
            wclip = jnp.clip(w_id, 0, Vc - 1)
            col = emit_full[:, wclip] + lps + (lpc - lps) * copy_mask[:, wclip]  # word-action offset (case-insens copy)
            new_alpha = band_mask(_word_row_update(la, col, wd, wn), nw + 1)
            incr = jnp.where(chose_eos, 0.0, logsumexp(new_alpha) - Z)
            advance = (~dn) & (~chose_eos)
            incr = jnp.where(dn, 0.0, incr)
            incr = jnp.where(jnp.isnan(incr), -jnp.inf, incr)
            cb2 = jnp.where(advance, cb.at[cl].set(w_id.astype(jnp.int32)), cb)
            wpos = jnp.clip(nw, 0, Wmax - 1)
            wl2 = wl.at[wpos].set(jnp.where(advance, jnp.int32(1), wl[wpos]))
            wsf2 = wsf.at[wpos].set(jnp.where(advance, w_id.astype(jnp.int32), wsf[wpos]))
            return (cb2, jnp.where(advance, cl + 1, cl), jnp.where(advance, nw + 1, nw), wl2, wsf2,
                    jnp.where(advance, new_alpha, la), dn | chose_eos), incr

        states, ws = jax.vmap(one)(keys, ctx_buf, ctx_len, n_words, word_len, word_surf, log_alpha,
                                   done, lmlog, lp_copy, lp_sub, wdel_p, wins_p)
        return states, ws

    for s in range(M + slack):
        ctx_buf, ctx_len, n_words, word_len, word_surf, log_alpha, done = state
        lmlog = model.lm_fn(ctx_buf, ctx_len)                            # batched LM call (P, vocab)
        if rejuv_stats is not None:
            rejuv_stats["filter_lm_calls"] += 1
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, P)
        if proposal == "caprop":
            if n_mt > 0:                                                 # chain-rule scores for MT cands
                mt_chain = tail_fn(ctx_buf, ctx_len, mt_span_b, mt_len_b)
            else:
                mt_chain = jnp.zeros((P, 1))
            cand_span, cand_len, cand_surf, emit_cols, scores = _assemble(
                n_words, log_alpha, lmlog, mt_chain, lp_copy, lp_sub, wdel_p, wins_p)
            state, incr = extend_caprop(keys, ctx_buf, ctx_len, n_words, word_len, word_surf,
                                        log_alpha, done, cand_span, cand_len, cand_surf, emit_cols,
                                        scores, wdel_p, wins_p)
        else:
            state, incr = extend_bootstrap(keys, ctx_buf, ctx_len, n_words, word_len, word_surf,
                                           log_alpha, done, lmlog, lp_copy, lp_sub, wdel_p, wins_p)
        log_w = log_w + incr
        ess_pre = float(_ess(log_w))     # the ESS that triggers/avoids resampling (recorded for the viz)
        resampled, rejuv_info = False, None
        if ess_pre < 0.5 * P:            # ESS-triggered resampling keeps early diversity
            resampled = True
            logZ = logZ + logsumexp(log_w) - jnp.log(P)
            key, sub = jax.random.split(key)
            anc = jax.random.categorical(sub, log_w, shape=(P,))
            state = jax.tree_util.tree_map(lambda a: a[anc], state)
            lp_copy, lp_sub = lp_copy[anc], lp_sub[anc]   # per-particle action costs follow the ancestors
            wdel_p, wins_p = wdel_p[anc], wins_p[anc]
            if theta is not None:
                theta = theta[anc]
            log_w = jnp.zeros(P)
            if rj_sweep is not None:     # SWEEP-THEN-REFRESH (plan sec 3): run the theta-aware word move
                ctx_buf, ctx_len, n_words, word_len, word_surf, _, done = state  # pre-resample (GOAL3 b)
                hi = min(s + 1, M + slack)                   # frontier word count (upper bound)
                lo = max(0, hi - rejuv_lookback)
                # theta_costs: the current per-particle action costs so the sweep scores every candidate
                # against the LIVE channel (theta). OFF -> None -> char-copy/zero-action (bit-identical).
                theta_costs = None
                if ON:
                    a0p = jax.vmap(lambda wn: band_mask(
                        jnp.concatenate([jnp.zeros((1,), wn.dtype), jnp.cumsum(wn)]), 0))(wins_p)  # (P,M+1)
                    theta_costs = (lp_copy, lp_sub, wdel_p, wins_p, a0p, copy_mask)
                key, sub = jax.random.split(key)
                cb, cl2, wl2, ws2, la, mlw = rj_sweep(sub, ctx_buf, ctx_len, word_len, word_surf,
                                                      positions=range(lo, hi), done=done,
                                                      dedup_stats=rj_dedup_stats, theta_costs=theta_costs)
                state = (cb, cl2, n_words, wl2, ws2, la, done)   # word count fixed; spans/lengths may move
                log_w = log_w + mlw
                rejuv_info = {"words": [int(lo), int(hi)], "ess_after": float(_ess(log_w)),
                              "mean_abs_w": float(jnp.mean(jnp.abs(mlw)))}
                if rejuv_stats is not None:                   # KV scorer: 1 shared prefill/word/particle
                    window = hi - lo                          # (full forward) + cheap tail steps
                    rejuv_stats["sweep_prefills"] += window
                    rejuv_stats["sweep_tail_steps"] += window * rejuv_stats["max_tail"]
                    rejuv_stats["uniq_frac"].append(_uniq_frac(cb, ctx_len, seed_len))
                    if rj_dedup_stats is not None:            # R3 1b: actual unique rows fed to tail_fn
                        rejuv_stats["dedup_rows_in"] = rj_dedup_stats.rows_in
                        rejuv_stats["dedup_rows_computed"] = rj_dedup_stats.rows_computed
                if rj_theta:             # ...THEN refresh: Dirichlet-conjugate theta on the CORRECTED parse
                    ctx_buf, ctx_len, n_words, word_len, word_surf, _, done = state  # post-move (sec 3)
                    counts = _action_counts(word_surf, word_len, copy_mask, M)   # (P,4) positional action counts
                    key, sub = jax.random.split(key)
                    theta = jax.random.dirichlet(sub, action_alpha + counts)     # theta | counts ~ Dir(alpha+counts)
                    lp_copy, lp_sub, wdel_p, wins_p = _theta_to_costs(theta, enable_indel, wins_vec)
                    a0p = jax.vmap(lambda wn: band_mask(                          # new leading-spurious init per theta
                        jnp.concatenate([jnp.zeros((1,), wn.dtype), jnp.cumsum(wn)]), 0))(wins_p)
                    la = channel_carry(a0p, emit_full, band, M, word_surf, word_len,   # log_alpha consistent
                                       lp_copy, lp_sub, wdel_p, wins_p, copy_mask)      # with the new theta
                    state = (ctx_buf, ctx_len, n_words, word_len, word_surf, la, done)
                    # The word move gives the particle the escape route the refresh-alone path lacked: the
                    # sweep can RESTORE a dropped word, then the conjugate refresh re-estimates theta on the
                    # corrected parse (theta now reflects the data: clean context -> high p_copy).
                    rejuv_info["theta_mean"] = [round(float(x), 3) for x in jnp.mean(theta, axis=0)]
        if trace is not None:            # per-step snapshot of the cloud (post-extend/resample/rejuv)
            trace.append(_record_step(model, state, log_w, seed_len, s, ess_pre, resampled, logZ,
                                      rejuv_info))

    # Terminal full-consumption correction: caprop folds alpha[M] into the EOS candidate, so EOS'd
    # particles already paid it; both proposals still need it for particles live at the budget (else
    # raw forward mass over-rewards long junk parses). bootstrap never folds it -> applies to all.
    _, _, _, _, _, log_alpha, done = state
    need_term = jnp.ones_like(done) if proposal == "bootstrap" else ~done
    term = jnp.where(need_term, log_alpha[:, M] - logsumexp(log_alpha, axis=1), 0.0)
    term = jnp.where(jnp.isnan(term), -jnp.inf, term)
    log_w = log_w + term
    logZ = logZ + logsumexp(log_w) - jnp.log(P)
    if trace is not None:                # final snapshot: terminal-corrected weights (the true posterior)
        trace.append(_record_step(model, state, log_w, seed_len, len(trace), float(_ess(log_w)),
                                   False, logZ, None, final=True))
    return state, log_w, float(logZ), seed_len


def decode(state, log_w, model, skip=0, key=jax.random.PRNGKey(0), top=3):
    """Most-probable intended sentences from the weighted particle cloud, by EXACT posterior weight.

    Sums the normalized importance weights ``softmax(log_w)`` of all particles that decode to the same
    sentence -- the identical deterministic estimator the JSON trace reports in ``_record_step``'s
    ``dist`` (so stdout and the JSON agree exactly). Earlier this drew ``P`` categorical resamples from
    ``log_w`` and tallied frequencies, a 1/P-quantized Monte-Carlo estimate whose sampling noise could
    flip near-tied hypotheses; ``key`` is now unused but kept for signature compatibility."""
    del key                                                          # no longer resampled
    ctx_buf, ctx_len = state[0], state[1]
    w = np.asarray(jax.nn.softmax(log_w))
    mass = defaultdict(float)                                        # exact weighted posterior per latent
    for p in range(ctx_buf.shape[0]):
        s = model.decode_ids(tuple(int(t) for t in ctx_buf[p][skip:int(ctx_len[p])]))
        mass[s] += float(w[p])
    return sorted(mass.items(), key=lambda kv: -kv[1])[:top]
