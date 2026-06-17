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
"""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import genjax
from genjax import ChoiceMap

from genjax_port.genjax_model import factor
from genjax_port.poc_word_indel import _word_row_update, _wins_only_row, _ess


@dataclass
class PairHMMModel:
    """The model-specific injections that turn the generic filter into a concrete model.

    All inference (the kernel, the forward DP, resampling, the terminal correction) is generic;
    only these inputs differ between the toy bigram and Pythia.
    """

    lm_fn: Callable            # BATCHED (ctx_buf [P,LCTX], ctx_len [P]) -> logprobs [P, vocab]
    eos_id: int                # index of EOS in the lm_fn output row
    emit_vocab: int            # number of candidate-word columns (Vc) in the emission table
    vocab_char: jnp.ndarray    # (Vc, Lchar) char ids of every candidate word
    vocab_clen: jnp.ndarray    # (Vc,) char lengths
    channel_logpdf: Callable   # (obs_char_ids, intended_char_ids, n_x) -> channel logpdf
    char_ids: Callable         # word_str -> (list[int] padded char ids, n)
    candidate_ids: Callable    # (word_str, max_dist, Ke) -> list[int] candidate word ids
    obs_words: Callable        # observed_str -> list[word_str]
    decode_ids: Callable       # tuple[int] -> str (intended ids back to a sentence)
    seed_ids: Sequence[int] = ()  # context seed (toy: (); Pythia: [EOS] + prime tokens)


def _emit_table(model, obs_words, max_dist, Ke):
    """(M, Ke) padded candidate word ids per observed word; -1 = pad."""
    rows = []
    for w in obs_words:
        ids = list(model.candidate_ids(w, max_dist, Ke))[:Ke]
        rows.append(ids + [-1] * (Ke - len(ids)))
    return jnp.array(rows, jnp.int32)


def _caprop_scores(log_alpha, lmlog, n_emitted, emit_tab, emit_full, offs, J, M, band_mask,
                   t_new, eos_id, emit_vocab, WDEL, WINS, insert_action):
    """Channel-aware (fully-adapted) candidate scores: [word(Kw), INSERT(1), EOS(1)].

    Candidate set C = emission candidates in a window around the alignment frontier (channel-
    compatible) + top-J LM words (fluency/deletion bridges), deduped. Each word's score is
    ``lm logprob + dZ`` where ``dZ`` is the forward-mass increment of one DP row update; the
    incremental importance weight is ``logsumexp(scores)`` (independent of the draw). The INSERT
    slot consumes one spurious observed word, emitting no intended word; EOS reads the terminal
    full-consumption mass ``alpha[M]``.
    """
    Z = logsumexp(log_alpha)
    fpos = jnp.clip(jnp.argmax(log_alpha), 0, M - 1)                       # alignment frontier
    emit_ids = emit_tab[jnp.clip(fpos + offs, 0, M - 1)].reshape(-1)       # window of emission cands
    lm_word = jnp.where(jnp.arange(lmlog.shape[0]) < emit_vocab, lmlog, -jnp.inf)
    lm_word = lm_word.at[eos_id].set(-jnp.inf)                             # EOS handled separately
    top_j = jax.lax.top_k(lm_word, J)[1]
    cand = jnp.concatenate([emit_ids, top_j])
    kw = cand.shape[0]
    valid = cand >= 0
    cand_c = jnp.clip(cand, 0, emit_vocab - 1)
    earlier_eq = (cand[:, None] == cand[None, :]) & valid[None, :] & jnp.tril(
        jnp.ones((kw, kw), bool), -1)
    valid = valid & ~jnp.any(earlier_eq, axis=1)                          # keep first occurrence

    emit_cols = emit_full[:, cand_c]                                      # (M, Kw)

    def cand_dZ(col):
        return logsumexp(band_mask(_word_row_update(log_alpha, col, WDEL, WINS), t_new)) - Z

    dZ = jax.vmap(cand_dZ, in_axes=1)(emit_cols)
    score_word = jnp.where(valid, lmlog[cand_c] + dZ, -jnp.inf)

    dZ_insert = logsumexp(band_mask(_wins_only_row(log_alpha, WINS), t_new)) - Z
    allow_insert = insert_action & (jnp.argmax(log_alpha) > n_emitted)    # frontier ahead of emits
    score_insert = jnp.where(allow_insert, dZ_insert, -jnp.inf)
    score_eos = lmlog[eos_id] + (log_alpha[M] - Z)
    scores = jnp.concatenate([score_word, score_insert[None], score_eos[None]])
    return cand, emit_cols, scores


def _make_kernel(seed_len, M, band, WDEL, WINS):
    ks = jnp.arange(M + 1)

    def band_mask(log_alpha, t):
        if band is None:                                                 # toy: no band (identity)
            return log_alpha
        return jnp.where(jnp.abs(ks - t) <= band, log_alpha, -jnp.inf)

    @genjax.gen
    def kernel(state, cand, emit_cols, scores):
        ctx_buf, ctx_len, log_alpha, done = state
        action = genjax.categorical(scores) @ "action"
        incr = logsumexp(scores)
        incr = jnp.where(done, 0.0, incr)
        incr = jnp.where(jnp.isnan(incr), -jnp.inf, incr)                # -inf - -inf (dead) -> -inf
        _ = factor(incr) @ "ev"

        kw = cand.shape[0]
        chose_insert = action == kw
        chose_eos = action == kw + 1
        ci = jnp.clip(action, 0, kw - 1)
        w_id = jnp.where(chose_eos | chose_insert, jnp.int32(0), cand[ci])
        col = emit_cols[:, ci]

        advance_word = (~done) & (~chose_eos) & (~chose_insert)
        advance_insert = (~done) & chose_insert
        t_after = ctx_len + 1 - seed_len
        new_alpha_word = band_mask(_word_row_update(log_alpha, col, WDEL, WINS), t_after)
        new_alpha_insert = band_mask(_wins_only_row(log_alpha, WINS), t_after)
        new_alpha = jnp.where(chose_insert, new_alpha_insert, new_alpha_word)
        ctx_buf2 = jnp.where(advance_word, ctx_buf.at[ctx_len].set(w_id.astype(jnp.int32)), ctx_buf)
        return (ctx_buf2,
                jnp.where(advance_word, ctx_len + 1, ctx_len),
                jnp.where(advance_word | advance_insert, new_alpha, log_alpha),
                done | chose_eos)

    return kernel, band_mask


def run(observed, key, model, P=4000, wdel=jnp.log(0.1), wins=jnp.log(0.05), slack=3, band=None,
        max_dist=2, Ke=6, J=4, cwin=1, proposal="caprop", enable_indel=True, insert_action=True):
    """Sequential RB-SMC over intended words; the word alignment ``alpha`` is marginalized.

    Returns ``(state, log_w, logZ, seed_len)``. ``proposal="caprop"`` is the fully-adapted kernel;
    ``"bootstrap"`` proposes from the LM prior (baseline). ``band=None`` disables the band mask
    (the toy); Pythia passes an integer band. ``insert_action`` toggles the explicit INSERT move.
    """
    seed_ids = list(model.seed_ids)
    seed_len = len(seed_ids)
    obs_words = model.obs_words(observed)
    M = len(obs_words)
    obs_char = jnp.stack([jnp.asarray(model.char_ids(w)[0], jnp.int32) for w in obs_words])  # (M,Lc)
    emit_full = jax.vmap(jax.vmap(model.channel_logpdf, in_axes=(None, 0, 0)),
                         in_axes=(0, None, None))(obs_char, model.vocab_char, model.vocab_clen)
    emit_tab = _emit_table(model, obs_words, max_dist, Ke)
    WDEL = wdel if enable_indel else -jnp.inf
    WINS = wins if enable_indel else -jnp.inf
    offs = jnp.arange(-cwin, cwin + 1)
    eos_id, Vc = model.eos_id, model.emit_vocab
    LCTX = seed_len + M + slack + 1

    kernel, band_mask = _make_kernel(seed_len, M, band, WDEL, WINS)
    constraint = ChoiceMap.d({"ev": jnp.float32(0.0)})
    ks = jnp.arange(M + 1)
    a0 = band_mask(jnp.where(ks == 0, 0.0, ks * WINS), 0)                 # leading spurious words
    ctx0 = jnp.full((P, LCTX), eos_id, jnp.int32)
    if seed_len:
        ctx0 = ctx0.at[:, :seed_len].set(jnp.array(seed_ids, jnp.int32))
    state = (ctx0, jnp.full(P, seed_len, jnp.int32),
             jnp.broadcast_to(a0, (P, M + 1)), jnp.zeros(P, bool))
    log_w = jnp.zeros(P)
    logZ = 0.0

    def _assemble(ctx_len, log_alpha, lmlog):
        n_emitted = ctx_len - seed_len
        return jax.vmap(
            lambda la, lm, ne: _caprop_scores(la, lm, ne, emit_tab, emit_full, offs, J, M,
                                              band_mask, ne + 1, eos_id, Vc, WDEL, WINS,
                                              insert_action),
            in_axes=(0, 0, 0))(log_alpha, lmlog, n_emitted)

    @jax.jit
    def extend_caprop(keys, ctx_buf, ctx_len, log_alpha, done, cand, emit_cols, scores):
        def one(k, cb, cl, la, dn, c, ec, sc):
            tr, w = kernel.importance(k, constraint, ((cb, cl, la, dn), c, ec, sc))
            rv = tr.get_retval()
            return rv[0], rv[1], rv[2], rv[3], w

        cb, cl, la, dn, ws = jax.vmap(one)(keys, ctx_buf, ctx_len, log_alpha, done,
                                           cand, emit_cols, scores)
        return (cb, cl, la, dn), ws

    @jax.jit
    def extend_bootstrap(keys, ctx_buf, ctx_len, log_alpha, done, lmlog):
        def one(k, cb, cl, la, dn, lm):
            Z = logsumexp(la)
            s = jax.random.categorical(k, lm)
            chose_eos = s == eos_id
            w_id = jnp.where(chose_eos, 0, s)
            col = emit_full[:, jnp.clip(w_id, 0, Vc - 1)]
            new_alpha = band_mask(_word_row_update(la, col, WDEL, WINS), cl + 1 - seed_len)
            incr = jnp.where(chose_eos, 0.0, logsumexp(new_alpha) - Z)
            advance = (~dn) & (~chose_eos)
            incr = jnp.where(dn, 0.0, incr)
            incr = jnp.where(jnp.isnan(incr), -jnp.inf, incr)
            cb2 = jnp.where(advance, cb.at[cl].set(w_id.astype(jnp.int32)), cb)
            return (cb2, jnp.where(advance, cl + 1, cl),
                    jnp.where(advance, new_alpha, la), dn | chose_eos), incr

        states, ws = jax.vmap(one)(keys, ctx_buf, ctx_len, log_alpha, done, lmlog)
        return states, ws

    for _ in range(M + slack):
        ctx_buf, ctx_len, log_alpha, done = state
        lmlog = model.lm_fn(ctx_buf, ctx_len)                            # batched LM call (P, vocab)
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, P)
        if proposal == "caprop":
            cand, emit_cols, scores = _assemble(ctx_len, log_alpha, lmlog)
            state, incr = extend_caprop(keys, ctx_buf, ctx_len, log_alpha, done,
                                        cand, emit_cols, scores)
        else:
            state, incr = extend_bootstrap(keys, ctx_buf, ctx_len, log_alpha, done, lmlog)
        log_w = log_w + incr
        if _ess(log_w) < 0.5 * P:        # ESS-triggered resampling keeps early diversity
            logZ = logZ + logsumexp(log_w) - jnp.log(P)
            key, sub = jax.random.split(key)
            anc = jax.random.categorical(sub, log_w, shape=(P,))
            state = jax.tree_util.tree_map(lambda a: a[anc], state)
            log_w = jnp.zeros(P)

    # Terminal full-consumption correction: caprop folds alpha[M] into the EOS candidate, so EOS'd
    # particles already paid it; both proposals still need it for particles live at the budget (else
    # raw forward mass over-rewards long junk parses). bootstrap never folds it -> applies to all.
    _, _, log_alpha, done = state
    need_term = jnp.ones_like(done) if proposal == "bootstrap" else ~done
    term = jnp.where(need_term, log_alpha[:, M] - logsumexp(log_alpha, axis=1), 0.0)
    term = jnp.where(jnp.isnan(term), -jnp.inf, term)
    log_w = log_w + term
    logZ = logZ + logsumexp(log_w) - jnp.log(P)
    return state, log_w, float(logZ), seed_len


def decode(state, log_w, model, skip=0, key=jax.random.PRNGKey(0), top=3):
    """Most-probable intended sentences from the weighted particle cloud (decode by weight)."""
    ctx_buf, ctx_len, _, _ = state
    anc = jax.random.categorical(key, log_w, shape=(ctx_buf.shape[0],))
    trajs = [tuple(int(t) for t in ctx_buf[int(a)][skip:int(ctx_len[int(a)])]) for a in anc]
    counts = Counter(trajs)
    n = len(trajs)
    return [(model.decode_ids(t), c / n) for t, c in counts.most_common(top)]
