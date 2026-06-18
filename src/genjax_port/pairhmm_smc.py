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

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

import genjax
from genjax import ChoiceMap

from genjax_port.genjax_model import factor
from genjax_port.poc_word_indel import _word_row_update, _ess


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
    tail_logprobs: Callable = None  # (ctx_bufs, ctx_lens, tails [B,K,w], tail_lens) -> [B,K] chain-rule
    #                                 log P(tail | ctx). Used by the rejuvenation sweep (R3) to score
    #                                 only the candidate-dependent SUFFIX (the prefix LM cancels in the
    #                                 conditional). Pythia injects the KV scorer; None => a generic
    #                                 uncached fallback built from lm_fn (pairhmm_rejuv._tail_chain_uncached).
    seed_ids: Sequence[int] = ()  # context seed (toy: (); Pythia: [EOS] + prime tokens)
    word_mask: jnp.ndarray = None  # (emit_vocab,) bool: True where the token is a lexical word; the
    #                                top-J LM candidate pool is restricted to it so the prior cannot
    #                                emit non-word tokens (\n, '#', '****') as intended words. None =
    #                                no restriction (the toy, whose vocab is already all words).


def _emit_table(model, obs_words, max_dist, Ke):
    """(M, Ke) padded candidate word ids per observed word; -1 = pad."""
    rows = []
    for w in obs_words:
        ids = list(model.candidate_ids(w, max_dist, Ke))[:Ke]
        rows.append(ids + [-1] * (Ke - len(ids)))
    return jnp.array(rows, jnp.int32)


def _caprop_scores(log_alpha, lmlog, emit_tab, emit_full, offs, J, M, band_mask,
                   t_new, eos_id, emit_vocab, WDEL, WINS, word_mask):
    """Channel-aware (fully-adapted) candidate scores: [word(Kw), EOS(1)].

    Candidate set C = emission candidates in a window around the alignment frontier (channel-
    compatible) + top-J LM words (fluency/deletion bridges), deduped. Each word's score is
    ``lm logprob + dZ`` where ``dZ`` is the forward-mass increment of one ``_word_row_update`` row;
    the incremental importance weight is ``logsumexp(scores)`` (independent of the draw). EOS reads
    the terminal full-consumption mass ``alpha[M]``.

    There is no explicit INSERT action. A spurious observed word is a *channel* event, not an LM
    one, and is marginalized exactly inside ``_word_row_update``'s insertion sweep (cost ``WINS``);
    the band gives it reach (consumption may run up to ``band`` ahead of emission). Making INSERT a
    peer action in this LM-normalized step instead injects a channel event into the LM's action
    distribution -- empirically a +0.2-nat logZ bias against exact enumeration (it adds mass with no
    LM factor). Every step here is therefore a clean LM word/EOS choice; the sweep handles the rest.

    ``word_mask`` (when given) restricts the top-J LM pool to lexical word tokens, so the prior
    cannot propose non-word tokens (newlines, '#'/'****', tabs) as intended/missing words -- the
    document-start boilerplate the LM otherwise emits as spurious sentence-initial tokens. Emission
    candidates are unaffected (they are tied to observed words, so a copied punctuation token is
    fine); only the fluency/deletion bridge pool is constrained.
    """
    Z = logsumexp(log_alpha)
    fpos = jnp.clip(jnp.argmax(log_alpha), 0, M - 1)                       # alignment frontier
    emit_ids = emit_tab[jnp.clip(fpos + offs, 0, M - 1)].reshape(-1)       # window of emission cands
    lm_word = jnp.where(jnp.arange(lmlog.shape[0]) < emit_vocab, lmlog, -jnp.inf)
    lm_word = lm_word.at[eos_id].set(-jnp.inf)                             # EOS handled separately
    if word_mask is not None:                                             # restrict bridges to words
        lm_word = lm_word.at[:emit_vocab].set(
            jnp.where(word_mask, lm_word[:emit_vocab], -jnp.inf))
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

    score_eos = lmlog[eos_id] + (log_alpha[M] - Z)
    scores = jnp.concatenate([score_word, score_eos[None]])
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
        chose_eos = action == kw
        ci = jnp.clip(action, 0, kw - 1)
        w_id = jnp.where(chose_eos, jnp.int32(0), cand[ci])
        col = emit_cols[:, ci]

        advance = (~done) & (~chose_eos)
        t_after = ctx_len + 1 - seed_len
        new_alpha = band_mask(_word_row_update(log_alpha, col, WDEL, WINS), t_after)
        ctx_buf2 = jnp.where(advance, ctx_buf.at[ctx_len].set(w_id.astype(jnp.int32)), ctx_buf)
        return (ctx_buf2,
                jnp.where(advance, ctx_len + 1, ctx_len),
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
    ctx_buf, ctx_len, log_alpha, done = state
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


def run(observed, key, model, P=4000, wdel=jnp.log(0.1), wins=jnp.log(0.05), slack=3, band=None,
        max_dist=2, Ke=6, J=4, cwin=1, proposal="caprop", enable_indel=True,
        rejuv="off", rejuv_pool=None, rejuv_lookback=3, rejuv_stats=None, trace=None):
    """Sequential RB-SMC over intended words; the word alignment ``alpha`` is marginalized.

    Returns ``(state, log_w, logZ, seed_len)``. ``proposal="caprop"`` is the fully-adapted kernel;
    ``"bootstrap"`` proposes from the LM prior (baseline). ``band=None`` disables the band mask
    (the toy); Pythia passes an integer band, which also gives insertions/deletions their reach
    (consumption ``k`` may run up to ``band`` ahead of / behind the emission count).

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
    additive and only runs inside the ``rejuv == "gibbs"`` branch.
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

    # Flag-gated rejuvenation (R2): build the sweep ONCE (jitted step reused across resample events).
    rj_sweep = None
    if rejuv == "gibbs":
        from genjax_port import pairhmm_rejuv as RJ
        rj_ctx = RJ.RejuvCtx(model, emit_full, a0, M, seed_len, M + slack, WDEL, WINS, band, 1)
        rj_sweep = RJ.make_sweep(rj_ctx, *rejuv_pool, max_tail=rejuv_lookback + 1)
        if rejuv_stats is not None:
            rejuv_stats.update(P=P, Kt=rejuv_pool[0].shape[1] + 1, max_tail=rejuv_lookback + 1,
                               filter_lm_calls=0, sweep_prefills=0, sweep_tail_steps=0, uniq_frac=[])

    word_mask = model.word_mask
    def _assemble(ctx_len, log_alpha, lmlog):
        n_emitted = ctx_len - seed_len
        return jax.vmap(
            lambda la, lm, ne: _caprop_scores(la, lm, emit_tab, emit_full, offs, J, M,
                                              band_mask, ne + 1, eos_id, Vc, WDEL, WINS, word_mask),
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

    for s in range(M + slack):
        ctx_buf, ctx_len, log_alpha, done = state
        lmlog = model.lm_fn(ctx_buf, ctx_len)                            # batched LM call (P, vocab)
        if rejuv_stats is not None:
            rejuv_stats["filter_lm_calls"] += 1
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, P)
        if proposal == "caprop":
            cand, emit_cols, scores = _assemble(ctx_len, log_alpha, lmlog)
            state, incr = extend_caprop(keys, ctx_buf, ctx_len, log_alpha, done,
                                        cand, emit_cols, scores)
        else:
            state, incr = extend_bootstrap(keys, ctx_buf, ctx_len, log_alpha, done, lmlog)
        log_w = log_w + incr
        ess_pre = float(_ess(log_w))     # the ESS that triggers/avoids resampling (recorded for the viz)
        resampled, rejuv_info = False, None
        if ess_pre < 0.5 * P:            # ESS-triggered resampling keeps early diversity
            resampled = True
            logZ = logZ + logsumexp(log_w) - jnp.log(P)
            key, sub = jax.random.split(key)
            anc = jax.random.categorical(sub, log_w, shape=(P,))
            state = jax.tree_util.tree_map(lambda a: a[anc], state)
            log_w = jnp.zeros(P)
            if rj_sweep is not None:     # post-resample windowed Gibbs/SMCP3 sweep; fold its weight
                ctx_buf, ctx_len, _, done = state            # pre-next-resample (REJUV_GOAL3 (b))
                hi = min(s + 1, M + slack)                   # frontier word count (upper bound)
                lo = max(0, hi - rejuv_lookback)
                key, sub = jax.random.split(key)
                cb, la, mlw = rj_sweep(sub, ctx_buf, ctx_len, positions=range(lo, hi), done=done)
                state = (cb, ctx_len, la, done)
                log_w = log_w + mlw
                rejuv_info = {"words": [int(lo), int(hi)], "ess_after": float(_ess(log_w)),
                              "mean_abs_w": float(jnp.mean(jnp.abs(mlw)))}
                if rejuv_stats is not None:                   # KV scorer: 1 shared prefill/word/particle
                    window = hi - lo                          # (full forward) + cheap tail steps
                    rejuv_stats["sweep_prefills"] += window
                    rejuv_stats["sweep_tail_steps"] += window * rejuv_stats["max_tail"]
                    rejuv_stats["uniq_frac"].append(_uniq_frac(cb, ctx_len, seed_len))
        if trace is not None:            # per-step snapshot of the cloud (post-extend/resample/rejuv)
            trace.append(_record_step(model, state, log_w, seed_len, s, ess_pre, resampled, logZ,
                                      rejuv_info))

    # Terminal full-consumption correction: caprop folds alpha[M] into the EOS candidate, so EOS'd
    # particles already paid it; both proposals still need it for particles live at the budget (else
    # raw forward mass over-rewards long junk parses). bootstrap never folds it -> applies to all.
    _, _, log_alpha, done = state
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
    """Most-probable intended sentences from the weighted particle cloud (decode by weight)."""
    ctx_buf, ctx_len, _, _ = state
    anc = jax.random.categorical(key, log_w, shape=(ctx_buf.shape[0],))
    trajs = [tuple(int(t) for t in ctx_buf[int(a)][skip:int(ctx_len[int(a)])]) for a in anc]
    counts = Counter(trajs)
    n = len(trajs)
    return [(model.decode_ids(t), c / n) for t, c in counts.most_common(top)]
