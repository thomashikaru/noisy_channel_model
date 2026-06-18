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

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import genjax
from genjax import ChoiceMap, Update, Diff

from genjax_port.genjax_model import factor
from genjax_port.poc_word_indel import _word_row_update


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
    wins: float
    band: Optional[int]      # |k - t| <= band; None = no band (matches exact enumeration)
    t_max: int = 1           # capacity: max tokens per intended word (run at 1 until Phase D)


def make_rejuv_ctx(observed, model, wdel, wins, band=None, slack=3, t_max=1):
    obs_words = model.obs_words(observed)
    M = len(obs_words)
    obs_char = jnp.stack([jnp.asarray(model.char_ids(w)[0], jnp.int32) for w in obs_words])
    emit_full = jax.vmap(jax.vmap(model.channel_logpdf, in_axes=(None, 0, 0)),
                         in_axes=(0, None, None))(obs_char, model.vocab_char, model.vocab_clen)
    ks = jnp.arange(M + 1)
    a0 = jnp.where(ks == 0, 0.0, ks * wins)
    if band is not None:                                            # match the filter's band_mask(.,0)
        a0 = jnp.where(jnp.abs(ks) <= band, a0, -jnp.inf)
    return RejuvCtx(model, emit_full, a0, M, len(model.seed_ids), M + slack, wdel, wins, band, t_max)


def _band_mask(ctx, alpha, t):
    if ctx.band is None:
        return alpha
    ks = jnp.arange(ctx.M + 1)
    return jnp.where(jnp.abs(ks - t) <= ctx.band, alpha, -jnp.inf)


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


def _unpack_single_token(ctx_buf, ctx_len, sl, Wmax, t_max):
    """Unpack the filter's flat single-token buffer into per-word slots. The forward filter is
    single-token (T_max=1: word w == token ctx_buf[:, sl+w]); Phase D will hand us per-word token
    boundaries and this becomes the general unpack. Returns (word_tok [P,Wmax,T_max], word_len
    [P,Wmax], n_words [P]). word_len is 1 for active words (w < n_words), 0 past them."""
    P = ctx_buf.shape[0]
    n_words = ctx_len - sl                                         # (P,)
    toks = ctx_buf[:, sl:sl + Wmax]                                # (P, Wmax) one token per word slot
    word_tok = jnp.zeros((P, Wmax, t_max), ctx_buf.dtype).at[:, :, 0].set(toks)
    word_len = (jnp.arange(Wmax)[None, :] < n_words[:, None]).astype(jnp.int32)
    return word_tok, word_len, n_words


def _flat_buffer(ctx, word_tok, word_len, LCTX, n_out):
    """Pack per-word spans (+ seed) into a flat LM buffer [N, LCTX]. Returns (bufs, total_tokens)."""
    N = word_tok.shape[0]
    eos = ctx.model.eos_id
    packed, total = _pack(word_tok, word_len, n_out)              # (N, n_out)
    bufs = jnp.full((N, LCTX), eos, jnp.int32)
    sl = ctx.seed_len
    if sl:
        bufs = bufs.at[:, :sl].set(jnp.array(ctx.model.seed_ids, jnp.int32))
    valid = jnp.arange(n_out)[None, :] < total[:, None]
    bufs = bufs.at[:, sl:sl + n_out].set(jnp.where(valid, packed, eos).astype(jnp.int32))
    return bufs, total


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


def _channel_carry(ctx, word_surf, word_len):
    """(N, M+1) forward carry after consuming all active words (band-masked at each step) -- the same
    quantity the filter carries as ``log_alpha``. Loops over WORD slots (token-count-agnostic):
    ``word_surf [N, Wmax]`` is each word's channel surface id (T_max=1: its single token; a multi-
    token word maps to its surface string's emission column). A slot is active iff it holds >= 1
    token. The terminal channel marginal is ``carry[:, M]``; the partial (mid-loop) forward mass is
    ``logsumexp(carry)`` -- the running likelihood of the observed-prefix-so-far under the words."""
    N = word_surf.shape[0]
    Vc = ctx.model.emit_vocab
    alpha = jnp.broadcast_to(ctx.a0, (N, ctx.M + 1))
    upd = jax.vmap(lambda a, c: _word_row_update(a, c, ctx.wdel, ctx.wins))
    for i in range(ctx.Wmax):
        surf_i = jnp.clip(word_surf[:, i], 0, Vc - 1)
        col = ctx.emit_full[:, surf_i].T                          # (N, M)
        new = _band_mask(ctx, upd(alpha, col), i + 1)
        alpha = jnp.where((word_len[:, i] > 0)[:, None], new, alpha)
    return alpha


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


def make_sweep(ctx, pool_tok, pool_len, max_tail=None):
    """Build a reusable ``sweep(key, ctx_buf, ctx_len, positions, done) -> (ctx_buf, log_alpha,
    move_logw)`` whose per-word ``step`` is jitted ONCE (closure over the constant ``ctx`` + pool), so
    interleaving it across many resample events in one run does not recompile. See ``gibbs_sweep`` for
    semantics. ``done`` is a per-word ``step`` ARG (not closed over) so the same compiled step serves
    every sweep call; ``positions`` only sets the Python loop length (more/fewer calls to one step).

    **LM scoring (R3): only the SUFFIX is scored.** The conditional ``q(x) ∝ LM(x|prefix) +
    LM(suffix|prefix,x) + channel(x)`` -- the prefix LM is identical for every candidate (the move
    only touches word ``w``), so it CANCELS in the softmax / SMCP3 weight and is never computed. We
    score the candidate-dependent tail ``[x, suffix words, EOS?]`` via ``ctx.model.tail_logprobs``
    (Pythia: a KV-cached scorer that prefills the prefix once and shares it across candidates; toy /
    default: a generic uncached chain-rule). This replaces R2's wasteful whole-sentence re-score (the
    ×151 LM-forward balloon). ``max_tail`` bounds the suffix width (default ``Wmax+1`` for a full
    sweep; the filter passes ``rejuv_lookback+1`` for a windowed sweep)."""
    sl, Wmax, T, M = ctx.seed_len, ctx.Wmax, ctx.t_max, ctx.M
    n_out = Wmax * T
    K = pool_tok.shape[1]
    Kt = 1 + K
    mt = (Wmax + 1) if max_tail is None else max_tail
    eos_id = ctx.model.eos_id
    pool_surf = pool_tok[:, :, 0]                                  # (Wmax, K) T_max=1 surface == token
    slot_model, slot_proposal = _slot_gf(Kt)
    tail_fn = ctx.model.tail_logprobs or (
        lambda cb, cl, t, tl: _tail_chain_uncached(ctx.model.lm_fn, cb, cl, t, tl))

    @jax.jit
    def step(key, w, word_tok, word_len, word_surf, move_logw, done, n_words):
        P = word_tok.shape[0]
        # candidates: [COPY] ++ pool[w], shapes (P, Kt, ...)
        cur_tok = word_tok[:, w, :]                               # (P, T)
        cur_len = word_len[:, w]                                  # (P,)
        cur_surf = word_surf[:, w]                                # (P,)
        pt, pl, ps = pool_tok[w], pool_len[w], pool_surf[w]       # (K,T),(K,),(K,)
        cand_tok = jnp.concatenate(
            [cur_tok[:, None, :], jnp.broadcast_to(pt[None], (P, K, T))], axis=1)   # (P,Kt,T)
        cand_len = jnp.concatenate(
            [cur_len[:, None], jnp.broadcast_to(pl[None], (P, K))], axis=1)         # (P,Kt)
        cand_surf = jnp.concatenate(
            [cur_surf[:, None], jnp.broadcast_to(ps[None], (P, K))], axis=1)        # (P,Kt)
        # validity: pool pads invalid; pool entries equal to COPY de-duplicated (keep COPY at 0)
        dup = ps[None, :] == cur_surf[:, None]                                      # (P,K)
        valid = jnp.concatenate([jnp.ones((P, 1), bool),
                                 (pl[None] > 0) & ~dup], axis=1)                    # (P,Kt)

        # channel: splice slot w into the word sequence, run the forward DP -> done-aware chan (P,Kt)
        wl = jnp.broadcast_to(word_len[:, None], (P, Kt, Wmax)).at[:, :, w].set(cand_len)
        ws = jnp.broadcast_to(word_surf[:, None], (P, Kt, Wmax)).at[:, :, w].set(cand_surf)
        N = P * Kt
        LCTX = sl + n_out + 1
        done_rep = jnp.repeat(done, Kt)                                             # (N,) row = p*Kt+j
        carry = _channel_carry(ctx, ws.reshape(N, Wmax), wl.reshape(N, Wmax))       # (N, M+1)
        chan = jnp.where(done_rep, carry[:, M], logsumexp(carry, axis=1)).reshape(P, Kt)

        # LM: tail = [cand, suffix words w+1.., EOS?]; prefix [0, sl+w) cancels (not scored). (T_max=1)
        j = jnp.arange(mt)
        suff = word_surf[:, jnp.clip(w + j, 0, Wmax - 1)]                           # (P, mt) word w+j
        nw_w = n_words - w                                                          # (P,) words w..end
        is_suffix = (j[None, :] >= 1) & (j[None, :] < nw_w[:, None])                # tail pos j = word w+j
        is_eos = (j[None, :] == nw_w[:, None]) & done[:, None]                      # EOS after last word
        base = jnp.where(is_suffix, suff, eos_id)                                   # (P, mt) pad/eos slots
        base = jnp.where(is_eos, eos_id, base)
        tail = jnp.broadcast_to(base[:, None, :], (P, Kt, mt)).at[:, :, 0].set(cand_surf)  # (P,Kt,mt)
        tail_len = jnp.broadcast_to(
            jnp.clip(nw_w + done.astype(jnp.int32), 0, mt)[:, None], (P, Kt))       # (P,Kt)
        ctx_bufs, _ = _flat_buffer(ctx, word_tok, word_len, LCTX, n_out)            # (P, LCTX) current
        ctx_lens = jnp.full((P,), sl, jnp.int32) + w                               # prefix end = sl+w
        chain = tail_fn(ctx_bufs, ctx_lens, tail, tail_len)                         # (P, Kt)

        target = jnp.where(valid, chain + chan, -jnp.inf)

        keys = jax.random.split(key, P)
        s_new, weight = _smcp3_move(slot_model, slot_proposal, keys, target, jnp.zeros(P, jnp.int32))
        weight = jnp.where(jnp.isnan(weight), 0.0, weight)

        gather = lambda a: jnp.take_along_axis(a, s_new[:, None], axis=1)[:, 0]
        new_tok = jnp.take_along_axis(cand_tok, s_new[:, None, None], axis=1)[:, 0, :]   # (P,T)
        new_len, new_surf = gather(cand_len), gather(cand_surf)
        active = w < n_words                                                            # (P,)
        word_tok = word_tok.at[:, w, :].set(jnp.where(active[:, None], new_tok, cur_tok))
        word_len = word_len.at[:, w].set(jnp.where(active, new_len, cur_len))
        word_surf = word_surf.at[:, w].set(jnp.where(active, new_surf, cur_surf))
        move_logw = move_logw + jnp.where(active, weight, 0.0)
        return word_tok, word_len, word_surf, move_logw

    def sweep(key, ctx_buf, ctx_len, positions=None, done=None):
        P, LCTX = ctx_buf.shape
        done = jnp.ones(P, bool) if done is None else done
        word_tok, word_len, n_words = _unpack_single_token(ctx_buf, ctx_len, sl, Wmax, T)
        word_surf = word_tok[:, :, 0]
        move_logw = jnp.zeros(P)
        for w in (range(Wmax) if positions is None else positions):
            key, sub = jax.random.split(key)
            word_tok, word_len, word_surf, move_logw = step(
                sub, jnp.int32(w), word_tok, word_len, word_surf, move_logw, done, n_words)
        bufs, _ = _flat_buffer(ctx, word_tok, word_len, LCTX, n_out)
        log_alpha = _channel_carry(ctx, word_surf, word_len)                        # (P, M+1) for filter
        return bufs, log_alpha, move_logw

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
    default, which reproduces the certified terminal scoring exactly)."""
    return make_sweep(ctx, pool_tok, pool_len)(key, ctx_buf, ctx_len, positions, done)


def pool_from_table(cand_table, t_max=1):
    """Adapt a flat (Wmax, K) candidate-token table (single-token, the toy whole-vocab Gibbs set or a
    SymSpell table) into (pool_tok [Wmax, K, T_max], pool_len [Wmax, K]). -1 entries are pads."""
    Wmax, K = cand_table.shape
    pool_tok = jnp.full((Wmax, K, t_max), -1, jnp.int32).at[:, :, 0].set(cand_table)
    pool_len = jnp.where(cand_table >= 0, 1, 0).astype(jnp.int32)
    return pool_tok, pool_len


def build_pool(observed, model, max_dist, Ke, Wmax, t_max=1):
    """Per-intended-slot candidate pool from the model's ``candidate_ids`` (COPY + SymSpell). Slot
    ``i`` uses the candidates of observed word ``min(i, M-1)`` (1:1 alignment, clipped past the
    observed length -- those slots are inactive for an M-word parse anyway). Returns
    (pool_tok [Wmax, Ke, T_max], pool_len [Wmax, Ke])."""
    obs_words = model.obs_words(observed)
    M = len(obs_words)
    rows = []
    for i in range(Wmax):
        ids = list(model.candidate_ids(obs_words[min(i, M - 1)], max_dist, Ke))[:Ke]
        rows.append(ids + [-1] * (Ke - len(ids)))
    return pool_from_table(jnp.array(rows, jnp.int32), t_max=t_max)


def decode_counts(ctx_buf, ctx_len, model, seed_len, top=50):
    """Posterior over sentences from an EQUALLY-WEIGHTED cloud (uniform count, no log_w)."""
    P = ctx_buf.shape[0]
    trajs = [tuple(int(t) for t in ctx_buf[p][seed_len:int(ctx_len[p])]) for p in range(P)]
    c = Counter(trajs)
    return {model.decode_ids(t): n / P for t, n in c.most_common(top)}
