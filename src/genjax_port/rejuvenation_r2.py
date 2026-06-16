"""M5/R2: add/delete reversible-jump rejuvenation (trans-dimensional reanalysis).

REFERENCE / oracle -- NOT on the production path. STRATEGIC PIVOT (2026-06-16): the forward filter
already does add/delete (M2 deletion gap + M3 insertion), so add/delete *rejuvenation* adds little,
and routing the move through the ``@gen`` trace cost W LM forwards per edit. Production rejuvenation
is the MANUAL, flat-buffer, single-forward substitution move ``rejuv_bridge.manual_subflip_move`` /
``run_smc_conditional_rejuv_aligned``; this ``@gen`` gap chain is kept as the correctness oracle
(``MaskCombinator.edit`` gives the reversible-jump weight for free) that validates that math, and as
the carrier should trans-dimensional rejuvenation ever be revived. See ``planning/R2_PLAN.md`` and the
STRATEGIC PIVOT note in memory ``genjax-native-migration``.

R1 (``rejuvenation.py``) flips a word's intended token at fixed word count. R2 changes the *length*
of the intended sentence: ``add`` inserts an omitted intended word (reversing a deletion), ``delete``
removes a posited one. This lets rejuvenation revise the *alignment* once later context arrives -- an
omission the filtering sweep missed can be recovered. Ports Gen.jl's ``rejuv_proposal_add_delete`` +
``involution_add_delete`` (``src/gen_inference.jl``), proven correct by ``docs/model.tex`` §sec:moves
(R2) and the keystone spike (``/tmp/genjax_spike6_maskflip.py``). See ``planning/R2_PLAN.md``.

Representation (the crux): a **masked deletion-gap chain** that keeps the address set FIXED, so the
trans-dimensional move is a fixed-address ``Update`` (and vmaps later). Per observed word ``t``:

  ``del{t}``    ~ flip(p_del)              -- omitted intended word before observed word ``t``?
  ``gap{t}/xd`` ~ mask(lm_token)           -- the omitted token (scored only when ``del{t}`` true)
  ``x{t}``      ~ lm_token                 -- intended token aligned to observed word ``t``
  ``o{t}``      ~ obs_dist(x, cand)        -- channel (observed token, constrained)

The LM buffer is threaded from the choices, so flipping any ``del{t}`` re-scores the whole suffix
through the ``Update`` (later context votes). genjax's ``MaskCombinator.edit`` supplies the
reversible-jump birth/death weight for free: False->True gives the born token's full score,
True->False gives minus the old score (Jacobian unity). Spike 6 verified the ``Update`` weight equals
``log p(t')/p(t)`` exactly, including the Bernoulli del-prior + LM(omitted) + suffix re-score.

Unlike R1 we do NOT use genjax ``Rejuvenate``: the forward (add, draws a token) and backward (delete,
draws none) proposals are asymmetric, so we assemble the SMCP3 weight ``W = w_upd + s_bwd - s_fwd``
(docs/model.tex eq smcp3) by hand, using ``model.edit(Update(...))`` for the free ``w_upd``. The
binary slot forces the move type per state (add from off, delete from on); the two are mutual
reverses, so detailed balance holds (Thm 2). Scope: one omitted token per gap (``MAX_DELETIONS=1``),
single-token words -- same as R1.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import genjax
from genjax import ChoiceMap as C

from . import lm_penzai as L
from .lm_genjax import lm_token, lm_logp
from .genjax_model import obs_dist, token_candidates
from .rejuvenation import cand_prop
from .config import P_DELETE_PRIOR, LOOKAHEAD_K
from genjax._src.generative_functions.static import StaticRequest
from genjax.inference.requests import Rejuvenate


# --- the masked deletion-gap inner model (one possibly-omitted intended token) ---------------

@genjax.gen
def _del_inner(buf, il):
    xd = lm_token(buf, il) @ "xd"
    return xd

masked_del = genjax.mask(_del_inner)


def make_gap_chain(W, p_del=P_DELETE_PRIOR):
    """Unrolled masked deletion-gap chain over ``W`` single-token observed words (see module doc).

    Args: ``(buf0, ilen0, cand_xs [W,K], cand_ls [W,K])``. Flipping ``del{t}`` shifts the threaded
    buffer for the suffix, so the suffix LM terms re-score automatically under ``Update``.
    """
    @genjax.gen
    def chain(buf0, ilen0, cand_xs, cand_ls):
        buf, il = buf0, ilen0
        for t in range(W):
            d = genjax.flip(p_del) @ f"del{t}"
            m = masked_del(d, buf, il) @ f"gap{t}"
            di = d.astype(jnp.int32)
            buf = jnp.where(d, buf.at[il].set(m.value.astype(jnp.int32)), buf)
            il = il + di
            x = lm_token(buf, il) @ f"x{t}"
            _ = obs_dist(x, cand_xs[t], cand_ls[t]) @ f"o{t}"
            buf = buf.at[il].set(x.astype(jnp.int32))
            il = il + 1
        return None
    return chain


def gap_chain_inputs(obs_ids, m_extra=8, p_del=P_DELETE_PRIOR):
    """Build ``(model, obs, buf0, ilen0, cand_xs, cand_ls)`` for single-token observed words.

    ``m_extra`` must cover the worst case of one omitted token per word (buffer width ``1 + 2W``).
    """
    obs = [int(o) for o in obs_ids]
    W = len(obs)
    M = 1 + 2 * W + m_extra
    buf0 = jnp.full(M, L.EOS_ID, jnp.int32)
    ilen0 = jnp.array(1, jnp.int32)
    cxs, cls = zip(*(token_candidates(o) for o in obs))
    return make_gap_chain(W, p_del), obs, buf0, ilen0, jnp.stack(cxs), jnp.stack(cls)


def literal_trace(key, model, obs, buf0, ilen0, cand_xs, cand_ls):
    """Trace at the LITERAL no-deletion reading: every ``del{t}`` false, ``x{t}`` = observed token."""
    W = len(obs)
    chm = C.d({**{f"del{t}": jnp.bool_(False) for t in range(W)},
               **{f"x{t}": jnp.int32(obs[t]) for t in range(W)},
               **{f"o{t}": jnp.int32(obs[t]) for t in range(W)}})
    tr, _ = model.importance(key, chm, (buf0, ilen0, cand_xs, cand_ls))
    return tr


# --- the omitted-token proposal q (LM top-k reweighted by one-step lookahead) -----------------
# Mirrors smc_substitution.deletion_gap's proposal so the move targets the sweep's posterior. The
# support is the top-``lookahead_k`` LM tokens at the gap context; q reweights them by how well each
# makes the next observed token likely. Evaluated at the SAME (buf, il) for both move directions, so
# its support is identical forward and backward -> detailed balance is exact on the reachable set.

def _q_logits(buf, il, obs_next, lookahead_k):
    """Return ``(cand_ids [K], q_logits [K], logZ)`` for the gap proposal at context ``(buf, il)``."""
    lp = lm_logp(buf, il)                                   # [V]
    cand_lm, cand_ids = jax.lax.top_k(lp, lookahead_k)      # [K]
    bufs = jnp.broadcast_to(buf, (lookahead_k,) + buf.shape)
    bufs = bufs.at[jnp.arange(lookahead_k), il].set(cand_ids)
    look = jax.nn.log_softmax(
        L.next_token_logits(bufs, jnp.full(lookahead_k, il + 1)), axis=-1)[:, obs_next]  # [K]
    q_logits = cand_lm + look
    return cand_ids, q_logits, logsumexp(q_logits)


def _q_sample(key, buf, il, obs_next, lookahead_k):
    cand_ids, q_logits, _ = _q_logits(buf, il, obs_next, lookahead_k)
    return cand_ids[jax.random.categorical(key, q_logits)].astype(jnp.int32)


def _q_logpdf(x, buf, il, obs_next, lookahead_k):
    cand_ids, q_logits, logZ = _q_logits(buf, il, obs_next, lookahead_k)
    m = cand_ids == x
    return jnp.where(jnp.any(m), jnp.max(jnp.where(m, q_logits, -jnp.inf)) - logZ, -jnp.inf)


# --- the add/delete move ----------------------------------------------------------------------

def _gap_tok(chm, t):
    """The (possibly masked) omitted token at gap ``t`` as an int32 -- unwraps the ``Mask``."""
    return chm[f"gap{t}", "xd"].value.astype(jnp.int32)


def _context_at_gap(tr, k, buf0, ilen0):
    """Reconstruct ``(buf, il)`` just before gap ``k`` from the trace (mirrors the model threading)."""
    chm = tr.get_choices()
    buf, il = buf0, ilen0
    for j in range(k):
        dj = chm[f"del{j}"]
        buf = jnp.where(dj, buf.at[il].set(_gap_tok(chm, j)), buf)
        il = il + dj.astype(jnp.int32)
        buf = buf.at[il].set(chm[f"x{j}"].astype(jnp.int32))
        il = il + 1
    return buf, il


def add_delete_step(key, tr, k, buf0, ilen0, cand_xs, cand_ls, obs, lookahead_k=LOOKAHEAD_K):
    """One reversible-jump add/delete MH move at gap ``k``: toggle ``del{k}``, accept/reject.

    From ``del{k}`` false -> ADD (propose an omitted token from ``q``); from true -> DELETE. Returns
    ``(trace, W, accepted)`` with ``W = w_upd + s_bwd - s_fwd`` (docs/model.tex eq smcp3); accept iff
    ``log u < W``. The two directions are mutual reverses (Thm 2 / detailed balance).
    """
    model, args = tr.get_gen_fn(), tr.get_args()
    chm = tr.get_choices()
    cur_del = chm[f"del{k}"]                       # bool: gap currently on?
    xd_cur = _gap_tok(chm, k)
    buf_k, il_k = _context_at_gap(tr, k, buf0, ilen0)
    obs_next = jnp.int32(obs[k])

    key, kp, ke, ku = jax.random.split(key, 4)
    xprop = _q_sample(kp, buf_k, il_k, obs_next, lookahead_k)
    adding = jnp.logical_not(cur_del)
    # ADD writes the proposed token; DELETE leaves the (masked) token untouched.
    gap_tok = jnp.where(adding, xprop, xd_cur)
    upd = C.d({f"del{k}": jnp.logical_not(cur_del), f"gap{k}": C.d({"xd": gap_tok})})
    new_tr, w_upd, _, _ = model.edit(ke, tr, genjax.Update(upd), genjax.Diff.no_change(args))

    # forward proposal score (only ADD draws a token); backward (reverse) is the opposite direction.
    s_fwd = jnp.where(adding, _q_logpdf(xprop, buf_k, il_k, obs_next, lookahead_k), 0.0)
    s_bwd = jnp.where(adding, 0.0, _q_logpdf(xd_cur, buf_k, il_k, obs_next, lookahead_k))
    W = w_upd + s_bwd - s_fwd

    accept = jnp.log(jax.random.uniform(ku)) < W
    tr = jax.tree_util.tree_map(lambda a, b: jnp.where(accept, a, b), new_tr, tr)
    return tr, W, accept


# --- substitution-flip on the gap chain (R1's move, but aware of the deletion gaps) -----------
# The gap chain's x{k} (the intended token for observed word k) is the same lm_token + obs_dist
# structure R1 flips on its plain chain; the only difference is the context, which must include any
# omitted (gap) tokens before x{k}. We reuse R1's candidate proposal (cand_prop) so a combined
# post-sweep can revise both substitutions (here) and add/deletes on one trace.

def _context_before_word(tr, k, buf0, ilen0):
    """``(buf, il)`` just before ``x{k}`` -- the gap-``k`` context advanced by gap ``k`` if it is on."""
    buf, il = _context_at_gap(tr, k, buf0, ilen0)
    chm = tr.get_choices()
    dk = chm[f"del{k}"]
    buf = jnp.where(dk, buf.at[il].set(_gap_tok(chm, k)), buf)
    il = il + dk.astype(jnp.int32)
    return buf, il


def sub_flip_step(key, tr, k, buf0, ilen0, cand_xs, cand_ls, obs):
    """One MH substitution-flip on ``x{k}`` of the gap chain (R1 reanalysis, gap-aware context).

    Returns ``(trace, weight, accepted)``. Editing ``x{k}`` re-scores the suffix via the ``Update``,
    so later context (and any later gaps) vote on the flip -- exactly R1, on the gap-chain trace."""
    buf, il = _context_before_word(tr, k, buf0, ilen0)
    args = (buf, il, cand_xs[k], cand_ls[k])
    req = StaticRequest({f"x{k}": Rejuvenate(cand_prop, lambda _chm: args)})
    k1, k2 = jax.random.split(key)
    new_tr, w, _, _ = req.edit(k1, tr, genjax.Diff.no_change(tr.get_args()))
    accept = jnp.log(jax.random.uniform(k2)) < w
    tr = jax.tree_util.tree_map(lambda a, b: jnp.where(accept, a, b), new_tr, tr)
    return tr, w, accept


def add_delete_sweep(key, tr, buf0, ilen0, cand_xs, cand_ls, obs,
                     positions=None, n_sweeps=1, lookahead_k=LOOKAHEAD_K):
    """Run ``n_sweeps`` add/delete sweeps over gap ``positions`` (default: all words)."""
    W = cand_xs.shape[0]
    positions = range(W) if positions is None else positions
    for _ in range(n_sweeps):
        for k in positions:
            key, sk = jax.random.split(key)
            tr, _, _ = add_delete_step(sk, tr, k, buf0, ilen0, cand_xs, cand_ls, obs, lookahead_k)
    return tr


# --- vectorized over particles (Phase 2: the move batched across the SMC particle set) ---------
# The whole point of the genjax port: vmap the move so the LM forward batches (P=64 ~ small x P=1,
# not 64x; spike 7). The masked-gap chain is fixed-address, so the batched trace stays rectangular
# and -- critically -- the MaskCombinator vmaps over PER-PARTICLE ``del{t}`` flags (each particle
# carries its own deletion config). Parity with a per-particle loop holds to XLA float-tiling noise
# (~1e-2, as for bucketed forwards). This is the canonical R2 move for the filtering-sweep bridge.

def vmapped_add_delete(keys, trs, k, buf0, ilen0, cand_xs, cand_ls, obs, lookahead_k=LOOKAHEAD_K):
    """``jax.vmap`` of :func:`add_delete_step` over a P-batched gap-chain trace ``trs``.

    ``keys`` is ``[P]`` PRNG keys; ``trs`` is the per-particle batched trace (leading P axis), with
    per-particle deletion configs allowed. ``buf0/ilen0/cand_xs/cand_ls/obs`` are shared. Returns the
    batched ``(trace, W [P], accepted [P])`` -- one batched LM forward, not P of them.
    """
    step = lambda key, tr: add_delete_step(
        key, tr, k, buf0, ilen0, cand_xs, cand_ls, obs, lookahead_k)
    return jax.vmap(step)(keys, trs)


# --- readout ----------------------------------------------------------------------------------

def gap_config(tr, W):
    """The gap on/off + token per word: tuple of ``(bool, token_or_None)`` of length ``W``."""
    chm = tr.get_choices()
    return tuple((bool(chm[f"del{t}"]),
                  int(_gap_tok(chm, t)) if bool(chm[f"del{t}"]) else None) for t in range(W))


def decoded_gap_chain(tr, W):
    """Intended token ids in order, including inserted (omitted) gap tokens."""
    chm = tr.get_choices()
    out = []
    for t in range(W):
        if bool(chm[f"del{t}"]):
            out.append(int(_gap_tok(chm, t)))
        out.append(int(chm[f"x{t}"]))
    return out
