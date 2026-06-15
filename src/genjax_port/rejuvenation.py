"""M5/R1: substitution-flip rejuvenation (incremental reanalysis) via native genjax Rejuvenate.

Rejuvenation revisits an earlier word and re-decides its intended token using the FULL context --
including words that arrived later -- so an early commitment the filtering sweep got wrong can be
corrected once disambiguating context is in the trace. It targets the same posterior; it does not
change it (it lets the chain *reach* high-posterior reanalyses and re-diversifies particles).

This module implements R1: a Metropolis--Hastings substitution-flip on a single-token word's
intended-token address, using genjax's ``Rejuvenate`` edit request (the SMCP3 move whose weight is
the MH log-acceptance ratio; see ``docs/model.tex`` Thm 2). The model is an *unrolled* per-word
noisy-channel chain (addresses ``x0..x{W-1}``, ``o0..o{W-1}``) that threads the LM buffer, so
editing ``x_k`` re-scores the suffix's LM terms through the ``Update`` -- that is how later context
votes on the flip.

Scope: single-token words (the canonical reanalysis case, e.g. ``too``->``to``). Trans-dimensional
add/delete (changing word count) is R2; surprisal-gated scheduling is R3.

Key genjax gotcha (cost real time): a ``Rejuvenate`` ``argument_mapping`` receives only the LOCAL
sub-trace at the edited address (just ``x_k``), NOT the full trace. So the position-k context is
reconstructed from the full trace OUTSIDE the mapping and closed over (``flip_request``).
"""

from collections import Counter

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import genjax
from genjax import ChoiceMap as C

from . import lm_penzai as L
from .lm_genjax import lm_token, lm_logp
from .genjax_model import obs_dist, token_candidates
from genjax._src.generative_functions.static import StaticRequest
from genjax.inference.requests import Rejuvenate

try:
    from genjax import exact_density
except ImportError:
    from genjax._src.generative_functions.distributions.distribution import exact_density


def make_chain_model(W):
    """Unrolled noisy-channel chain over ``W`` single-token words.

    Addresses: ``x{t}`` (intended token, LM-prior), ``o{t}`` (channel, candidates as args). The
    buffer is threaded so ``x{t}``'s LM score depends on ``x{<t}`` -- editing an early ``x`` rescates
    the suffix. Args: ``(buf0, ilen0, cand_xs [W,K], cand_ls [W,K])``.
    """
    @genjax.gen
    def model(buf0, ilen0, cand_xs, cand_ls):
        buf, il = buf0, ilen0
        for t in range(W):
            x = lm_token(buf, il) @ f"x{t}"
            _ = obs_dist(x, cand_xs[t], cand_ls[t]) @ f"o{t}"
            buf = buf.at[il].set(x.astype(jnp.int32))
            il = il + 1
        return None
    return model


def chain_inputs(obs_ids, m_extra=6):
    """Build ``(model, obs, buf0, ilen0, cand_xs, cand_ls)`` for single-token observed words.

    ``obs_ids`` is a sequence of observed single-token ids; ``cand_xs[t]/cand_ls[t]`` are the padded
    copy+substitution candidates for word ``t`` (``genjax_model.token_candidates``).
    """
    obs = [int(o) for o in obs_ids]
    W = len(obs)
    M = 1 + W + m_extra
    buf0 = jnp.full(M, L.EOS_ID, jnp.int32)
    ilen0 = jnp.array(1, jnp.int32)
    cxs, cls = zip(*(token_candidates(o) for o in obs))
    return make_chain_model(W), obs, buf0, ilen0, jnp.stack(cxs), jnp.stack(cls)


def literal_trace(key, model, obs, buf0, ilen0, cand_xs, cand_ls):
    """A trace fixed at the LITERAL reading: every ``x{t}`` = the observed token, ``o`` constrained."""
    W = len(obs)
    chm = C.d({**{f"x{t}": jnp.int32(obs[t]) for t in range(W)},
               **{f"o{t}": jnp.int32(obs[t]) for t in range(W)}})
    tr, _ = model.importance(key, chm, (buf0, ilen0, cand_xs, cand_ls))
    return tr


# Local-posterior proposal over a word's candidate set, reweighted by the live LM: q(x) prop
# LM(x | ctx) * channel(x). Used as both the K and L kernel of the SMCP3/Rejuvenate move.
def _prop_sample(key, buf, il, cx, cl):
    sc = lm_logp(buf, il)[cx] + cl
    return cx[jax.random.categorical(key, sc)].astype(jnp.int32)


def _prop_logpdf(x, buf, il, cx, cl):
    sc = lm_logp(buf, il)[cx] + cl
    m = cx == x
    return jnp.where(jnp.any(m), jnp.max(jnp.where(m, sc, -jnp.inf)) - logsumexp(sc), -jnp.inf)


cand_prop = exact_density(_prop_sample, _prop_logpdf, "cand_prop")


def flip_request(tr, k, buf0, ilen0, cand_xs, cand_ls):
    """Build the ``Rejuvenate`` substitution-flip request for position ``k``.

    Reconstructs the position-k context ``(buf, il)`` from the full trace's ``x{<k}`` and closes
    over it (the ``argument_mapping`` only sees the local ``x{k}`` sub-trace, so it cannot rebuild
    the context itself).
    """
    chm = tr.get_choices()
    buf, il = buf0, ilen0
    for j in range(k):
        buf = buf.at[il].set(chm[f"x{j}"].astype(jnp.int32))
        il = il + 1
    args = (buf, il, cand_xs[k], cand_ls[k])
    return StaticRequest({f"x{k}": Rejuvenate(cand_prop, lambda _chm: args)})


def rejuv_step(key, tr, k, buf0, ilen0, cand_xs, cand_ls):
    """One MH substitution-flip at position ``k``: propose, score the SMCP3 weight, accept/reject.

    Returns ``(trace, weight, accepted)``. The weight is the MH log-acceptance ratio
    ``w_upd + bwd - fwd`` (``docs/model.tex`` Thm 2); accept iff ``log u < weight``.
    """
    req = flip_request(tr, k, buf0, ilen0, cand_xs, cand_ls)
    key, k1, k2 = jax.random.split(key, 3)
    new_tr, w, _, _ = req.edit(k1, tr, genjax.Diff.no_change(tr.get_args()))
    accept = jnp.log(jax.random.uniform(k2)) < w
    tr = jax.tree_util.tree_map(lambda a, b: jnp.where(accept, a, b), new_tr, tr)
    return tr, w, accept


def rejuv_sweep(key, tr, buf0, ilen0, cand_xs, cand_ls, positions=None, n_sweeps=1):
    """Run ``n_sweeps`` MH substitution-flip sweeps over ``positions`` (default: all words)."""
    W = cand_xs.shape[0]
    positions = range(W) if positions is None else positions
    for _ in range(n_sweeps):
        for k in positions:
            key, sk = jax.random.split(key)
            tr, _, _ = rejuv_step(sk, tr, k, buf0, ilen0, cand_xs, cand_ls)
    return tr


def decoded_intended(tr, W):
    """The current intended tokens ``[x0..x{W-1}]`` of a chain trace as a list of ids."""
    return [int(tr.get_choices()[f"x{t}"]) for t in range(W)]
