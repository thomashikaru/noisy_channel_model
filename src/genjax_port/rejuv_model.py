"""Phase 2b: the masked chain model -- the fixed-shape carrier for trans-dimensional rejuvenation.

R2 (add/delete, multi-token sub) changes the number of intended tokens per particle, which a real
dimension change can't `vmap`. Phase 2a proved the lever: emulate it with a FIXED-shape MASKED trace
-- each slot carries a `present` flag (a `flip` choice) gating a `mask(kernel)` sub-call, and toggling
`present` is a dimension-preserving edit whose `MaskCombinator.edit` weight is the correct add/delete
weight (detailed-balance verified).

This module builds the **masked autoregressive chain**: `K` single-token slots, each
`present_k ~ flip(p)` gating `lm_token(buf, il)`. The novel piece beyond 2a is the autoregression --
the LM buffer must thread over **active slots only**, so an inactive slot neither scores nor shifts
later context. We do that by threading `(buf, il)` manually with `jnp.where(present, advanced,
unchanged)` (the `Mask` retval exposes the inner `.value` regardless of flag), while the `Mask`
combinator zeroes the inactive slot's LM score. So the active intended tokens form a contiguous prefix
in `buf` and each active slot's `lm_token` sees the correct context.

Single-token slots = the add/delete representation (a present slot is an intended word; flipping its
flag is delete/add). Multi-token word slots (COPY *n* / SUB 1 via `genjax_model.make_word_model`) and
the gap+word grid layer on top in the rest of Phase 2b/2c.
"""

import jax
import jax.numpy as jnp
import genjax
from genjax import ChoiceMap as C

from . import lm_penzai as L
from .lm_genjax import lm_token, lm_logp

P_PRESENT_DEFAULT = 0.5


@genjax.gen
def _token_slot(buf, il):
    """One intended-token slot: sample x ~ lm_token(buf, il), append it. Returns (x, buf', il')."""
    x = lm_token(buf, il) @ "x"
    xi = x.astype(jnp.int32)
    return xi, buf.at[il].set(xi), il + 1


def make_masked_chain_model(K, p_present=P_PRESENT_DEFAULT):
    """A masked autoregressive chain over ``K`` single-token slots.

    Per slot: ``present_k ~ flip(p_present) @ f"p{k}"`` gates ``_token_slot.mask()`` at ``f"s{k}"``.
    The carry ``(buf, il)`` advances iff ``present_k`` (so active tokens are a contiguous prefix and
    inactive slots cost nothing). Args ``(buf0, ilen0)``; returns the final ``(buf, il)``.
    """
    masked_slot = _token_slot.mask()

    @genjax.gen
    def model(buf0, ilen0):
        buf, il = buf0, ilen0
        for k in range(K):
            present = genjax.flip(p_present) @ f"p{k}"
            mret = masked_slot(present, buf, il) @ f"s{k}"
            _, cand_buf, cand_il = mret.value          # inner result regardless of mask flag
            buf = jnp.where(present, cand_buf, buf)
            il = jnp.where(present, cand_il, il)
        return buf, il

    return model


def chain_constraints(present, tokens):
    """Choicemap fixing every slot's ``present`` flag and (active slots') token.

    ``present``: ``[K]`` bool; ``tokens``: ``[K]`` int (the token for each present slot; entries for
    absent slots are inert -- the masked region ignores them).
    """
    K = len(present)
    d = {}
    for k in range(K):
        d[f"p{k}"] = jnp.bool_(present[k])
        d[f"s{k}"] = C.d({"x": jnp.int32(tokens[k])})
    return C.d(d)


def active_tokens(present, tokens):
    """The intended token ids of the active slots, in order (what the chain actually emits)."""
    return [int(t) for p, t in zip(present, tokens) if p]
