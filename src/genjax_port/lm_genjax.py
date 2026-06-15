"""The penzai LM wrapped as a genjax custom distribution (the genjax-native port's bedrock).

This is the bridge proven by the de-risking spikes (see ``MIGRATION_PLAN.md`` §2/§4): the
in-graph Pythia LM is exposed to genjax as an ``exact_density`` categorical distribution over
the next BPE token. With it, the autoregressive intended-sentence generator becomes a plain
``@gen`` ``Scan`` model whose trace auto-records every token choice -- which is exactly the
addressable structure native ``Rejuvenate`` (incremental reanalysis) needs and the hand-rolled
``particle_filter_unified.py`` had to fake.

Design rules carried over from the spikes (each one cost time to learn -- keep them):

- **Single-sequence logpdf/sampler.** ``lm_logp``/``_lm_sample`` operate on one buffer ``[M]``
  and a scalar ``i_len``; you let ``jax.vmap`` add the particle axis. Do NOT pre-batch a
  particle dimension inside -- that defeats genjax's vmap batching of the penzai forward
  (P=64 ran ~6x P=1, not 64x, precisely because the forward stays one batched call).
- **int32 tokens.** beartype in genjax is strict; the sampler returns ``int32``.
- **Name the distribution.** ``exact_density`` warns if ``name`` is omitted.

Buffer convention is the same as :mod:`lm_penzai`: position 0 is the ``EOS_ID`` BOS seed, a
buffer with ``i_len`` filled positions reads its next-token logits at position ``i_len - 1``,
and padded positions hold ``EOS_ID`` (causal attention ignores them).
"""

import jax
import jax.numpy as jnp

from . import lm_penzai as L

try:
    from genjax import exact_density
except ImportError:  # older layout -- the spikes hit this fallback
    from genjax._src.generative_functions.distributions.distribution import exact_density


def lm_logits_single(buf, i_len):
    """Next-token logits for one buffer: ``buf [M]``, scalar ``i_len`` -> ``[V]``.

    Wraps :func:`lm_penzai.next_token_logits` (which is batched) by adding and dropping a
    singleton batch axis, so the call site stays single-sequence and ``vmap`` supplies the
    particle axis.
    """
    return L.next_token_logits(buf[None, :], jnp.asarray([i_len]))[0]


def lm_logp(buf, i_len):
    """Normalized next-token log-probabilities for one buffer: ``[V]``."""
    return jax.nn.log_softmax(lm_logits_single(buf, i_len))


def _lm_sample(key, buf, i_len):
    return jax.random.categorical(key, lm_logits_single(buf, i_len)).astype(jnp.int32)


def _lm_logpdf(tok, buf, i_len):
    return lm_logp(buf, i_len)[tok]


# The penzai LM as a genjax categorical distribution over the next token.
# Use inside @gen as:  tok = lm_token(buf, i_len) @ "tok"
lm_token = exact_density(_lm_sample, _lm_logpdf, "lm_token")
