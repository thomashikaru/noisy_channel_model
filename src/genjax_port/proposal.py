"""Proposal and importance-weight mechanics for the token-level particle filter.

The proposal for the per-step latent ``(action, intended_token)`` is the **local posterior**
over the enumerated explanations assembled by :func:`model.step_log_evidence` -- i.e. we
propose each option in proportion to its joint log-density ``log_ev``. With that (locally
optimal) proposal, the incremental importance weight collapses to the one-step predictive
log-likelihood of the observed token:

    log w_t = log p - log q = log_ev[chosen] - (log_ev[chosen] - logsumexp(log_ev))
            = logsumexp_k log_ev[:, k]

which is the same regardless of which option is sampled. The sampled option only determines
how each particle's intended context advances (mirroring the original's proposal + model split).
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def propose(key, log_ev):
    """Sample one option per particle from the local posterior; return weights too.

    Args:
        key: PRNG key.
        log_ev: ``[P, K+1]`` per-option joint log-densities from ``step_log_evidence``.

    Returns:
        option: ``[P]`` int index of the sampled option per particle
            (``< K`` => emitting candidate ``k``; ``== K`` => insert).
        log_w: ``[P]`` incremental importance log-weight = ``logsumexp_k log_ev``.
    """
    option = jax.random.categorical(key, log_ev, axis=1)  # sample from softmax(log_ev)
    log_w = logsumexp(log_ev, axis=1)
    return option, log_w
