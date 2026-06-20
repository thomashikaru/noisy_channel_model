"""Core genjax combinator: ``factor`` — a deterministic log-weight injector.

Extracted from the (now-archived) M-series ``genjax_model`` so the live pair-HMM RB-SMC path does not
transitively pull in the dead word-scan model. ``factor`` observes the value ``0.0`` with
``logpdf = the supplied weight``, which folds a deterministic log-weight into a genjax trace. Used live
by the unified RB-SMC filter (``pairhmm_smc``) and its rejuvenation sweep (``pairhmm_rejuv``) to inject
the marginalized channel evidence into the importance weight via ``factor(logsumexp_C) @ "ev"``.
"""

import jax.numpy as jnp

try:
    from genjax import exact_density
except ImportError:
    from genjax._src.generative_functions.distributions.distribution import exact_density


# Inject a deterministic log-weight into a trace: observe value 0.0 with logpdf = the weight.
factor = exact_density(lambda key, lw: jnp.float32(0.0), lambda v, lw: lw, "factor")
