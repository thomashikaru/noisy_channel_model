"""Shared model + inference hyperparameters for the noisy-channel port.

These constants define the noise model's action/deletion priors and the inference
proposal's exploration knobs. They live here -- rather than in any one filter -- because
both the production word-scan SMC (:mod:`smc_substitution`, :mod:`rejuv_bridge`) and the
hand-rolled reference filter (:mod:`particle_filter_unified`) share them; previously they were
homed in the (now-deleted) ``particle_filter`` prototype and imported *out* of the reference path
into production, which inverted the dependency. Keep this module dependency-light (no LM / genjax).
"""

import jax.numpy as jnp

# Dirichlet concentration for the action prior (copy, sub, insert); copy favored,
# matching --normal_alpha=3 / --error_alpha=1 in the original Gen.jl config.
ACTION_ALPHAS = jnp.array([3.0, 1.0, 1.0])

# Deletion gap: before each emission a particle may posit up to MAX_DELETIONS intended
# tokens that were dropped (produce no observation), reconstructing an omitted word. They
# are proposed lookahead-guided toward the next observed token (or, in the token-level
# filter, from the LM prior), so they contribute nothing to the importance weight directly
# -- they only reshape the LM context, and the next emission's predictive likelihood (via
# resampling) selects the helpful ones.
MAX_DELETIONS = 1  # max consecutive deletions per gap; D=1 (was 2) ~2x faster -- halves the
                   # P*K lookahead forwards/step AND shrinks the intended buffer (2T+2 vs 3T+2).
                   # Two-in-a-row deletions are rare, so D=1 is the chosen cost/coverage trade.

# A deletion must COST something, or the filter hallucinates fluent dropped words ("who",
# "a") for free. We decouple the model's a-priori deletion rate from the proposal's
# exploration rate: deletions are proposed optimistically (P_DELETE_PROPOSAL) but each
# carries an importance-weight penalty log(p_model / q) toward the smaller model prior
# (P_DELETE_PRIOR). A spuriously posited deletion must earn back that penalty in downstream
# LM fluency to survive resampling, while a genuine omission (which sharply improves the
# following predictive likelihood) does. (A data-aware deletion proposal / rejuvenation is
# the more powerful fix.)
P_DELETE_PRIOR = 0.02
P_DELETE_PROPOSAL = 0.20

# Candidate omitted tokens per deletion slot for the lookahead deletion proposal. Swept
# 6-vs-12 at D=1: identical posteriors, ~1.6x faster at K=6, so 6 is the default. Cost is
# linear in K (it drives the P*K lookahead forward).
LOOKAHEAD_K = 6
