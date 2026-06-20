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
# NOTE: this 3-way prior is read only by the TOKEN-level reference filter (particle_filter_unified,
# smc_substitution) and must stay 3-way for those call-sites. The pair-HMM word-action channel
# (planning/WORD_ACTION_CHANNEL_PLAN.md) uses the 4-way extension (copy, sub, insert, DELETE) settled by
# calibration_word_action_prior_search.py and homed as ``pythia_word_caprop.ACTION_ALPHA_DEFAULT =
# (3,1,1,1)`` (next to where it is consumed) -- NOT here, to avoid breaking the 3-way consumers.
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
# In the pair-HMM filter this is the word-level deletion cost WDEL = log(P_DELETE_PRIOR): a posited
# MISSING intended word (no observation) pays it. Lowered 0.02 -> 0.005 (-3.91 -> -5.30 nats) so a
# hallucinated fluent word must earn back >5.3 nats of predictive payoff to survive -- this stops
# inference "cheating" by filling LM-cheap boilerplate as cheap word-deletions at small P.
P_DELETE_PRIOR = 0.005
P_DELETE_PROPOSAL = 0.20

# Candidate omitted tokens per deletion slot for the lookahead deletion proposal. Swept
# 6-vs-12 at D=1: identical posteriors, ~1.6x faster at K=6, so 6 is the default. Cost is
# linear in K (it drives the P*K lookahead forward).
LOOKAHEAD_K = 6

# Max single-token substitution candidates kept per word (nearest by edit distance) in
# noise_word.word_sub_candidates. This caps the per-word candidate count K used by the filter's
# evidence gather AND -- the dominant cost -- the suffix-aware rejuvenation move, which forwards
# P*K buffers per revisited word. Almost every common word saturated the old cap of 128 (so K was
# effectively pinned at 129 regardless of the sentence); 32 cuts the rejuv move's forward ~4x.
# The dropped candidates are the FARTHEST edits, already crushed by SUB_PARAM**d, so the loss is
# near-zero; nearest-first ordering keeps all distance-1 reanalysis targets (e.g. threats->treats).
# Raise it if a word's genuine distance-1 neighbor set exceeds 32; sweep with eval_rejuv.py.
MAX_SUB_CANDIDATES = 32
