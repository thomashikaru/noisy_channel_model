"""Token-level SMC particle filter for the GPT-class noisy-channel model.

Mirrors Gen.jl's ``initialize_particle_filter`` / ``particle_filter_step!``, but at BPE
token granularity with an in-graph LM and a data-driven proposal (the toy plain-importance
filter degenerates at a 50k vocab). For each observed token we:

1. build the small candidate set of intended-token explanations on the host (``noise.py``),
2. score them with a batched LM forward across particles (``lm_penzai.py``) + the noise model
   (``model.step_log_evidence``),
3. sample one explanation per particle from its local posterior and weight by the one-step
   predictive likelihood (``proposal.propose``),
4. always resample, then advance each particle's intended-token buffer.

M2 scope: actions copy / sub / insert (no deletions; ``D=0``). The intended sentence is the
sequence of emitted intended tokens (insert adds nothing). The observed axis is the loop
bound; particles are the vectorized axis.
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from . import lm_penzai as L
from . import noise as N
from .model import step_log_evidence
from .proposal import propose
from .tokenizer import decode

# Dirichlet concentration for the action prior (copy, sub, insert); copy favored,
# matching --normal_alpha=3 / --error_alpha=1 in the original config.
ACTION_ALPHAS = jnp.array([3.0, 1.0, 1.0])

# Deletion gap (M3): before each emission a particle may posit up to MAX_DELETIONS
# intended tokens that were dropped (produce no observation), reconstructing an omitted
# word. They are proposed from the LM prior (no observation to guide them), so they
# contribute nothing to the importance weight directly -- they only reshape the LM context,
# and the next emission's predictive likelihood (via resampling) selects the helpful ones.
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
# the more powerful M4+ fix.)
P_DELETE_PRIOR = 0.02
P_DELETE_PROPOSAL = 0.20
_LOG_DEL = math.log(P_DELETE_PRIOR) - math.log(P_DELETE_PROPOSAL)            # < 0: cost to delete
_LOG_KEEP = math.log(1 - P_DELETE_PRIOR) - math.log(1 - P_DELETE_PROPOSAL)   # > 0: credit to stop


def _build_candidates(obs_id):
    """Host-side candidate arrays for one observed token (copy first, then subs)."""
    subs = N.sub_candidates(obs_id)
    cand_x = jnp.array([obs_id] + [x for x, _ in subs], dtype=jnp.int32)
    cand_obs_loglik = jnp.array([0.0] + [ll for _, ll in subs], dtype=jnp.float32)
    cand_is_copy = jnp.array([True] + [False] * len(subs))
    return cand_x, cand_obs_loglik, cand_is_copy


def run_particle_filter(key, obs_ids, num_particles=32, max_intended=None):
    """Run the token-level filter over an observed token sequence.

    Args:
        key: ``jax.random.key``.
        obs_ids: int array ``[T_obs]`` of observed BPE token ids.
        num_particles: number of particles ``P``.
        max_intended: buffer size; defaults to ``T_obs + 2`` (no deletions in M2).

    Returns:
        sentences: list of ``P`` decoded intended-sentence strings.
        log_marginal: scalar estimate of ``log P(observed)``.
        min_ess: smallest per-step effective sample size (variance diagnostic).
    """
    P = num_particles
    T = int(obs_ids.shape[0])
    if max_intended is None:
        # copy/sub emit one intended token each; each emission may be preceded by up to
        # MAX_DELETIONS extra intended tokens, plus the BOS seed.
        max_intended = T * (1 + MAX_DELETIONS) + 2
    V = L.vocab_size()
    insertion_loglik = N.insertion_loglik(V)
    rows = jnp.arange(P)
    min_ess = float("inf")

    key, prior_key = jax.random.split(key)
    log_action_prior = jnp.log(
        jax.random.dirichlet(prior_key, ACTION_ALPHAS, shape=(P,))
    )  # [P, 3]

    # Buffers: position 0 seeded with EOS_ID as the start-of-sequence context.
    intended_buf = jnp.full((P, max_intended), L.EOS_ID, dtype=jnp.int32)
    i_len = jnp.ones((P,), dtype=jnp.int32)  # only the BOS seed filled so far
    log_marginal = 0.0

    for t in range(T):
        obs_id = int(obs_ids[t])
        cand_x, cand_obs_loglik, cand_is_copy = _build_candidates(obs_id)

        # Phase A -- deletion gap: up to MAX_DELETIONS intended tokens that emit nothing,
        # reconstructing omitted words. The deleted token is proposed from the LM prior (so
        # it cancels in the weight), but the delete/stop *decision* is proposed at
        # P_DELETE_PROPOSAL while scored under the model prior P_DELETE_PRIOR, accruing a
        # per-decision weight correction in log_w_gap (a real cost for positing a deletion).
        still_deleting = jnp.ones((P,), dtype=bool)
        log_w_gap = jnp.zeros((P,))
        for _ in range(MAX_DELETIONS):
            key, dk_decide, dk_tok = jax.random.split(key, 3)
            lp_del = L.next_token_logprobs(intended_buf, i_len)  # [P, V]
            delete_now = still_deleting & (jax.random.uniform(dk_decide, (P,)) < P_DELETE_PROPOSAL)
            # Weight correction only for particles that actually made a decision this slot.
            decision_w = jnp.where(delete_now, _LOG_DEL, _LOG_KEEP)
            log_w_gap += jnp.where(still_deleting, decision_w, 0.0)
            del_tok = jax.random.categorical(dk_tok, lp_del, axis=1)  # ~ LM prior
            intended_buf = intended_buf.at[rows, i_len].set(
                jnp.where(delete_now, del_tok, intended_buf[rows, i_len])
            )
            i_len = i_len + delete_now.astype(jnp.int32)
            still_deleting = still_deleting & delete_now  # stop after first non-delete

        # Phase B -- emission of the observed token (copy / sub / insert).
        lm_logprobs = L.next_token_logprobs(intended_buf, i_len)  # [P, V]
        log_ev = step_log_evidence(
            lm_logprobs, log_action_prior,
            cand_x, cand_obs_loglik, cand_is_copy, insertion_loglik,
        )  # [P, K+1]

        key, step_key, resample_key = jax.random.split(key, 3)
        option, log_w = propose(step_key, log_ev)  # [P], [P]
        log_w = log_w + log_w_gap  # add the deletion-gap weight corrections from Phase A

        log_marginal += logsumexp(log_w) - jnp.log(P)

        # Effective sample size diagnostic (on the incremental weights, pre-resample).
        norm_w = jax.nn.softmax(log_w)
        ess = 1.0 / jnp.sum(norm_w ** 2)
        min_ess = min(min_ess, float(ess))

        # Always resample; reindex all per-particle state by the chosen parents.
        parents = jax.random.categorical(resample_key, log_w - logsumexp(log_w), shape=(P,))
        intended_buf = intended_buf[parents]
        i_len = i_len[parents]
        log_action_prior = log_action_prior[parents]
        option = option[parents]

        # Advance buffers: emitting options (option < K) append their intended token;
        # the insert option (option == K) leaves the intended sequence unchanged.
        K = cand_x.shape[0]
        emitting = option < K
        emitted_token = cand_x[jnp.clip(option, 0, K - 1)]
        intended_buf = intended_buf.at[rows, i_len].set(
            jnp.where(emitting, emitted_token, intended_buf[rows, i_len])
        )
        i_len = i_len + emitting.astype(jnp.int32)

    sentences = [
        decode(intended_buf[p, 1:int(i_len[p])]).strip()  # drop BOS at position 0
        for p in range(P)
    ]
    return sentences, float(log_marginal), min_ess
