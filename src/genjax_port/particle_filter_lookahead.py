"""Prototype: lookahead-aware deletion proposal (Option A) to make omitted-word
reconstruction reliable.

A deletion emits no observation, so the baseline filter samples the dropped intended token
*blindly* from the LM prior -- the right omitted word (e.g. "the") rarely gets proposed at a
50k vocab. Here we make the deletion proposal *data-aware* using one-step lookahead: the
dropped token ``x`` sits between the current context and the next observed token ``o``, so a
good ``x`` both is likely under the LM and makes ``o`` likely next.

For each particle we take the top-``K`` candidate tokens by ``LM(x | context)``, score each by
the lookahead ``LM(o | context + x)`` (one extra batched forward over ``P*K`` sequences), and
sample from ``q(x) ∝ LM(x | context) · LM(o | context + x)``. The importance weight for the
token then becomes ``log Z - log LM(o | context + x)`` where ``Z = Σ_x LM(x|ctx)·LM(o|ctx+x)``
(the LM-prior factor cancels). Restricting ``q`` to the top-K is a biased-but-reasonable
proposal restriction (we only reconstruct high-LM-probability omissions), the same spirit as
the edit-neighbor restriction for substitutions.

Everything else (model, emission proposal, weights, always-resample) is identical to
:mod:`particle_filter`; only the deletion phase differs.
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
from .particle_filter import (
    ACTION_ALPHAS, MAX_DELETIONS, P_DELETE_PRIOR, P_DELETE_PROPOSAL,
    _build_candidates,
)

LOOKAHEAD_K = 6  # candidate omitted tokens per deletion slot. Swept 6-vs-12 at D=1: identical
                 # posteriors, ~1.6x faster at K=6, so 6 is the default. Cost is linear in K
                 # (drives the P*K lookahead forward).


def run_particle_filter_lookahead(key, obs_ids, num_particles=32, max_intended=None,
                                  lookahead_k=LOOKAHEAD_K, p_delete_prior=P_DELETE_PRIOR,
                                  progress=False,
                                  next_logprobs_fn=None, next_logits_fn=None):
    # The LM forwards are injectable. They DEFAULT to the prefix-dedup wrappers (cache_dedup),
    # which compute only the unique buffer rows -- the every-step resampling makes the particle
    # set ~75-90% redundant, giving a measured ~1.5-2x speedup that grows with sentence length.
    # Dedup is numerically EXACT (each batch row is independent), so it is safe as the default;
    # pass next_*_fn=L.next_token_{logprobs,logits} to disable (e.g. for an A/B baseline).
    if next_logprobs_fn is None or next_logits_fn is None:
        from .cache_dedup import make_dedup_fns
        _dd_logprobs, _dd_logits = make_dedup_fns()
        if next_logprobs_fn is None:
            next_logprobs_fn = _dd_logprobs
        if next_logits_fn is None:
            next_logits_fn = _dd_logits
    # Deletion decision weight terms (model prior vs proposal rate); p_delete_prior is
    # exposed so we can show that reconstruction is governed by the MODEL prior, not the
    # proposal.
    log_del = math.log(p_delete_prior) - math.log(P_DELETE_PROPOSAL)
    log_keep = math.log(1 - p_delete_prior) - math.log(1 - P_DELETE_PROPOSAL)
    P = num_particles
    T = int(obs_ids.shape[0])
    if max_intended is None:
        max_intended = T * (1 + MAX_DELETIONS) + 2
    V = L.vocab_size()
    insertion_loglik = N.insertion_loglik(V)
    rows = jnp.arange(P)
    min_ess = float("inf")

    key, prior_key = jax.random.split(key)
    log_action_prior = jnp.log(jax.random.dirichlet(prior_key, ACTION_ALPHAS, shape=(P,)))
    intended_buf = jnp.full((P, max_intended), L.EOS_ID, dtype=jnp.int32)
    i_len = jnp.ones((P,), dtype=jnp.int32)
    log_marginal = 0.0

    steps = range(T)
    if progress:
        # Per-token progress: the first step pays the JIT-compile cost, so it is much slower
        # than the rest -- the bar makes that visible instead of looking hung.
        try:
            from tqdm import tqdm
            steps = tqdm(steps, desc=f"SMC (P={P}, T={T})", unit="tok")
        except ImportError:
            pass

    for t in steps:
        obs_id = int(obs_ids[t])
        cand_x, cand_obs_loglik, cand_is_copy = _build_candidates(obs_id)

        # Phase A -- lookahead-aware deletion gap.
        still_deleting = jnp.ones((P,), dtype=bool)
        log_w_gap = jnp.zeros((P,))
        for _ in range(MAX_DELETIONS):
            key, dk_decide, dk_tok = jax.random.split(key, 3)
            lp_del = next_logprobs_fn(intended_buf, i_len)  # [P, V]
            delete_now = still_deleting & (jax.random.uniform(dk_decide, (P,)) < P_DELETE_PROPOSAL)
            log_w_gap += jnp.where(still_deleting,
                                   jnp.where(delete_now, log_del, log_keep), 0.0)

            # Top-K candidate dropped tokens by LM(x | context).
            cand_lm, cand_ids = jax.lax.top_k(lp_del, lookahead_k)  # [P, K]

            # Lookahead: LM(obs_id | context + candidate) via one batched forward [P*K, .].
            bufs = jnp.repeat(intended_buf, lookahead_k, axis=0)     # [P*K, M]
            ilens = jnp.repeat(i_len, lookahead_k)                   # [P*K]
            rep_rows = jnp.arange(P * lookahead_k)
            bufs = bufs.at[rep_rows, ilens].set(cand_ids.reshape(-1))
            look_o = jax.nn.log_softmax(
                next_logits_fn(bufs, ilens + 1), axis=-1
            )[:, obs_id].reshape(P, lookahead_k)                     # [P, K] log LM(o | ctx+x)

            q_logits = cand_lm + look_o                             # [P, K]  unnormalized log q
            logZ = logsumexp(q_logits, axis=1)                      # [P]
            choice = jax.random.categorical(dk_tok, q_logits, axis=1)
            del_tok = cand_ids[rows, choice]                        # [P]
            # token weight = log p_model(x) - log q(x) = logZ - log LM(o | ctx+x_chosen)
            token_w = logZ - look_o[rows, choice]
            log_w_gap += jnp.where(delete_now, token_w, 0.0)

            intended_buf = intended_buf.at[rows, i_len].set(
                jnp.where(delete_now, del_tok, intended_buf[rows, i_len]))
            i_len = i_len + delete_now.astype(jnp.int32)
            still_deleting = still_deleting & delete_now

        # Phase B -- emission (identical to the baseline filter).
        lm_logprobs = next_logprobs_fn(intended_buf, i_len)
        log_ev = step_log_evidence(lm_logprobs, log_action_prior,
                                   cand_x, cand_obs_loglik, cand_is_copy, insertion_loglik)
        key, step_key, resample_key = jax.random.split(key, 3)
        option, log_w = propose(step_key, log_ev)
        log_w = log_w + log_w_gap
        log_marginal += logsumexp(log_w) - jnp.log(P)
        min_ess = min(min_ess, float(1.0 / jnp.sum(jax.nn.softmax(log_w) ** 2)))

        parents = jax.random.categorical(resample_key, log_w - logsumexp(log_w), shape=(P,))
        intended_buf = intended_buf[parents]
        i_len = i_len[parents]
        log_action_prior = log_action_prior[parents]
        option = option[parents]

        K = cand_x.shape[0]
        emitting = option < K
        emitted_token = cand_x[jnp.clip(option, 0, K - 1)]
        intended_buf = intended_buf.at[rows, i_len].set(
            jnp.where(emitting, emitted_token, intended_buf[rows, i_len]))
        i_len = i_len + emitting.astype(jnp.int32)

    sentences = [decode(intended_buf[p, 1:int(i_len[p])]).strip() for p in range(P)]
    return sentences, float(log_marginal), min_ess
