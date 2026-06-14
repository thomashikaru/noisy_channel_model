"""Generative semantics for one observed step of the token-level noisy-channel model.

For a single observed token ``o_t``, the generative story (conditioned on a particle's
intended-context so far) enumerates a small set of explanations:

- *emitting* options: an intended token ``x`` is drawn from the LM and emitted as ``o_t``
  either unchanged (``copy``) or via a form substitution (``sub``). Each emitting option
  contributes joint log-density ``log P(action) + log P_LM(x | context) + log P(o_t | x, action)``.
- the *insert* option: ``o_t`` is spurious, no intended token is consumed; joint log-density
  ``log P(insert) + log P(o_t | insert)``. (The phantom LM draw cancels against the proposal,
  so it is omitted.)

``step_log_evidence`` assembles these per-option joint log-densities as ``[P, K+1]`` (the
final column is insert). Stacked this way it is simultaneously the model's joint over the
enumerated latent and -- once normalized -- the locally-optimal proposal (see ``proposal.py``).
This is the token-level analog of ``choose_action`` in the original ``gen_inference.jl``.

Action indices: COPY=0, SUB=1, INSERT=2.
"""

import jax.numpy as jnp

COPY, SUB, INSERT = 0, 1, 2


def step_log_evidence(
    lm_logprobs,        # [P, V]  normalized next-token log-probs per particle
    log_action_prior,   # [P, 3]  log P(action) per particle (copy, sub, insert)
    cand_x,             # [K]     emitting-candidate intended token ids (cand_x[0] == o_t, the copy)
    cand_obs_loglik,    # [K]     log P(o_t | x, action) for each emitting candidate
    cand_is_copy,       # [K]     bool: candidate uses the copy action (else sub)
    insertion_loglik,   # scalar  log P(o_t | insert)
):
    """Return per-option joint log-density ``[P, K+1]`` (last column = insert)."""
    # LM log-prob of each emitting candidate's intended token, per particle: [P, K]
    lm_x = lm_logprobs[:, cand_x]

    # Action-prior mass for each emitting candidate (copy vs sub): [K] -> broadcast [P, K]
    action_idx = jnp.where(cand_is_copy, COPY, SUB)            # [K]
    emit_action_lp = log_action_prior[:, action_idx]           # [P, K]

    emit_ev = emit_action_lp + lm_x + cand_obs_loglik[None, :]  # [P, K]

    insert_ev = log_action_prior[:, INSERT] + insertion_loglik  # [P]
    return jnp.concatenate([emit_ev, insert_ev[:, None]], axis=1)  # [P, K+1]
