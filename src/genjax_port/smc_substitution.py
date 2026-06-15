"""M1: substitution-only word-scan SMC -- the genjax-native port's forward filter.

Scope: COPY + SUB only (deletions are M2, insertion M3). Per observed word the model weighs
explanations exactly as the per-word ``Switch`` in :mod:`genjax_model` (``make_word_model``):

  COPY : intended word == observed word; emit its n BPE tokens, LM chain-rule scored.
  SUB  : intended word is a single vocab token x (char-edit neighbor of the observed word);
         emit 1 token, scored ``log P_LM(x | ctx) + word_sub_loglik(d)``.

``word_log_evidence`` assembles the per-branch joint log-densities ``[P, 1 + n_sub]`` -- the same
quantity ``make_word_model(...).importance`` returns per branch (cross-checked in
``tests/test_smc_substitution.py``). The forward filter computes it directly (one LM forward,
gather all sub candidates) rather than enumerating ``Switch`` branches through ``importance``, so
it scales to many candidates; the ``@gen`` word model is the trace carrier for M5 rejuvenation.

With the local-posterior proposal (``proposal.propose``) the incremental SMC weight collapses to
``logsumexp`` over branches; we resample every word. Particle state is manual ``vmap``-friendly
buffers (``intended_buf [P, M]``, ``i_len [P]``), mirroring the hand-rolled unified filter.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from . import lm_penzai as L
from . import noise_word as NW
from .tokenizer import decode
from .particle_filter import ACTION_ALPHAS
from .model import COPY, SUB
from .proposal import propose


def word_log_evidence(intended_buf, i_len, log_action_prior, span_ids, subs,
                      next_logprobs_fn):
    """Per-branch joint log-density for one observed word: ``[P, 1 + len(subs)]``.

    Column 0 is COPY (LM chain-rule over the word's ``span_ids``); columns ``1..`` are the SUB
    candidates ``subs = [(token_id, char_dist), ...]``. Matches ``make_word_model`` branch
    importances (see test). ``next_logprobs_fn(buf, i_len) -> [P, V]`` is injectable so dedup /
    a stub LM can be swapped in.
    """
    P = intended_buf.shape[0]
    rows = jnp.arange(P)
    n = len(span_ids)
    obs0 = span_ids[0]

    lm0 = next_logprobs_fn(intended_buf, i_len)              # [P, V]
    copy_ev = log_action_prior[:, COPY] + lm0[:, obs0]       # [P]
    tmp_buf = intended_buf.at[rows, i_len].set(obs0)
    tmp_len = i_len + 1
    for j in range(1, n):                                    # extra forwards only when n > 1
        lmj = next_logprobs_fn(tmp_buf, tmp_len)
        copy_ev = copy_ev + lmj[:, span_ids[j]]
        tmp_buf = tmp_buf.at[rows, tmp_len].set(span_ids[j])
        tmp_len = tmp_len + 1

    cols = [copy_ev[:, None]]
    if subs:
        sub_x = jnp.asarray([x for x, _ in subs], jnp.int32)
        sub_ll = jnp.asarray([NW.word_sub_loglik(d) for _, d in subs], jnp.float32)
        cols.append(log_action_prior[:, SUB][:, None] + lm0[:, sub_x] + sub_ll[None, :])
    return jnp.concatenate(cols, axis=1)                     # [P, 1 + n_sub]


def run_smc_substitution(key, obs_ids, num_particles=64, max_intended=None,
                         max_dist=2, progress=False, next_logprobs_fn=None):
    """Substitution-only word-scan SMC. Returns ``(sentences, log_marginal, min_ess)``."""
    if next_logprobs_fn is None:
        next_logprobs_fn = L.next_token_logprobs

    P = num_particles
    words = NW.segment_words([int(i) for i in obs_ids])
    W = len(words)
    total_obs = sum(len(ids) for ids, _ in words)
    if max_intended is None:
        max_intended = total_obs + 4  # subs only shrink length; +slack for the EOS seed
    rows = jnp.arange(P)
    min_ess = float("inf")

    key, prior_key = jax.random.split(key)
    log_action_prior = jnp.log(
        jax.random.dirichlet(prior_key, jnp.asarray(ACTION_ALPHAS, jnp.float32), shape=(P,)))
    intended_buf = jnp.full((P, max_intended), L.EOS_ID, jnp.int32)
    i_len = jnp.ones((P,), jnp.int32)
    log_marginal = 0.0

    steps = range(W)
    if progress:
        try:
            from tqdm import tqdm
            steps = tqdm(steps, desc=f"sub-SMC (P={P}, W={W})", unit="word")
        except ImportError:
            pass

    for wi in steps:
        span_ids, word_str = words[wi]
        n = len(span_ids)
        subs = NW.word_sub_candidates(word_str, max_dist=max_dist)

        log_ev = word_log_evidence(intended_buf, i_len, log_action_prior, span_ids, subs,
                                   next_logprobs_fn)
        Cn = log_ev.shape[1]

        key, step_key, resample_key = jax.random.split(key, 3)
        option, log_w = propose(step_key, log_ev)
        log_marginal += float(logsumexp(log_w) - jnp.log(P))
        min_ess = min(min_ess, float(1.0 / jnp.sum(jax.nn.softmax(log_w) ** 2)))

        parents = jax.random.categorical(resample_key, log_w - logsumexp(log_w), shape=(P,))
        intended_buf = intended_buf[parents]
        i_len = i_len[parents]
        log_action_prior = log_action_prior[parents]
        option = option[parents]

        # Emit the chosen branch's intended tokens (COPY = n span tokens, SUB = 1 token).
        cand_tok = np.zeros((Cn, n), np.int32)
        cand_len = np.zeros((Cn,), np.int32)
        cand_tok[0, :n] = span_ids
        cand_len[0] = n
        for k, (x, _) in enumerate(subs):
            cand_tok[1 + k, 0] = x
            cand_len[1 + k] = 1
        cand_tok = jnp.asarray(cand_tok)
        cand_len = jnp.asarray(cand_len)
        chosen_tok = cand_tok[option]
        chosen_len = cand_len[option]
        for j in range(n):
            writing = j < chosen_len
            intended_buf = intended_buf.at[rows, i_len].set(
                jnp.where(writing, chosen_tok[:, j], intended_buf[rows, i_len]))
            i_len = i_len + writing.astype(jnp.int32)

    sentences = [decode(intended_buf[p, 1:int(i_len[p])]).strip() for p in range(P)]
    return sentences, float(log_marginal), min_ess
