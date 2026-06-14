"""Unified word-scan noisy-channel particle filter: copy / substitution / insertion / deletion
in one model.

This is the single filter the project runs, combining every operation:

- **word-span substitution** (N:1, SymSpell candidates over the word string) -- fixes
  BPE-token-count typos ("experimemt"->"experiment") that the token-level filters can't;
- the **lookahead deletion gap** and **insertion** (ported from ``particle_filter_lookahead``) --
  reconstructs omitted words ("he wants _ go"->"to") and drops spurious ones (doubled words).

Per observed word w (a fixed span of n BPE tokens -- segmentation is deterministic data, so all
particles stay in lockstep):

  Phase A -- deletion gap: up to MAX_DELETIONS omitted *intended* single-token words may be
    posited before w, each proposed lookahead-guided toward w's first token (so a good omitted
    word both is LM-likely and makes w likely next), scored under the model deletion prior.
  Phase B -- explain w with one action:
    COPY  : intended word == w; emit its n tokens, LM chain-rule scored.   action prior COPY.
    SUB(x): intended word is a single vocab token x (char-edit neighbor of w); emit 1 token,
            scored log P_LM(x|ctx) + d*log SUB_PARAM.                       action prior SUB.
    INSERT: w is spurious; consume it, emit nothing; score its n tokens as
            inserts (n * -log V).                                           action prior INSERT.

Then resample (every word). Scope: deletions/substitutions are N:1 (omitted/intended word is a
single BPE token); multi-token intended words remain the deferred M:N extension. Dedup LM
forwards are the default (exact, faster). This is intended to replace the per-filter prototypes;
run.py points here.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from . import lm_penzai as L
from . import noise_word as NW
from .tokenizer import decode
from .particle_filter import (
    ACTION_ALPHAS, MAX_DELETIONS, P_DELETE_PRIOR, P_DELETE_PROPOSAL,
)
from .particle_filter_lookahead import LOOKAHEAD_K
from .model import COPY, SUB, INSERT
from .proposal import propose
from .cache_dedup import make_dedup_fns


def run_particle_filter_unified(key, obs_ids, num_particles=32, max_intended=None,
                                max_dist=2, lookahead_k=LOOKAHEAD_K,
                                p_delete_prior=P_DELETE_PRIOR, progress=False,
                                next_logprobs_fn=None, next_logits_fn=None):
    if next_logprobs_fn is None or next_logits_fn is None:
        _dd_lp, _dd_lo = make_dedup_fns()  # dedup default (exact, ~1.5-2x faster)
        if next_logprobs_fn is None:
            next_logprobs_fn = _dd_lp
        if next_logits_fn is None:
            next_logits_fn = _dd_lo

    log_del = math.log(p_delete_prior) - math.log(P_DELETE_PROPOSAL)
    log_keep = math.log(1 - p_delete_prior) - math.log(1 - P_DELETE_PROPOSAL)

    P = num_particles
    words = NW.segment_words([int(i) for i in obs_ids])
    W = len(words)
    total_obs = sum(len(ids) for ids, _ in words)
    if max_intended is None:
        # copy emits a word's tokens (<= total_obs); each of the W gaps may add MAX_DELETIONS.
        max_intended = total_obs + W * MAX_DELETIONS + 4
    V = L.vocab_size()
    insert_tok_loglik = -math.log(V)  # per spurious observed token
    rows = jnp.arange(P)
    min_ess = float("inf")

    key, prior_key = jax.random.split(key)
    log_action_prior = jnp.log(jax.random.dirichlet(prior_key, ACTION_ALPHAS, shape=(P,)))  # [P,3]
    intended_buf = jnp.full((P, max_intended), L.EOS_ID, dtype=jnp.int32)
    i_len = jnp.ones((P,), dtype=jnp.int32)
    log_marginal = 0.0

    steps = range(W)
    if progress:
        try:
            from tqdm import tqdm
            steps = tqdm(steps, desc=f"unified-SMC (P={P}, W={W})", unit="word")
        except ImportError:
            pass

    for wi in steps:
        span_ids, word_str = words[wi]
        n = len(span_ids)
        obs0 = span_ids[0]  # first token of this word; deletion lookahead target

        # --- Phase A: lookahead deletion gap (omitted intended single-token words) ---
        still_deleting = jnp.ones((P,), dtype=bool)
        log_w_gap = jnp.zeros((P,))
        for _ in range(MAX_DELETIONS):
            key, dk_decide, dk_tok = jax.random.split(key, 3)
            lp_del = next_logprobs_fn(intended_buf, i_len)  # [P, V]
            delete_now = still_deleting & (jax.random.uniform(dk_decide, (P,)) < P_DELETE_PROPOSAL)
            log_w_gap += jnp.where(still_deleting,
                                   jnp.where(delete_now, log_del, log_keep), 0.0)
            cand_lm, cand_ids = jax.lax.top_k(lp_del, lookahead_k)  # [P, K]
            bufs = jnp.repeat(intended_buf, lookahead_k, axis=0)
            ilens = jnp.repeat(i_len, lookahead_k)
            rep_rows = jnp.arange(P * lookahead_k)
            bufs = bufs.at[rep_rows, ilens].set(cand_ids.reshape(-1))
            look_o = jax.nn.log_softmax(
                next_logits_fn(bufs, ilens + 1), axis=-1
            )[:, obs0].reshape(P, lookahead_k)  # [P, K] log LM(obs0 | ctx + x)
            q_logits = cand_lm + look_o
            logZ = logsumexp(q_logits, axis=1)
            choice = jax.random.categorical(dk_tok, q_logits, axis=1)
            del_tok = cand_ids[rows, choice]
            token_w = logZ - look_o[rows, choice]
            log_w_gap += jnp.where(delete_now, token_w, 0.0)
            intended_buf = intended_buf.at[rows, i_len].set(
                jnp.where(delete_now, del_tok, intended_buf[rows, i_len]))
            i_len = i_len + delete_now.astype(jnp.int32)
            still_deleting = still_deleting & delete_now

        # --- Phase B: explain the observed word (copy / sub / insert) ---
        lm0 = next_logprobs_fn(intended_buf, i_len)  # [P, V]

        # COPY: LM chain rule over the word's n tokens. n-1 extra forwards only when n > 1.
        copy_ev = log_action_prior[:, COPY] + lm0[:, obs0]
        tmp_buf = intended_buf.at[rows, i_len].set(obs0)
        tmp_len = i_len + 1
        for j in range(1, n):
            lmj = next_logprobs_fn(tmp_buf, tmp_len)
            copy_ev = copy_ev + lmj[:, span_ids[j]]
            tmp_buf = tmp_buf.at[rows, tmp_len].set(span_ids[j])
            tmp_len = tmp_len + 1

        # SUB candidates (single-token intended words within max_dist of the observed word).
        subs = NW.word_sub_candidates(word_str, max_dist=max_dist)
        n_sub = len(subs)
        cols = [copy_ev[:, None]]
        if n_sub:
            sub_x = jnp.array([x for x, _ in subs], dtype=jnp.int32)
            sub_ll = jnp.array([NW.word_sub_loglik(d) for _, d in subs], dtype=jnp.float32)
            cols.append(log_action_prior[:, SUB][:, None] + lm0[:, sub_x] + sub_ll[None, :])

        # INSERT: the whole observed word is spurious; consume it, emit nothing.
        insert_ev = log_action_prior[:, INSERT] + n * insert_tok_loglik  # [P]
        cols.append(insert_ev[:, None])

        log_ev = jnp.concatenate(cols, axis=1)  # [P, 1 + n_sub + 1]
        C = log_ev.shape[1]

        # Local-posterior proposal; gap weight folds into the incremental weight.
        key, step_key, resample_key = jax.random.split(key, 3)
        option, log_w = propose(step_key, log_ev)
        log_w = log_w + log_w_gap
        log_marginal += float(logsumexp(log_w) - jnp.log(P))
        min_ess = min(min_ess, float(1.0 / jnp.sum(jax.nn.softmax(log_w) ** 2)))

        parents = jax.random.categorical(resample_key, log_w - logsumexp(log_w), shape=(P,))
        intended_buf = intended_buf[parents]
        i_len = i_len[parents]
        log_action_prior = log_action_prior[parents]
        option = option[parents]

        # Emit the chosen candidate's intended tokens (COPY=n, SUB=1, INSERT=0).
        cand_tok = np.zeros((C, n), dtype=np.int32)
        cand_len = np.zeros((C,), dtype=np.int32)
        cand_tok[0, :n] = span_ids
        cand_len[0] = n
        for k, (x, _) in enumerate(subs):
            cand_tok[1 + k, 0] = x
            cand_len[1 + k] = 1
        # last column = INSERT, cand_len 0 (emit nothing)
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
