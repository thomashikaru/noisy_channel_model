"""Word-scan SMC -- the genjax port's filtering sweep (M1 substitution + M2 deletion).

Per observed word the model weighs explanations exactly as the per-word ``Switch`` in
:mod:`genjax_model` (``make_word_model``):

  COPY : intended word == observed word; emit its n BPE tokens, LM chain-rule scored.
  SUB  : intended word is a single vocab token x (char-edit neighbor of the observed word);
         emit 1 token, scored ``log P_LM(x | ctx) + word_sub_loglik(d)``.

``word_log_evidence`` assembles the per-branch joint log-densities ``[P, 1 + n_sub]`` -- the same
quantity ``make_word_model(...).importance`` returns per branch (cross-checked in
``tests/test_smc_substitution.py``). The filtering sweep computes it directly (one LM forward,
gather all sub candidates) rather than enumerating ``Switch`` branches through ``importance``, so
it scales to many candidates; the ``@gen`` word model is the trace carrier for M5 rejuvenation.

**Deletion (M2, ``max_deletions > 0``).** Before each observed word a particle may posit up to
``max_deletions`` omitted *intended* single-token words (a "gap"), each proposed lookahead-guided
toward the word's first token (``deletion_gap``, ported from the unified filter's Phase A). The gap
contributes a weight ``log_w_gap`` that folds into the step's incremental weight. Insertion is M3.

With the local-posterior proposal (``proposal.propose``) the per-word incremental SMC weight
collapses to ``logsumexp`` over branches (times the gap weight); we resample every word. Particle
state is manual ``vmap``-friendly buffers (``intended_buf [P, M]``, ``i_len [P]``), mirroring the
hand-rolled unified filter.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from . import lm_penzai as L
from . import noise_word as NW
from .cache_dedup import make_dedup_fns
from .tokenizer import decode
from .particle_filter import (
    ACTION_ALPHAS, MAX_DELETIONS, P_DELETE_PRIOR, P_DELETE_PROPOSAL,
)
from .particle_filter_lookahead import LOOKAHEAD_K
from .model import COPY, SUB, INSERT
from .proposal import propose


def word_log_evidence(intended_buf, i_len, log_action_prior, span_ids, subs,
                      next_logprobs_fn, insertion_loglik=None):
    """Per-branch joint log-density for one observed word: ``[P, 1 + len(subs) (+ 1)]``.

    Column 0 is COPY (LM chain-rule over the word's ``span_ids``); columns ``1..len(subs)`` are the
    SUB candidates ``subs = [(token_id, char_dist), ...]``. If ``insertion_loglik`` is given (M3), a
    final INSERT column is appended (the observed word is spurious; emits nothing), scored
    ``log P(insert) + n * insertion_loglik`` with ``insertion_loglik = -log V``. The COPY/SUB columns
    match ``make_word_model`` branch importances (see test). ``next_logprobs_fn(buf, i_len) -> [P, V]``
    is injectable so dedup / a stub LM can be swapped in.
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
    if insertion_loglik is not None:                         # INSERT: spurious word, emit nothing
        insert_ev = log_action_prior[:, INSERT] + n * insertion_loglik   # [P]
        cols.append(insert_ev[:, None])
    return jnp.concatenate(cols, axis=1)                     # [P, 1 + n_sub (+ 1)]


def deletion_gap(key, intended_buf, i_len, obs0, max_deletions, lookahead_k,
                 log_del, log_keep, next_logprobs_fn, next_logits_fn):
    """Posit up to ``max_deletions`` omitted single-token intended words before the observed word
    whose first token is ``obs0`` (the unified filter's Phase A, vmapped over particles).

    Each deletion slot: decide to delete with proposal rate ``P_DELETE_PROPOSAL`` (priced against
    ``P_DELETE_PRIOR`` via ``log_del``/``log_keep``); if deleting, propose the omitted token from
    the LM top-``lookahead_k`` reweighted by how well it makes ``obs0`` likely next (lookahead), and
    fold the proposal correction into the gap weight. Returns ``(key, intended_buf, i_len,
    log_w_gap)`` with the buffer/length advanced by the posited deletions.
    """
    P = intended_buf.shape[0]
    rows = jnp.arange(P)
    still_deleting = jnp.ones((P,), dtype=bool)
    log_w_gap = jnp.zeros((P,))
    for _ in range(max_deletions):
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
    return key, intended_buf, i_len, log_w_gap


def run_smc_substitution(key, obs_ids, num_particles=64, max_intended=None,
                         max_dist=2, max_deletions=0, allow_insertion=False,
                         lookahead_k=LOOKAHEAD_K, p_delete_prior=P_DELETE_PRIOR, progress=False,
                         dedup=True, dedup_stats=None, return_state=False,
                         post_resample_hook=None,
                         next_logprobs_fn=None, next_logits_fn=None):
    """Word-scan SMC (substitution + optional deletion gap + optional insertion).

    ``max_deletions=0`` and ``allow_insertion=False`` (defaults) give the pure-substitution M1
    filter; ``max_deletions > 0`` enables the M2 lookahead deletion gap; ``allow_insertion=True``
    enables the M3 INSERT action (spurious observed word). Returns
    ``(sentences, log_marginal, min_ess)``, or, with ``return_state=True``,
    ``(sentences, log_marginal, min_ess, (intended_buf, i_len, log_action_prior))`` -- the final
    per-particle buffers, used by the rejuvenation bridge (:mod:`rejuv_bridge`) to materialize a
    genjax trace per particle for post-sweep reanalysis.

    ``dedup`` (default on, like the reference filter) routes the LM forwards through
    :func:`cache_dedup.make_dedup_fns`, which collapses identical intended-prefix rows to one LM
    call and scatters the result back -- numerically exact (same RNG => same posterior) but much
    cheaper after a resample degenerates the particle set (biggest win on the ``[P*lookahead_k]``
    deletion-lookahead batch). Pass a ``cache_dedup.DedupStats()`` as ``dedup_stats`` to measure the
    saved fraction. An explicitly injected ``next_logprobs_fn`` / ``next_logits_fn`` overrides dedup
    for that seam (a caller-supplied stub LM is used as-is).
    """
    if dedup and (next_logprobs_fn is None or next_logits_fn is None):
        _dd_lp, _dd_lo = make_dedup_fns(dedup_stats)
        if next_logprobs_fn is None:
            next_logprobs_fn = _dd_lp
        if next_logits_fn is None:
            next_logits_fn = _dd_lo
    if next_logprobs_fn is None:
        next_logprobs_fn = L.next_token_logprobs
    if next_logits_fn is None:
        next_logits_fn = L.next_token_logits
    insertion_loglik = -math.log(L.vocab_size()) if allow_insertion else None

    log_del = math.log(p_delete_prior) - math.log(P_DELETE_PROPOSAL)
    log_keep = math.log(1 - p_delete_prior) - math.log(1 - P_DELETE_PROPOSAL)

    P = num_particles
    words = NW.segment_words([int(i) for i in obs_ids])
    W = len(words)
    total_obs = sum(len(ids) for ids, _ in words)
    if max_intended is None:
        # subs shrink length; each of the W gaps may add max_deletions tokens; +EOS-seed slack
        max_intended = total_obs + W * max_deletions + 4
    rows = jnp.arange(P)
    min_ess = float("inf")

    key, prior_key = jax.random.split(key)
    log_action_prior = jnp.log(
        jax.random.dirichlet(prior_key, jnp.asarray(ACTION_ALPHAS, jnp.float32), shape=(P,)))
    intended_buf = jnp.full((P, max_intended), L.EOS_ID, jnp.int32)
    i_len = jnp.ones((P,), jnp.int32)
    # Per-observed-word alignment: align[p, t] = buffer position of word t's emitted intended token
    # (-1 if particle p chose INSERT, i.e. the word emitted nothing). Lets the interleaved
    # rejuvenation locate each observed word's token even when deletions/insertions have shifted the
    # buffer per particle (the 1:1 "position == 1 + word index" assumption no longer holds).
    align = jnp.full((P, W), -1, jnp.int32)
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

        # Phase A (M2): lookahead deletion gap before the observed word. The gap advances the
        # buffer (posited omitted tokens enter the LM context) and contributes log_w_gap.
        log_w_gap = jnp.zeros((P,))
        if max_deletions:
            key, intended_buf, i_len, log_w_gap = deletion_gap(
                key, intended_buf, i_len, span_ids[0], max_deletions, lookahead_k,
                log_del, log_keep, next_logprobs_fn, next_logits_fn)

        log_ev = word_log_evidence(intended_buf, i_len, log_action_prior, span_ids, subs,
                                   next_logprobs_fn, insertion_loglik)
        Cn = log_ev.shape[1]

        key, step_key, resample_key = jax.random.split(key, 3)
        option, log_w = propose(step_key, log_ev)
        log_w = log_w + log_w_gap
        step_lmw = float(logsumexp(log_w) - jnp.log(P))   # log mean weight = word's log-evidence
        log_marginal += step_lmw
        min_ess = min(min_ess, float(1.0 / jnp.sum(jax.nn.softmax(log_w) ** 2)))

        parents = jax.random.categorical(resample_key, log_w - logsumexp(log_w), shape=(P,))
        intended_buf = intended_buf[parents]
        i_len = i_len[parents]
        log_action_prior = log_action_prior[parents]
        align = align[parents]
        option = option[parents]
        word_pos = i_len                                  # this word's token lands here (pre-emission)

        # Emit the chosen branch's intended tokens (COPY = n span tokens, SUB = 1 token,
        # INSERT = 0 tokens). The optional trailing INSERT column keeps its zero-init length, so a
        # particle that chose INSERT writes nothing.
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
        align = align.at[rows, wi].set(jnp.where(chosen_len > 0, word_pos, jnp.int32(-1)))

        # Post-resample rejuvenation hook (vectorized over particles): given this word's surprisal
        # (-step_lmw), the hook may run a windowed MH reanalysis and return an updated buffer. Used
        # by rejuv_bridge for interleaved conditional rejuvenation; None = plain filtering sweep.
        # ``align`` lets the hook find each observed word's token under deletion/insertion shifts.
        if post_resample_hook is not None:
            key, intended_buf = post_resample_hook(
                wi, key, intended_buf, i_len, align, log_action_prior, -step_lmw)

    sentences = [decode(intended_buf[p, 1:int(i_len[p])]).strip() for p in range(P)]
    if return_state:
        return sentences, float(log_marginal), min_ess, (intended_buf, i_len, log_action_prior)
    return sentences, float(log_marginal), min_ess


def required_buffer_size(obs_ids, max_deletions=0):
    """Minimum ``max_intended`` for a sentence (same formula ``run_smc_substitution`` uses when
    ``max_intended is None``): the EOS seed + an all-COPY intended sequence + one deletion per gap,
    plus slack. A bucket must be >= this for every sentence it holds.
    """
    words = NW.segment_words([int(i) for i in obs_ids])
    total_obs = sum(len(ids) for ids, _ in words)
    return total_obs + len(words) * max_deletions + 4


def run_smc_batch(key, obs_id_list, bucket, num_particles=64, max_dist=2, max_deletions=0,
                  allow_insertion=False, progress=False, **kw):
    """Run the filter on many sentences in ONE process at a FIXED buffer width ``bucket``.

    Fixing ``max_intended = bucket`` keeps the two compiled LM-forward shapes (``[P, bucket]`` and
    the lookahead ``[P*lookahead_k, bucket]``) constant across sentences, so the ~8s/410m XLA
    compile is paid once (on the first sentence) and every later sentence runs warm (~exec only) --
    the bucketing optimization from the latency note. Padding is inert (EOS-padded positions are
    ignored by causal attention; ``i_len`` tracks the true length), so results are identical to the
    per-sentence ``max_intended``. Every sentence must fit: ``required_buffer_size(obs) <= bucket``.

    Returns a list of ``(sentences, log_marginal, min_ess)``, one per input.
    """
    too_big = [(i, required_buffer_size(o, max_deletions)) for i, o in enumerate(obs_id_list)
               if required_buffer_size(o, max_deletions) > bucket]
    if too_big:
        raise ValueError(
            f"bucket={bucket} too small for sentences {[(i, n) for i, n in too_big]}; "
            f"raise the bucket to >= {max(n for _, n in too_big)}")
    results = []
    items = obs_id_list
    if progress:
        try:
            from tqdm import tqdm
            items = tqdm(obs_id_list, desc=f"batch (bucket={bucket})", unit="sent")
        except ImportError:
            pass
    for obs in items:
        key, sub = jax.random.split(key)
        results.append(run_smc_substitution(
            sub, obs, num_particles=num_particles, max_intended=bucket, max_dist=max_dist,
            max_deletions=max_deletions, allow_insertion=allow_insertion, **kw))
    return results
