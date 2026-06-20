"""Mathematical-correctness gate for the RB-SMC pair-HMM noisy-channel model.

The PoCs only ever compared caprop vs bootstrap; neither checked the SMC against ground truth.
This does. For a SHORT observed sentence over the toy vocab it BRUTE-FORCES the exact posterior over
intended sentences -- enumerating every intended sentence up to a max length, scoring

    joint(intended) = LM_prior(intended) + channel_loglik(observed | intended)

where the channel marginalizes the edit-alignment with the SAME word-level forward DP the filter
uses (terminal read alpha[M]). The exact posterior is softmax(joint); the exact log-marginal is
logsumexp(joint). We then assert the SMC reproduces both:

  * SMC logZ        == exact log-marginal  (up to the a0 leading-spurious constant, deterministic)
  * SMC MAP         == exact MAP            (substitution + spurious-word cases)
  * SMC posterior   ~= exact posterior      (small total-variation distance at large P)

and (A3) that with the production band the filter recovers all four edit types behaviourally
(``test_edit_types_recovered_with_band``) -- the evidence that insertions need no explicit INSERT
action: the channel sweep marginalizes them and the band gives them reach.

caprop's lower-variance-than-bootstrap logZ is the fully-adapted proposal's signature, but at TOY
scale the edge is small and case/seed-sensitive (plan finding #4: it grows with LM cost), so it is
NOT a pass/fail gate -- it is reported by ``main()`` as a diagnostic only.

Enumeration is VECTORIZED (one jitted batched DP per length) and kept small (short sentence, capped
length) so it runs in seconds. The harder missing-word case (intended length 6, too big to
enumerate over V=12) is covered by the behavioural band gate above, not by exact enumeration.

Run as a script:  python -m genjax_port.tests.test_pairhmm_exact
Run as a test:    pytest src/genjax_port/tests/test_pairhmm_exact.py
"""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from genjax_port.tests.toy_channel import channel_logpdf, channel_form_logpdf, encode
from genjax_port.tests.toy_vocab import V, VOCAB, VOCAB_IDS, VOCAB_LEN, WORD2IDX
from genjax_port.tests.toy_bigram import BOS, EOS, LOG_BIGRAM, lm_logits
from genjax_port.word_dp import _word_row_update, channel_carry
from genjax_port.noise_word import _damerau_levenshtein
from genjax_port.tests import toy_caprop as caprop
from genjax_port import pairhmm_smc
from genjax_port import pairhmm_rejuv as rejuv

WDEL = float(jnp.log(0.1))
WINS = float(jnp.log(0.05))


def _toy_model(lm_fn):
    """A :class:`pairhmm_smc.PairHMMModel` over the toy vocab, LM injected. The toy bigram is the
    same filter as Pythia with a different LM, so certifying it here certifies the shared code."""
    def candidate_words(word, obs_span, max_dist, Ke):   # obs_span unused (toy COPY is faithful already)
        cands = sorted((_damerau_levenshtein(word, w, max_dist), WORD2IDX[w]) for w in VOCAB)
        return [((i,), VOCAB[i]) for d, i in cands if d <= max_dist][:Ke]  # single-token toy words

    return pairhmm_smc.PairHMMModel(
        lm_fn=jax.vmap(lm_fn),                 # batch the single-particle toy LM over the cloud
        eos_id=EOS, emit_vocab=V,
        vocab_char=VOCAB_IDS, vocab_clen=VOCAB_LEN, channel_logpdf=channel_logpdf,
        char_ids=encode, candidate_words=candidate_words, obs_words=str.split,
        decode_ids=lambda t: " ".join(VOCAB[i] for i in t), seed_ids=(),
        channel_form=channel_form_logpdf)      # base-rate-decoupled FORM channel (word-action / ON path)


# --------------------------------------------------------------------------------------------------
# Exact posterior over intended sentences -- vectorized brute force (one batched DP per length).
# --------------------------------------------------------------------------------------------------
def _emit_table(observed):
    """EMIT[k, w] = char-level channel logpdf of observed word k given intended word w. (M, V)."""
    obs_ids = jnp.stack([encode(w)[0] for w in observed.split()])
    return jax.vmap(jax.vmap(channel_logpdf, in_axes=(None, 0, 0)),
                    in_axes=(0, None, None))(obs_ids, VOCAB_IDS, VOCAB_LEN)  # (M, V)


def _joint_batch(seqs, n, emit, log_bigram, M, a0):
    """joint logp for a batch of intended sequences of fixed length n. seqs: (N, n) int. -> (N,)."""
    if n == 0:
        lm = jnp.full((seqs.shape[0],), log_bigram[BOS, EOS])
        chan = jnp.full((seqs.shape[0],), a0[M])  # all observed words spurious
        return lm + chan

    bos = jnp.full((seqs.shape[0], 1), BOS, seqs.dtype)
    eos = jnp.full((seqs.shape[0], 1), EOS, seqs.dtype)
    frm = jnp.concatenate([bos, seqs], axis=1)            # (N, n+1)
    to = jnp.concatenate([seqs, eos], axis=1)             # (N, n+1)
    lm = jnp.sum(log_bigram[frm, to], axis=1)             # (N,)

    def chan_one(seq):
        alpha = a0
        for i in range(n):                                # n is a Python int -> unrolled
            alpha = _word_row_update(alpha, emit[:, seq[i]], WDEL, WINS)
        return alpha[M]

    chan = jax.vmap(chan_one)(seqs)
    return lm + chan


def exact_posterior(observed, log_bigram=LOG_BIGRAM, Lmax=None):
    """Returns (posterior dict {sentence: prob}, exact log-marginal). Vectorized; small inputs only."""
    M = len(observed.split())
    Lmax = (M + 1) if Lmax is None else Lmax
    emit = _emit_table(observed)
    a0 = jnp.where(jnp.arange(M + 1) == 0, 0.0, jnp.arange(M + 1) * WINS)

    # Run EAGERLY (no jit): for these small enumerations jit's XLA constant-folding of the big
    # sequence array is far slower than plain op dispatch, and times out for longer sentences.
    sents, joints = [], []
    for n in range(Lmax + 1):
        seqs = jnp.array(list(itertools.product(range(V), repeat=n)),
                         jnp.int32).reshape(-1, n) if n else jnp.zeros((1, 0), jnp.int32)
        j = _joint_batch(seqs, n, emit, log_bigram, M, a0)
        sents.extend(" ".join(VOCAB[int(i)] for i in s) for s in seqs)
        joints.append(j)

    joints = jnp.concatenate(joints)
    logZ = float(logsumexp(joints))
    post = jax.nn.softmax(joints)
    words = {}
    for s, p in zip(sents, post):                         # collapse duplicate-key sentences
        words[s] = words.get(s, 0.0) + float(p)
    return words, logZ


# --------------------------------------------------------------------------------------------------
# SMC posterior (decode the caprop / bootstrap filter into the same {sentence: prob} form).
# --------------------------------------------------------------------------------------------------
def smc_posterior(observed, key, P=8000, proposal="caprop", lm_fn=lm_logits, band=None):
    model = _toy_model(lm_fn)
    st, dw, logZ, sl = pairhmm_smc.run(observed, key, model, P=P, proposal=proposal,
                                       wdel=WDEL, wins=WINS, band=band)
    return {s: p for s, p in pairhmm_smc.decode(st, dw, model, top=50)}, logZ


def tv_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _a0_const(M):
    a0 = jnp.where(jnp.arange(M + 1) == 0, 0.0, jnp.arange(M + 1) * WINS)
    return float(logsumexp(a0))  # the filter's logZ is measured relative to this initial mass


def _peaked():
    return caprop.make_bigram([("the cat sat on the mat", 50), ("the dog ran", 1),
                               ("a big cat slept", 1), ("the small dog sat", 1),
                               ("a dog sat on the hat", 1), ("the big dog slept", 1)])


# --------------------------------------------------------------------------------------------------
# Tests (small, exact, fast). Correctness is checked on NON-DEGENERATE cases: a peaked-LM
# substitution ('teh cat sat' -> 'the cat sat') whose exact posterior is sharply concentrated on a
# sensible sentence, and a flat-bigram spurious-word case ('the cat cat' -> 'the cat'). The flat
# 'teh cat' case is deliberately NOT used as a gate: under a too-weak bigram its exact MAP is the
# empty sentence, so it tests nothing linguistic (it only confirmed the logZ machinery).
#
# logZ note: the comparison uses the CAPROP proposal. Its candidate set captures the dominant
# posterior mass, so it targets ~the same short-sentence support the enumeration covers, giving a
# tight logZ match. (Bootstrap explores the full LM-prior length range up to M+slack, beyond the
# enumerable Lmax, so its logZ legitimately exceeds the truncated-exact value -- a support artifact,
# not an error. We therefore gate logZ on caprop; bootstrap is only the variance contrast in main.)
_PEAKED_OBS, _PEAKED_LMAX, _PEAKED_TRUTH = "teh cat sat", 4, "the cat sat"


@functools.lru_cache(maxsize=1)
def _peaked_exact():
    lm = _peaked()
    exact, logZ = exact_posterior(_PEAKED_OBS, log_bigram=_bigram_table(lm), Lmax=_PEAKED_LMAX)
    return lm, exact, logZ


def test_caprop_logZ_matches_exact():
    """caprop logZ reproduces the exact log-marginal (up to the deterministic a0 constant)."""
    lm, _, exact_logZ = _peaked_exact()
    target = exact_logZ - _a0_const(len(_PEAKED_OBS.split()))
    zs = jnp.array([smc_posterior(_PEAKED_OBS, jax.random.PRNGKey(s), P=6000, lm_fn=lm)[1]
                    for s in range(4)])
    assert abs(float(zs.mean()) - target) < 0.08, \
        f"caprop logZ {float(zs.mean()):.3f} != exact-rel {target:.3f}"


def test_map_matches_exact():
    """SMC MAP equals the exact MAP -- substitution (peaked) and spurious-word (flat) cases."""
    lm, exact, _ = _peaked_exact()
    smc, _ = smc_posterior(_PEAKED_OBS, jax.random.PRNGKey(0), lm_fn=lm)
    assert max(smc, key=smc.get) == max(exact, key=exact.get) == _PEAKED_TRUTH

    ex2, _ = exact_posterior("the cat cat", Lmax=3)
    smc2, _ = smc_posterior("the cat cat", jax.random.PRNGKey(0))
    assert max(smc2, key=smc2.get) == max(ex2, key=ex2.get) == "the cat"


def test_posterior_mass_matches_exact():
    """The inferred posterior reproduces the exact one: MAP mass close + small overall TV."""
    lm, exact, _ = _peaked_exact()
    smc, _ = smc_posterior(_PEAKED_OBS, jax.random.PRNGKey(0), lm_fn=lm)
    map_s = max(exact, key=exact.get)
    assert abs(exact[map_s] - smc.get(map_s, 0.0)) < 0.12, "MAP probability mass off"
    assert tv_distance(exact, smc) < 0.15, "posterior shape too far from exact"


# A3 edit-type recovery gates. The peaked toy LM (heavily favouring 'the cat sat on the mat') makes
# each correction the genuine MAP, so these are behavioural MAP-recovery checks rather than exact-
# enumeration ones (the missing case needs intended length 6 -- too big to enumerate over V=12).
_EDIT_CASES = [
    ("substitution",  "teh cat sat",                "the cat sat"),             # SUB corrected
    ("spurious",      "the cat sat sat on the mat", "the cat sat on the mat"),  # INSERTION -> shorter
    ("missing",       "the cat sat the mat",        "the cat sat on the mat"),  # DELETION  -> longer
    ("clean",         "the cat sat on the mat",     "the cat sat on the mat"),  # KEEP      -> unchanged
]


def test_edit_types_recovered_with_band():
    """A3: with the PRODUCTION band (=2, what Pythia uses) and NO explicit INSERT action, the swept
    filter recovers every edit type -- substitution, spurious->shorter, missing->longer, clean->
    unchanged. This is the evidence the design rests on: a spurious word needs no INSERT 'action'.
    The channel sweep in ``_word_row_update`` marginalizes the insertion (cost WINS) and the band
    (|k-t|<=2) gives consumption the room to run ahead of emission. Making INSERT a peer LM-action
    instead biases logZ +0.2 nats (a channel event injected into the LM's action distribution).

    Deletion case is 'the cat sat the mat' (NOT 'the cat on the mat', a valid noun phrase the model
    has no reason to lengthen): under the peaked LM the dropped 'on' makes 'sat the' disfluent, so
    restoring 'on' is the true MAP (plan A3). The clean case guards the opposite failure -- the
    filter must NOT shave a real word off by calling it spurious."""
    lm = _peaked()
    for tag, obs, truth in _EDIT_CASES:
        smc, _ = smc_posterior(obs, jax.random.PRNGKey(0), P=8000, lm_fn=lm, band=2)
        got = max(smc, key=smc.get)
        assert got == truth, f"{tag}: {obs!r} -> MAP {got!r}, expected {truth!r}"


# --------------------------------------------------------------------------------------------------
# R0 -- Gibbs rejuvenation correctness (plan REJUV_KV_REDESIGN_PLAN.md). The move (pairhmm_rejuv) is
# certified on the toy by the SAME exact enumeration: a full-conditional Gibbs sweep must (1) leave
# the exact posterior INVARIANT when applied to a cloud already at the posterior, and (2) RECOVER the
# exact MAP from a deliberately COLLAPSED cloud (the impoverishment cure -- the toy analogue of the
# 'cat/mat' -> 'car' collapse diagnosed in planning/kv_cache_spikes/). Toy = T_max=1, band=None to
# match the enumeration (no band). The candidate set is the WHOLE vocab -> a true Gibbs step.
# --------------------------------------------------------------------------------------------------
def _rejuv_ctx_and_cands(observed, lm):
    ctx = rejuv.make_rejuv_ctx(observed, _toy_model(lm), WDEL, WINS, band=None)
    cand_table = jnp.broadcast_to(jnp.arange(V)[None, :], (ctx.Wmax, V))  # full-vocab Gibbs
    pool_tok, pool_len = rejuv.pool_from_table(cand_table)
    return ctx, pool_tok, pool_len


def _cloud_from_dist(dist, P, key, Wmax):
    """Equally-weighted cloud of P particles drawn from a {sentence: prob} dict."""
    sents = list(dist)
    LCTX = Wmax + 1
    rows = [[WORD2IDX[w] for w in s.split()] for s in sents]
    buf0 = jnp.array([r + [EOS] * (LCTX - len(r)) for r in rows], jnp.int32)
    len0 = jnp.array([len(r) for r in rows], jnp.int32)
    idx = jax.random.categorical(key, jnp.log(jnp.array([dist[s] for s in sents])), shape=(P,))
    return buf0[idx], len0[idx]


def _cloud_from_sentence(sentence, P, Wmax):
    toks = [WORD2IDX[w] for w in sentence.split()]
    LCTX = Wmax + 1
    buf = jnp.full((P, LCTX), EOS, jnp.int32).at[:, :len(toks)].set(jnp.array(toks, jnp.int32))
    return buf, jnp.full((P,), len(toks), jnp.int32)


def test_rejuv_leaves_exact_posterior_invariant():
    """A full-conditional Gibbs sweep on a cloud already at the exact posterior leaves it there
    (invariance) -- the proof the move does not corrupt the certified posterior."""
    lm, exact, _ = _peaked_exact()
    ctx, pool_tok, pool_len = _rejuv_ctx_and_cands(_PEAKED_OBS, lm)
    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key)
    buf, clen = _cloud_from_dist(exact, 6000, sub, ctx.Wmax)
    for _ in range(3):
        key, sub = jax.random.split(key)
        buf, _, _ = rejuv.gibbs_sweep(sub, buf, clen, ctx, pool_tok, pool_len)
    after = rejuv.decode_counts(buf, clen, ctx.model, ctx.seed_len)
    assert max(after, key=after.get) == _PEAKED_TRUTH, "rejuv moved the MAP off the exact MAP"
    assert tv_distance(after, exact) < 0.08, \
        f"rejuv perturbed the posterior: TV {tv_distance(after, exact):.3f}"


def test_rejuv_recovers_collapsed_cloud():
    """From a cloud COLLAPSED onto a wrong (same-length) reading, Gibbs sweeps recover the exact MAP
    -- the impoverishment cure (word-identity Gibbs mixes within the length the cloud is stuck at)."""
    lm, exact, _ = _peaked_exact()
    ctx, pool_tok, pool_len = _rejuv_ctx_and_cands(_PEAKED_OBS, lm)
    buf, clen = _cloud_from_sentence("the dog sat", 4000, ctx.Wmax)  # wrong middle word, right length
    key = jax.random.PRNGKey(0)
    for _ in range(6):
        key, sub = jax.random.split(key)
        buf, _, _ = rejuv.gibbs_sweep(sub, buf, clen, ctx, pool_tok, pool_len)
    rec = rejuv.decode_counts(buf, clen, ctx.model, ctx.seed_len)
    assert max(rec, key=rec.get) == _PEAKED_TRUTH, \
        f"rejuv failed to escape the collapse: MAP {max(rec, key=rec.get)!r}"
    assert rec[_PEAKED_TRUTH] > 0.5, f"weak recovery: p={rec.get(_PEAKED_TRUTH, 0.0):.2f}"


def test_rejuv_smcp3_weight_zero():
    """R1: the genjax SMCP3 reweight of a full-conditional (symmetric, full-support) Gibbs move is 0
    -- the built-it-right check that ``Rejuvenate``'s ``w + bwd − fwd`` cancels for the exact
    conditional (REJUV_GOAL2/3). genjax produces the weight; we never hand-derive the ratio. A
    non-zero weight here would mean the proposal/target/model density are inconsistent."""
    lm, exact, _ = _peaked_exact()
    ctx, pool_tok, pool_len = _rejuv_ctx_and_cands(_PEAKED_OBS, lm)
    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key)
    buf, clen = _cloud_from_dist(exact, 2000, sub, ctx.Wmax)
    _, _, move_logw = rejuv.gibbs_sweep(sub, buf, clen, ctx, pool_tok, pool_len)
    assert float(jnp.max(jnp.abs(move_logw))) < 1e-3, \
        f"full-conditional SMCP3 weight not ~0: max|w|={float(jnp.max(jnp.abs(move_logw))):.2e}"


# ==================================================================================================
# Phase 2 (WORD_ACTION_REJUV_PLAN) -- word-action ON rejuvenation gates. The single-token rejuv gates
# above run the CHAR-COPY channel (``action_alpha=None``); these certify the SAME sweep on the
# WORD-ACTION channel, where each emission column pays a per-word action cost -- ``log p_copy`` for a
# (case-insensitive) verbatim copy, ``log p_sub`` for a substitution -- on top of the base-rate-
# DECOUPLED FORM channel, with insertion/deletion costs from ``theta``. At the CONCENTRATED-alpha limit
# a ``Dirichlet(c*theta)`` collapses to a deterministic ``theta``, so the per-particle costs are a fixed
# operating point and the exact word-action posterior is enumerable -- the same brute force as
# ``exact_posterior`` with the FORM emission + action offset, scored by the SHARED ``channel_carry`` DP
# the filter and sweep both use. The theta-aware sweep (``rejuv.sweep(theta_costs=...)``) must (1) leave
# that posterior invariant, (2) recover it from a collapsed cloud, and (3) carry a ~0 SMCP3 weight for
# the full-conditional move -- the word-action mirrors of the three char-copy rejuv gates above. This is
# the behaviour the boolean fork disabled (the sweep was gated off whenever the word-action channel was
# active because its scorer could not see ``theta``); these gates pin the theta-aware sweep before the
# filter is switched to call it.
# ==================================================================================================

# A copy-dominated operating point = the concentrated-alpha limit (theta == mean of Dirichlet(c*theta),
# c -> inf): copies cheap, substitution/insertion/deletion expensive -- the regime a clean channel sits
# in, where a typo's single substitution still beats dropping or duplicating a word.
_WA_THETA = jnp.array([0.85, 0.05, 0.05, 0.05], jnp.float32)   # (copy, sub, insert, delete)


def _wa_emit_copymask_costs(observed, theta, lm):
    """The word-action channel pieces for ``observed`` at a FIXED ``theta``, built EXACTLY as
    ``pairhmm_smc.run`` does when ``action_alpha`` is set (and at its concentrated limit): the FORM
    emission table ``emit_aug`` (M, Vc), the case-insensitive ``copy_mask`` (M, Vc), and the scalar
    per-particle action costs ``(lp_copy, lp_sub, wdel, wins)`` from ``_theta_to_costs``. Returns
    ``(model, emit_aug, copy_mask, costs)``."""
    model = _toy_model(lm)
    obs_words = observed.split(); M = len(obs_words)
    obs_char = jnp.stack([encode(w)[0] for w in obs_words])           # (M, Lc)
    emit_form = jax.vmap(jax.vmap(channel_form_logpdf, in_axes=(None, 0, 0)),
                         in_axes=(0, None, None))(obs_char, model.vocab_char, model.vocab_clen)  # (M, V)
    *_, emit_aug, copy_mask, _T, _nmt = pairhmm_smc._build_candidates(
        model, obs_words, [None] * M, obs_char, emit_form, max_dist=2, Ke=6, channel_fn=channel_form_logpdf)
    wins_vec = jnp.broadcast_to(jnp.float32(WINS), (M,))
    lp_copy, lp_sub, wdel_p, wins_p = pairhmm_smc._theta_to_costs(theta[None], True, wins_vec)
    costs = (float(lp_copy[0]), float(lp_sub[0]), float(wdel_p[0]), float(wins_p[0, 0]))
    return model, emit_aug, copy_mask.astype(jnp.float32), costs


def _wa_exact_posterior(observed, theta, lm, Lmax):
    """Exact posterior over intended sentences under the WORD-ACTION channel at fixed ``theta`` -- the
    same vectorized brute force as :func:`exact_posterior`, but with the FORM emission + per-word action
    offset + theta-derived indel costs, scored by the shared :func:`channel_carry` DP. Using the same DP
    the sweep uses makes the ground truth and the sweep score the IDENTICAL channel (the comparison is in
    the enumeration vs sampling, exactly as the char-copy gates)."""
    model, emit_aug, copy_mask, (lp_copy, lp_sub, wdel, wins) = _wa_emit_copymask_costs(observed, theta, lm)
    M = len(observed.split())
    log_bigram = _bigram_table(lm)
    ks = jnp.arange(M + 1)
    a0 = jnp.where(ks == 0, 0.0, ks * wins)                                # leading-spurious init

    def chan_batch(seqs, n):                                               # (N, n) -> (N,) channel logp
        N = seqs.shape[0]
        if n == 0:
            return jnp.full((N,), float(a0[M]))
        carry = channel_carry(jnp.broadcast_to(a0, (N, M + 1)), emit_aug, None, M,
                              seqs.astype(jnp.int32), jnp.ones((N, n), jnp.int32),
                              jnp.full(N, lp_copy), jnp.full(N, lp_sub), jnp.full(N, wdel),
                              jnp.broadcast_to(jnp.float32(wins), (N, M)), copy_mask)
        return carry[:, M]

    sents, joints = [], []
    for n in range(Lmax + 1):
        seqs = (jnp.array(list(itertools.product(range(V), repeat=n)), jnp.int32).reshape(-1, n)
                if n else jnp.zeros((1, 0), jnp.int32))
        if n == 0:
            lm_lp = jnp.full((1,), float(log_bigram[BOS, EOS]))
        else:
            frm = jnp.concatenate([jnp.full((seqs.shape[0], 1), BOS), seqs], axis=1)
            to = jnp.concatenate([seqs, jnp.full((seqs.shape[0], 1), EOS)], axis=1)
            lm_lp = jnp.sum(log_bigram[frm, to], axis=1)
        sents.extend(" ".join(VOCAB[int(i)] for i in s) for s in seqs)
        joints.append(lm_lp + chan_batch(seqs, n))
    joints = jnp.concatenate(joints)
    post = jax.nn.softmax(joints)
    words = {}
    for s, p in zip(sents, post):
        words[s] = words.get(s, 0.0) + float(p)
    return words, float(logsumexp(joints))


def _wa_ctx_pool_costs(observed, theta, lm, P, slack=3):
    """``RejuvCtx`` (FORM ``emit_full`` + full-vocab Gibbs pool) + the per-particle ``theta_costs`` tuple
    the word-action sweep takes, at the concentrated-alpha fixed ``theta`` -- mirroring how
    ``pairhmm_smc.run`` wires the sweep when ``action_alpha`` is set (the full vocab -> a true Gibbs
    step, matching the char-copy rejuv gates)."""
    model, emit_aug, copy_mask, (lp_copy, lp_sub, wdel, wins) = _wa_emit_copymask_costs(observed, theta, lm)
    M = len(observed.split()); Wmax = M + slack
    ks = jnp.arange(M + 1)
    a0 = jnp.where(ks == 0, 0.0, ks * wins)
    ctx = rejuv.RejuvCtx(model, emit_aug, a0, M, 0, Wmax, wdel, wins, None, t_max=1, lm_temp=1.0)
    cand_table = jnp.broadcast_to(jnp.arange(V)[None, :], (Wmax, V))       # full-vocab Gibbs
    pool_tok, pool_len = rejuv.pool_from_table(cand_table)
    theta_costs = (jnp.full(P, lp_copy), jnp.full(P, lp_sub), jnp.full(P, wdel),
                   jnp.broadcast_to(jnp.float32(wins), (P, M)), jnp.broadcast_to(a0, (P, M + 1)), copy_mask)
    return ctx, pool_tok, pool_len, theta_costs


def test_wa_rejuv_leaves_exact_posterior_invariant():
    """Phase 2: the theta-aware sweep on a cloud already at the WORD-ACTION exact posterior leaves it
    there (invariance) -- the word-action analog of ``test_rejuv_leaves_exact_posterior_invariant``,
    proving the channel-ON move does not corrupt the certified word-action posterior."""
    lm, P = _peaked(), 6000
    exact, _ = _wa_exact_posterior(_PEAKED_OBS, _WA_THETA, lm, _PEAKED_LMAX)
    ctx, pool_tok, pool_len, theta_costs = _wa_ctx_pool_costs(_PEAKED_OBS, _WA_THETA, lm, P)
    swp = rejuv.make_sweep(ctx, pool_tok, pool_len)
    key, sub = jax.random.split(jax.random.PRNGKey(0))
    buf, clen = _cloud_from_dist(exact, P, sub, ctx.Wmax)
    for _ in range(3):
        key, sub = jax.random.split(key)
        buf, clen, _wl, _ws, _la, _mlw = swp(sub, buf, clen, theta_costs=theta_costs)
    after = rejuv.decode_counts(buf, clen, ctx.model, ctx.seed_len)
    assert max(after, key=after.get) == max(exact, key=exact.get), \
        f"WA rejuv moved the MAP: {max(after, key=after.get)!r} vs exact {max(exact, key=exact.get)!r}"
    assert tv_distance(after, exact) < 0.08, f"WA rejuv perturbed the posterior: TV {tv_distance(after, exact):.3f}"


def test_wa_rejuv_recovers_collapsed_cloud():
    """Phase 2: from a cloud COLLAPSED onto a wrong same-length reading ('the dog sat'), theta-aware
    sweeps recover the word-action exact MAP -- the impoverishment cure on the ON channel, the concrete
    behaviour the boolean fork disabled. The channel must pull the wrong middle word ('dog') back to the
    cheap COPY of the observed 'cat'."""
    lm, P = _peaked(), 4000
    exact, _ = _wa_exact_posterior(_PEAKED_OBS, _WA_THETA, lm, _PEAKED_LMAX)
    truth = max(exact, key=exact.get)
    ctx, pool_tok, pool_len, theta_costs = _wa_ctx_pool_costs(_PEAKED_OBS, _WA_THETA, lm, P)
    swp = rejuv.make_sweep(ctx, pool_tok, pool_len)
    buf, clen = _cloud_from_sentence("the dog sat", P, ctx.Wmax)          # wrong middle word, right length
    key = jax.random.PRNGKey(0)
    for _ in range(6):
        key, sub = jax.random.split(key)
        buf, clen, _wl, _ws, _la, _mlw = swp(sub, buf, clen, theta_costs=theta_costs)
    rec = rejuv.decode_counts(buf, clen, ctx.model, ctx.seed_len)
    assert max(rec, key=rec.get) == truth, f"WA rejuv failed to escape the collapse: MAP {max(rec, key=rec.get)!r}"
    assert rec[truth] > 0.5, f"weak WA recovery: p={rec.get(truth, 0.0):.2f}"


def test_wa_rejuv_smcp3_weight_zero():
    """Phase 2: the genjax SMCP3 reweight of the theta-aware full-conditional move is ~0 -- the
    built-it-right check that threading the per-particle ``theta`` costs into ``channel_carry`` keeps the
    sweep's proposal the EXACT conditional (a non-zero weight would mean proposal/target/model drifted)."""
    lm, P = _peaked(), 2000
    exact, _ = _wa_exact_posterior(_PEAKED_OBS, _WA_THETA, lm, _PEAKED_LMAX)
    ctx, pool_tok, pool_len, theta_costs = _wa_ctx_pool_costs(_PEAKED_OBS, _WA_THETA, lm, P)
    swp = rejuv.make_sweep(ctx, pool_tok, pool_len)
    key, sub = jax.random.split(jax.random.PRNGKey(0))
    buf, clen = _cloud_from_dist(exact, P, sub, ctx.Wmax)
    *_, mlw = swp(sub, buf, clen, theta_costs=theta_costs)
    assert float(jnp.max(jnp.abs(mlw))) < 1e-3, \
        f"WA full-conditional SMCP3 weight not ~0: max|w|={float(jnp.max(jnp.abs(mlw))):.2e}"


def test_wa_run_gibbs_end_to_end():
    """Phase 2 INTEGRATION: the full filter with the word-action channel (``action_alpha`` set) AND
    ``rejuv='gibbs'`` runs the theta-aware SWEEP-THEN-REFRESH inside ``run`` end-to-end and recovers the
    MAP -- the path the boolean fork disabled (ON used to force theta-refresh-only because the sweep
    could not see theta). Concentrated ``action_alpha`` (~the _WA_THETA copy-dominated operating point);
    'teh cat sat' -> 'the cat sat'. Exercises the wiring the direct-sweep gates above cannot: the
    per-particle ``theta_costs`` assembled in ``run`` from the gathered theta, then the conjugate refresh
    on the swept parse."""
    lm = _peaked()
    model = _toy_model(lm)
    st, dw, _logZ, _sl = pairhmm_smc.run(_PEAKED_OBS, jax.random.PRNGKey(0), model, P=4000,
                                         proposal="caprop", wdel=WDEL, wins=WINS, band=2,
                                         rejuv="gibbs", action_alpha=[8.5, 0.5, 0.5, 0.5])
    smc = {s: p for s, p in pairhmm_smc.decode(st, dw, model, top=50)}
    assert max(smc, key=smc.get) == _PEAKED_TRUTH, \
        f"WA run+gibbs MAP {max(smc, key=smc.get)!r}, expected {_PEAKED_TRUTH!r}"


def test_dedup_forward_exact():
    """R3 item 1: ``cache_dedup.make_forward_dedup`` runs a batched forward on only the unique filled
    prefixes (keyed on ``buf[:i_len]``) and scatters back. EXACT -- the deduped output equals the raw
    per-row forward -- and on a degenerate batch the rows actually computed drop below rows-in. This is
    the bit-parity guardrail for the filter-forward dedup (R3 item 1): because the same logits are
    scattered to duplicate rows and each is sampled with its own RNG key downstream, the SMC posterior
    is bit-identical given the same RNG; only redundant LM forwards are removed. LM-free (toy forward)."""
    from genjax_port import cache_dedup

    def fwd(bufs, ilens):                        # deterministic in the FILLED PREFIX only (causal)
        M = bufs.shape[1]
        idx = jnp.arange(M)[None, :]
        pre = jnp.where(idx < ilens[:, None], bufs, 0)
        return jnp.stack([jnp.sum(pre, 1), jnp.sum(pre * idx, 1),
                          ilens, jnp.sum(pre * pre, 1)], axis=1).astype(jnp.float32)

    uniq = jnp.array([[5, 3, 7, 0, 0], [2, 9, 0, 0, 0], [2, 9, 4, 1, 0],
                      [8, 0, 0, 0, 0], [5, 3, 1, 0, 0]], jnp.int32)
    ulen = jnp.array([3, 2, 4, 1, 3], jnp.int32)
    sel = jnp.array([0, 1, 2, 3, 4] * 8)         # 40 rows, 5 unique (a post-resample degenerate cloud)
    bufs, ilens = uniq[sel], ulen[sel]

    stats = cache_dedup.DedupStats()
    out_dedup = cache_dedup.make_forward_dedup(fwd, stats)(bufs, ilens)
    out_raw = fwd(bufs, ilens)
    assert jnp.array_equal(out_dedup, out_raw), "deduped forward != raw forward (scatter is wrong)"
    assert stats.rows_in == 40 and stats.rows_computed < 40, f"dedup did not cut rows: {stats!r}"


def test_rejuv_dedup_bit_parity():
    """R3 item 1b: the sweep-tail dedup (run ``tail_fn`` on the unique buffers, scatter [P,Kt] back) is
    EXACT -- a dedup=on sweep equals dedup=off WORD-FOR-WORD and weight-for-weight given the same RNG. It
    dedups only the deterministic tail/channel scores; the per-particle SMCP3 sample is untouched, so
    duplicate particles still diverge. The degenerate cloud (sampled from the peaked posterior -> many
    duplicate buffers) is exactly where dedup fires."""
    from genjax_port import cache_dedup
    lm, exact, _ = _peaked_exact()
    ctx, pool_tok, pool_len = _rejuv_ctx_and_cands(_PEAKED_OBS, lm)
    seed, sub = jax.random.split(jax.random.PRNGKey(0))
    buf, clen = _cloud_from_dist(exact, 256, sub, ctx.Wmax)         # degenerate cloud (duplicate buffers)
    swp_off = rejuv.make_sweep(ctx, pool_tok, pool_len, dedup=False)
    swp_on = rejuv.make_sweep(ctx, pool_tok, pool_len, dedup=True)
    rk = jax.random.PRNGKey(7)
    b0, _cl0, _wl0, _ws0, la0, mlw0 = swp_off(rk, buf, clen)
    stats = cache_dedup.DedupStats()
    b1, _cl1, _wl1, _ws1, la1, mlw1 = swp_on(rk, buf, clen, dedup_stats=stats)
    assert jnp.array_equal(b0, b1), "dedup changed the swept buffers (not bit-exact)"
    assert float(jnp.max(jnp.abs(mlw0 - mlw1))) < 1e-4, "dedup changed move_logw"
    assert float(jnp.max(jnp.abs(la0 - la1))) < 1e-4, "dedup changed log_alpha"
    assert stats.rows_in > 0 and stats.rows_computed < stats.rows_in, f"dedup did not fire: {stats!r}"


# --------------------------------------------------------------------------------------------------
# Phase D (D0) -- MULTI-TOKEN intended words. The single-token gates above can't exercise multi-token
# words (every toy word is one LM token). This block is a minimal toy where some intended WORDS span
# >= 2 LM "sub-tokens" (kitten = kit+ten, kitchen = kit+chen), while the CHANNEL still scores the whole
# -word SURFACE. It exercises exactly the Phase-D machinery the single-token path can't: the chain-rule
# LM over a candidate's token span, the surface_id-indexed channel column, and the span splice. The
# same exact-enumeration method certifies it: the sub-token bigram collapses into an effective WORD
# bigram (the bigram only sees the previous word's LAST sub-token), so a word's internal sub-token
# chain-rule folds into one word->word transition and enumeration over WORD sequences is exact.
# --------------------------------------------------------------------------------------------------
# Sub-token surfaces (GPT-NeoX leading-space convention: word-initial pieces carry a leading space,
# continuations do not -- so decode by concatenation recovers the spacing, as Pythia's tokenizer does).
_MT_WORD_PIECES = {"the": [" the"], "cat": [" cat"], "sat": [" sat"], "mat": [" mat"], "dog": [" dog"],
                   "kitten": [" kit", "ten"], "kitchen": [" kit", "chen"]}
MT_VOCAB = list(_MT_WORD_PIECES)
MT_SUBTOK = []                                    # distinct sub-token surfaces, order = id
for _w, _ps in _MT_WORD_PIECES.items():
    for _p in _ps:
        if _p not in MT_SUBTOK:
            MT_SUBTOK.append(_p)
MT_NSUB = len(MT_SUBTOK)
MT_EOS = MT_NSUB                                  # EOS / BOS row id in the sub-token bigram
_MT_SUB2IDX = {s: i for i, s in enumerate(MT_SUBTOK)}
MT_SPAN = {w: tuple(_MT_SUB2IDX[p] for p in ps) for w, ps in _MT_WORD_PIECES.items()}  # word->subtok ids
# A sub-token is a "whole word" (eligible as a top-J LM bridge) iff it is some SINGLE-token word's piece.
_MT_WORD_SET = {ps[0] for w, ps in _MT_WORD_PIECES.items() if len(ps) == 1}
MT_WORD_MASK = jnp.array([s in _MT_WORD_SET for s in MT_SUBTOK], bool)
MT_SUB_IDS = jnp.stack([jnp.asarray(encode(s.strip())[0]) for s in MT_SUBTOK])   # (NSUB, Lchar)
MT_SUB_LEN = jnp.asarray([encode(s.strip())[1] for s in MT_SUBTOK])


def _mt_sub_log_bigram(weighted_sents):
    """Peaked sub-token bigram (rows: prev sub-token, BOS=MT_NSUB; cols: next, EOS=MT_NSUB), built
    from weighted word-sentences expanded to their sub-token spans -- so P(kitten|the) is the product
    P(kit|the)*P(ten|kit), the genuine multi-token chain-rule."""
    counts = jnp.ones((MT_NSUB + 1, MT_NSUB + 1))
    for sent, wt in weighted_sents:
        prev = MT_EOS                              # BOS uses the same row id
        for word in sent.split():
            for sub in MT_SPAN[word]:
                counts = counts.at[prev, sub].add(float(wt)); prev = sub
        counts = counts.at[prev, MT_EOS].add(float(wt))
    return jnp.log(counts / counts.sum(axis=1, keepdims=True))


def _mt_lm_fn(sub_log_bigram):
    def lm(ctx_buf, ctx_len):                      # ctx_buf holds SUB-TOKEN ids; bigram on the last one
        prev = jnp.where(ctx_len > 0, ctx_buf[jnp.maximum(ctx_len - 1, 0)], MT_EOS)
        return sub_log_bigram[prev]
    return lm


def _mt_decode(ids):
    return "".join(MT_SUBTOK[int(i)] for i in ids).strip()


def _mt_model(sub_log_bigram):
    def candidate_words(word, obs_span, max_dist, Ke):   # MT_VOCAB words within edit distance; COPY = d=0
        cands = sorted((_damerau_levenshtein(word, w, max_dist), w) for w in MT_VOCAB)
        return [(MT_SPAN[w], w) for d, w in cands if d <= max_dist][:Ke]

    return pairhmm_smc.PairHMMModel(
        lm_fn=jax.vmap(_mt_lm_fn(sub_log_bigram)), eos_id=MT_EOS, emit_vocab=MT_NSUB,
        vocab_char=MT_SUB_IDS, vocab_clen=MT_SUB_LEN, channel_logpdf=channel_logpdf,
        char_ids=encode, candidate_words=candidate_words, obs_words=str.split,
        decode_ids=_mt_decode, seed_ids=(), word_mask=MT_WORD_MASK)
    # tail_logprobs=None -> run() uses the uncached chain-rule fallback over lm_fn (correct, slow-but-toy).


def _mt_word_bigram(sub_log_bigram):
    """Collapse the sub-token bigram into an effective WORD bigram over MT_VOCAB (rows/cols, BOS=EOS=nW):
    a word's internal transitions fold in because the bigram only sees the previous word's last piece."""
    nW = len(MT_VOCAB)
    wlog = np.full((nW + 1, nW + 1), -np.inf)
    intern = {w: sum(float(sub_log_bigram[MT_SPAN[w][i - 1], MT_SPAN[w][i]])
                     for i in range(1, len(MT_SPAN[w]))) for w in MT_VOCAB}
    for j, w2 in enumerate(MT_VOCAB):
        first2 = MT_SPAN[w2][0]
        wlog[nW, j] = float(sub_log_bigram[MT_EOS, first2]) + intern[w2]      # from BOS
        for i, w1 in enumerate(MT_VOCAB):
            wlog[i, j] = float(sub_log_bigram[MT_SPAN[w1][-1], first2]) + intern[w2]
    for i, w1 in enumerate(MT_VOCAB):
        wlog[i, nW] = float(sub_log_bigram[MT_SPAN[w1][-1], MT_EOS])          # to EOS
    wlog[nW, nW] = float(sub_log_bigram[MT_EOS, MT_EOS])                       # empty sentence
    return jnp.asarray(wlog)


def _mt_exact(observed, sub_log_bigram, Lmax):
    """Exact posterior over MT_VOCAB word sequences (up to Lmax words), scored by the effective word
    bigram + the surface channel DP -- the same brute force as exact_posterior, for the multi-token toy."""
    obs_words = observed.split(); M = len(obs_words)
    obs_char = jnp.stack([encode(w)[0] for w in obs_words])
    wsurf = jnp.stack([encode(w)[0] for w in MT_VOCAB]); wlen = jnp.asarray([encode(w)[1] for w in MT_VOCAB])
    emit = jax.vmap(jax.vmap(channel_logpdf, in_axes=(None, 0, 0)),
                    in_axes=(0, None, None))(obs_char, wsurf, wlen)            # (M, nW)
    a0 = jnp.where(jnp.arange(M + 1) == 0, 0.0, jnp.arange(M + 1) * WINS)
    wlog, nW = _mt_word_bigram(sub_log_bigram), len(MT_VOCAB)
    sents, joints = [], []
    for n in range(Lmax + 1):
        seqs = (jnp.array(list(itertools.product(range(nW), repeat=n)), jnp.int32).reshape(-1, n)
                if n else jnp.zeros((1, 0), jnp.int32))
        if n == 0:
            lm = jnp.array([wlog[nW, nW]]); chan = jnp.array([a0[M]])
        else:
            frm = jnp.concatenate([jnp.full((seqs.shape[0], 1), nW), seqs], axis=1)
            to = jnp.concatenate([seqs, jnp.full((seqs.shape[0], 1), nW)], axis=1)
            lm = jnp.sum(wlog[frm, to], axis=1)

            def chan_one(seq):
                alpha = a0
                for i in range(n):
                    alpha = _word_row_update(alpha, emit[:, seq[i]], WDEL, WINS)
                return alpha[M]

            chan = jax.vmap(chan_one)(seqs)
        sents.extend(" ".join(MT_VOCAB[int(i)] for i in s) for s in seqs)
        joints.append(lm + chan)
    joints = jnp.concatenate(joints); post = jax.nn.softmax(joints)
    words = {}
    for s, p in zip(sents, post):
        words[s] = words.get(s, 0.0) + float(p)
    return words, float(logsumexp(joints))


def _mt_smc(observed, key, sub_log_bigram, P=8000, band=None):
    model = _mt_model(sub_log_bigram)
    st, dw, logZ, _ = pairhmm_smc.run(observed, key, model, P=P, proposal="caprop",
                                      wdel=WDEL, wins=WINS, band=band)
    return {s: p for s, p in pairhmm_smc.decode(st, dw, model, top=50)}, logZ


_MT_PEAKED = [("the kitten sat", 50), ("the cat sat", 1), ("the dog sat", 1),
              ("the kitchen mat", 1), ("the cat mat", 1)]


def test_multitoken_copy_matches_exact():
    """D0 (Phase D): a correctly-spelled MULTI-TOKEN word ('kitten' = kit+ten) is reconstructed
    VERBATIM, and the SMC posterior matches exact enumeration -- the COPY of a multi-token word, the
    case the single-token filter could not even represent. Certifies the (token span, surface_id)
    candidate + chain-rule LM + span splice on the toy by brute force (band=None to match enumeration)."""
    lm = _mt_sub_log_bigram(_MT_PEAKED)
    exact, _ = _mt_exact("the kitten sat", lm, Lmax=4)
    smc, _ = _mt_smc("the kitten sat", jax.random.PRNGKey(0), lm, P=8000)
    assert max(smc, key=smc.get) == max(exact, key=exact.get) == "the kitten sat", \
        f"multi-token COPY: SMC MAP {max(smc, key=smc.get)!r} vs exact {max(exact, key=exact.get)!r}"
    map_s = max(exact, key=exact.get)
    assert abs(exact[map_s] - smc.get(map_s, 0.0)) < 0.15, "multi-token MAP mass off vs exact"
    assert tv_distance(exact, smc) < 0.2, f"multi-token posterior too far from exact: TV {tv_distance(exact, smc):.3f}"


def test_multitoken_substitution_recovered():
    """D0 (Phase D): a typo whose correction is a MULTI-TOKEN word is recovered -- observed 'kiten'
    (not a word) is substituted to 'kitten' (= kit+ten), i.e. an N:1->1:N substitution to a word the
    single-token candidate set never contained. The dropped 't' is a char-channel edit-distance-1 to
    'kitten'; the peaked LM makes the restoration the MAP."""
    lm = _mt_sub_log_bigram(_MT_PEAKED)
    smc, _ = _mt_smc("the kiten sat", jax.random.PRNGKey(0), lm, P=8000)
    assert max(smc, key=smc.get) == "the kitten sat", \
        f"multi-token substitution: SMC MAP {max(smc, key=smc.get)!r}, expected 'the kitten sat'"


def _mt_rejuv_ctx_pool(observed, sub_log_bigram, slack=3, Ke=8):
    """Build the rejuvenation ctx + multi-token candidate pool the SAME way pairhmm_smc.run does
    (augmented emit_full + pool-from-inventory), for the multi-token toy. Returns
    (model, ctx, (pool_tok, pool_len, pool_surf), T_max, Wmax)."""
    model = _mt_model(sub_log_bigram)
    obs_words = observed.split(); M = len(obs_words)
    obs_spans = [None] * M                                 # toy: no obs_spans -> candidate_words re-encodes
    obs_char = jnp.stack([encode(w)[0] for w in obs_words])
    emit_full = jax.vmap(jax.vmap(channel_logpdf, in_axes=(None, 0, 0)),
                         in_axes=(0, None, None))(obs_char, model.vocab_char, model.vocab_clen)
    # _build_candidates gained a 9th return (copy_mask, word-action scoring); the rejuv ctx/pool
    # don't use it -- mirror live pairhmm_smc.run, which unpacks it then builds RejuvCtx without it.
    ef, es, em, mt_span, mt_len, emit_aug, _copy_mask, T_max, _n_mt = pairhmm_smc._build_candidates(
        model, obs_words, obs_spans, obs_char, emit_full, max_dist=2, Ke=Ke)
    Wmax = M + slack
    pool = pairhmm_smc._rejuv_pool_from_inventory(ef, es, em, mt_span, mt_len, M, Wmax, T_max)
    a0 = jnp.where(jnp.arange(M + 1) == 0, 0.0, jnp.arange(M + 1) * WINS)
    ctx = rejuv.RejuvCtx(model, emit_aug, a0, M, 0, Wmax, WDEL, WINS, 2, T_max, 1.0)
    return model, ctx, pool, T_max, Wmax


def test_multitoken_rejuv_invariance():
    """R4: the rejuvenation sweep operates over MULTI-TOKEN words. Applied to a cloud already at the
    multi-token posterior (the forward filter on 'the kitten sat', rejuv off), one full sweep leaves the
    MAP on the truth and barely perturbs the posterior -- the invariance that proves the move does not
    corrupt a good multi-token cloud. Exercises the whole R4 path end-to-end: the boundary-aware unpack,
    the span+surface-id candidate pool, the multi-token suffix-tail scorer, and the cumsum-scatter splice
    (none of which the single-token rejuv gates touch). Mirrors test_rejuv_leaves_exact_posterior_invariant
    for T_max>1 (band=2, restricted SymSpell pool, so a slightly looser TV than the full-vocab toy)."""
    lm = _mt_sub_log_bigram(_MT_PEAKED)
    model, ctx, pool, T_max, Wmax = _mt_rejuv_ctx_pool("the kitten sat", lm)
    st, dw, _, sl = pairhmm_smc.run("the kitten sat", jax.random.PRNGKey(0), model, P=3000,
                                    proposal="caprop", wdel=WDEL, wins=WINS, band=2)
    cb0, cl0, _nw, wl0, ws0, _la, dn0 = st
    anc = jax.random.categorical(jax.random.PRNGKey(1), dw, shape=(3000,))   # resample to equal weights
    g = lambda a: a[anc]
    cb0, cl0, wl0, ws0, dn0 = g(cb0), g(cl0), g(wl0), g(ws0), g(dn0)
    before = rejuv.decode_counts(cb0, cl0, model, sl)
    assert max(before, key=before.get) == "the kitten sat", "forward cloud not at the multi-token truth"
    swp = rejuv.make_sweep(ctx, *pool, max_tail=(3 + 1) * T_max + 1)
    cb, cl, _wl, _ws, _la2, _mlw = swp(jax.random.PRNGKey(2), cb0, cl0, wl0, ws0,
                                       positions=range(0, Wmax), done=dn0)
    after = rejuv.decode_counts(cb, cl, model, sl)
    assert max(after, key=after.get) == "the kitten sat", \
        f"multi-token rejuv moved the MAP off the truth: {max(after, key=after.get)!r}"
    assert tv_distance(before, after) < 0.2, \
        f"multi-token rejuv perturbed the posterior: TV {tv_distance(before, after):.3f}"


def _mt_find_in_pool(pool, slot, word):
    """(span, surf_id) of ``word`` among slot ``slot``'s rejuv pool candidates (matched by surface)."""
    pool_tok, pool_len, pool_surf = (np.asarray(a[slot]) for a in pool)
    for k in range(pool_tok.shape[0]):
        if pool_len[k] == 0:
            continue
        span = tuple(int(x) for x in pool_tok[k, :pool_len[k]])
        if _mt_decode(span) == word:
            return span, int(pool_surf[k])
    raise AssertionError(f"{word!r} not in slot {slot} pool")


def test_multitoken_rejuv_recovers_collapsed():
    """R4 + channel guard: a cloud COLLAPSED onto a wrong MULTI-TOKEN reading ('the kitchen sat') is
    pulled back to the truth 'the kitten sat' by the sweep, because the channel must favour the
    multi-token candidate ('kitten' is a perfect match to the observed 'kitten'; 'kitchen' is distance
    2). This is the gate that catches the channel-column bug invariance missed: if an augmented
    multi-token surface id is read at the wrong (clipped) column, the sweep cannot tell kitten from
    kitchen and the recovery fails."""
    lm = _mt_sub_log_bigram(_MT_PEAKED)
    model, ctx, pool, T_max, Wmax = _mt_rejuv_ctx_pool("the kitten sat", lm)
    words = [((0,), 0),                                   # 'the' (single-token; surf == token)
             _mt_find_in_pool(pool, 1, "kitchen"),         # wrong multi-token reading at the kitten slot
             ((2,), 2)]                                    # 'sat'
    P, LCTX = 4000, ctx.seed_len + Wmax * T_max + 1
    toks = [t for span, _s in words for t in span]
    buf = jnp.full((P, LCTX), 0, jnp.int32).at[:, :len(toks)].set(jnp.array(toks, jnp.int32))
    clen = jnp.full((P,), len(toks), jnp.int32)
    word_len = jnp.zeros((P, Wmax), jnp.int32).at[:, :3].set(
        jnp.array([len(span) for span, _s in words], jnp.int32))
    word_surf = jnp.zeros((P, Wmax), jnp.int32).at[:, :3].set(
        jnp.array([s for _span, s in words], jnp.int32))
    assert _mt_decode(toks) == "the kitchen sat", f"collapsed cloud decodes to {_mt_decode(toks)!r}"
    swp = rejuv.make_sweep(ctx, *pool, max_tail=(3 + 1) * T_max + 1)
    key = jax.random.PRNGKey(0)
    for _ in range(6):
        key, sub = jax.random.split(key)
        buf, clen, word_len, word_surf, _la, _mlw = swp(sub, buf, clen, word_len, word_surf,
                                                        positions=range(0, 3))
    rec = rejuv.decode_counts(buf, clen, model, ctx.seed_len)
    assert max(rec, key=rec.get) == "the kitten sat", \
        f"multi-token rejuv failed to recover the collapse: MAP {max(rec, key=rec.get)!r}"


# NOTE: caprop's lower-variance-than-bootstrap logZ is NOT a pass/fail gate. At toy scale the edge
# is small (~1.1-1.3x) and case/seed-sensitive -- at P=3000 on the flat spurious case it even
# inverts. It is the fully-adapted proposal's signature and grows with LM cost (plan finding #4), so
# main() reports caprop-vs-bootstrap logZ std as a diagnostic instead of asserting an ordering.


def _bigram_table(lm_fn):
    """Probe an injected lm_fn into a (V+1, V+1) log-bigram so exact enumeration can use it."""
    rows = []
    for prev in range(V + 1):
        if prev == BOS:
            cb, cl = jnp.zeros((16,), jnp.int32), jnp.int32(0)
        else:
            cb, cl = jnp.zeros((16,), jnp.int32).at[0].set(prev), jnp.int32(1)
        rows.append(lm_fn(cb, cl))
    return jnp.stack(rows)


def main():
    import time
    pr = functools.partial(print, flush=True)  # stream progress even when redirected to a file
    t_start = time.time()
    pr(f"[{0.0:5.1f}s] loading toy LM + char channel ...")
    peaked = _peaked()
    # Exact enumeration is feasible only for SHORT sentences (intended length <= ~4 over V=12).
    # The missing-word case ('the cat sat the mat' -> length 6) is too big to enumerate; it is an
    # A3 behavioural MAP-recovery gate instead.
    cases = [("teh cat", 3, lm_logits, "flat"),
             ("the cat cat", 3, lm_logits, "flat"),
             ("teh cat sat", 4, peaked, "peaked")]
    for ci, (obs, Lmax, lm, tag) in enumerate(cases, 1):
        et = lambda: time.time() - t_start
        M = len(obs.split())
        n_enum = sum(V ** n for n in range(Lmax + 1))
        pr(f"\n[{et():5.1f}s] case {ci}/{len(cases)}: {obs!r} (M={M}, {tag} LM, enumerating {n_enum} seqs)")
        lb = LOG_BIGRAM if lm is lm_logits else _bigram_table(lm)
        exact, exact_logZ = exact_posterior(obs, log_bigram=lb, Lmax=Lmax)
        rel = exact_logZ - _a0_const(M)
        pr(f"[{et():5.1f}s]   exact done. log-marginal (filter-relative): {rel:.3f}")
        for prop in ("bootstrap", "caprop"):
            zs = jnp.array([smc_posterior(obs, jax.random.PRNGKey(s), P=2000, lm_fn=lm, proposal=prop)[1]
                            for s in range(8)])
            smc, _ = smc_posterior(obs, jax.random.PRNGKey(0), P=2000, lm_fn=lm, proposal=prop)
            pr(f"[{et():5.1f}s]   [{prop:9}] logZ mean {float(zs.mean()):.3f} "
               f"(std {float(zs.std()):.3f}, gap {float(zs.mean())-rel:+.3f})  TV {tv_distance(exact, smc):.3f}")
        pr(f"           exact MAP: {max(exact, key=exact.get)!r}")
        for s, p in sorted(exact.items(), key=lambda kv: -kv[1])[:4]:
            pr(f"             exact p={p:.3f}  {s!r}")
    pr(f"\n[{et():5.1f}s] done.")


if __name__ == "__main__":
    main()
