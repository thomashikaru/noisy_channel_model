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

caprop's lower-variance-than-bootstrap logZ is the fully-adapted proposal's signature, but at TOY
scale the edge is small and case/seed-sensitive (plan finding #4: it grows with LM cost), so it is
NOT a pass/fail gate -- it is reported by ``main()`` as a diagnostic only.

Enumeration is VECTORIZED (one jitted batched DP per length) and kept small (short sentence, capped
length) so it runs in seconds. Behavioural recovery of the harder missing/spurious edits lives in
the A3 edit-type gates, not here.

Run as a script:  python -m genjax_port.tests.test_pairhmm_exact
Run as a test:    pytest src/genjax_port/tests/test_pairhmm_exact.py
"""

import functools
import itertools

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from genjax_port.poc_pairhmm_channel import channel_logpdf, encode
from genjax_port.poc_word_smc import V, VOCAB, VOCAB_IDS, VOCAB_LEN, WORD2IDX
from genjax_port.poc_word_indel import BOS, EOS, LOG_BIGRAM, _word_row_update, lm_logits
from genjax_port.noise_word import _damerau_levenshtein
from genjax_port import poc_word_indel_caprop as caprop
from genjax_port import pairhmm_smc

WDEL = float(jnp.log(0.1))
WINS = float(jnp.log(0.05))


def _toy_model(lm_fn):
    """A :class:`pairhmm_smc.PairHMMModel` over the toy vocab, LM injected. The toy bigram is the
    same filter as Pythia with a different LM, so certifying it here certifies the shared code."""
    def candidate_ids(word, max_dist, Ke):
        cands = sorted((_damerau_levenshtein(word, w, max_dist), WORD2IDX[w]) for w in VOCAB)
        return [i for d, i in cands if d <= max_dist][:Ke]

    return pairhmm_smc.PairHMMModel(
        lm_fn=jax.vmap(lm_fn),                 # batch the single-particle toy LM over the cloud
        eos_id=EOS, emit_vocab=V,
        vocab_char=VOCAB_IDS, vocab_clen=VOCAB_LEN, channel_logpdf=channel_logpdf,
        char_ids=encode, candidate_ids=candidate_ids, obs_words=str.split,
        decode_ids=lambda t: " ".join(VOCAB[i] for i in t), seed_ids=())


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
def smc_posterior(observed, key, P=8000, proposal="caprop", lm_fn=lm_logits):
    model = _toy_model(lm_fn)
    st, dw, logZ, sl = pairhmm_smc.run(observed, key, model, P=P, proposal=proposal,
                                       wdel=WDEL, wins=WINS)
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
