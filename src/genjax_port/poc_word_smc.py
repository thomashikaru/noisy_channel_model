"""Proof-of-concept #2: sentence-level noisy-channel inference by sequential SMC over words.

Builds directly on poc_pairhmm_channel.py. That script showed, for a single word, that the
edit-alignment can be *marginalized* by a forward DP (a pair-HMM) and exposed as a GenJAX
``exact_density``, so the model is pure ``@gen`` and built-in importance sampling recovers the
exact posterior. This script answers the next question, which is the real architectural risk:

    With an AUTOREGRESSIVE LM prior in the loop and SMC resampling between words, does each
    particle's state stay FIXED SHAPE and vmappable?

If yes, the whole "fighting JAX" problem dissolves: the trans-dimensional alignment is gone
(summed out by the DP), and the only sampled latent is the intended word at each step, drawn
left-to-right from the LM -- exactly a bootstrap particle filter.

Pythia stand-in
---------------
``ToyLM`` mimics the interface of the heavy model -- ``logits(context_buffer, ctx_len) -> [V]``,
next-token logits given a fixed-width context buffer -- but is a smoothed bigram table, so the
spike runs in milliseconds. Swapping in Pythia-via-penzai is literally replacing the body of
``ToyLM.logits``; the SMC scaffolding does not change.

Scope (skipping edge cases, per the plan): observed words are pre-segmented (split on spaces),
one intended word per observed word. Uncertain word boundaries / whole-word insert-delete are a
later DP edge, not a new paradigm -- the same way transposition was a missing edge in PoC #1.

Run:  python -m genjax_port.poc_word_smc
"""

from collections import Counter

import jax
import jax.numpy as jnp

import genjax
from genjax import ChoiceMap

from genjax_port.poc_pairhmm_channel import edit_channel, encode, L, PAD

# --- toy vocabulary + char encodings (for the channel) ----------------------------------------
VOCAB = ["the", "a", "cat", "dog", "sat", "ran", "slept",
         "on", "mat", "hat", "big", "small"]
V = len(VOCAB)
BOS = V  # context id for "start of sentence" (the bigram backoff row)

_enc = [encode(w) for w in VOCAB]
VOCAB_IDS = jnp.stack([e[0] for e in _enc])   # (V, L) char buffers
VOCAB_LEN = jnp.stack([e[1] for e in _enc])   # (V,)
WORD2IDX = {w: i for i, w in enumerate(VOCAB)}


# --- the Pythia stand-in: a smoothed bigram LM with the heavy model's interface ---------------
class ToyLM:
    """next-token logits given a context buffer. Same signature shape as a real LM forward."""

    def __init__(self, corpus):
        counts = jnp.ones((V + 1, V))  # add-one smoothing; rows indexed by prev word (BOS = V)
        for sent in corpus:
            prev = BOS
            for w in sent.split():
                cur = WORD2IDX[w]
                counts = counts.at[prev, cur].add(1.0)
                prev = cur
        self.log_bigram = jnp.log(counts / counts.sum(axis=1, keepdims=True))  # (V+1, V)

    def logits(self, ctx_buf, ctx_len):
        """Logits over the next word given the context buffer (uses only the last token: bigram)."""
        prev = jnp.where(ctx_len > 0, ctx_buf[jnp.maximum(ctx_len - 1, 0)], BOS)
        return self.log_bigram[prev]


CORPUS = [
    "the cat sat on the mat",
    "the dog ran",
    "a big cat slept",
    "the small dog sat",
    "the cat ran on the mat",
    "a dog sat on the hat",
    "the big dog slept",
    "a small cat sat on the mat",
]
LM = ToyLM(CORPUS)
LCTX = 16  # fixed context-buffer width: the per-particle state that must stay constant in shape


# --- one word of the model: pure @gen ---------------------------------------------------------
# Propose the intended word from the LM given the context so far; observe its noisy form through
# the DP edit channel. Used by SMC via its built-in .importance: constrain "obs" to the observed
# word, let GenJAX sample "w" from the LM prior and return the incremental weight = channel
# log-likelihood. That is precisely a bootstrap filter step (propose from prior, weight by
# likelihood) -- and GenJAX computes the weight; we never write one.
@genjax.gen
def word_step(carry):
    ctx_buf, ctx_len = carry
    logits = LM.logits(ctx_buf, ctx_len)
    idx = genjax.categorical(logits) @ "w"
    x_ids = VOCAB_IDS[idx]
    x_len = VOCAB_LEN[idx]
    _obs = edit_channel(x_ids, x_len) @ "obs"
    new_ctx = ctx_buf.at[ctx_len].set(idx.astype(jnp.int32))
    return (new_ctx, ctx_len + 1)


# --- sequential SMC over the words of the sentence --------------------------------------------
def _ess(log_w):
    w = jax.nn.softmax(log_w)
    return 1.0 / jnp.sum(w * w)


def run_smc(observed, key, P=2000, resample=True):
    """Bootstrap particle filter, one step per observed word. Returns (carry, log_w, ess_per_step).

    Particle state is ``(ctx_buf [P, LCTX], ctx_len [P])`` -- FIXED SHAPE across every word, for
    every particle. The step is GenJAX's ``word_step.importance`` vmapped over particles.
    """
    obs_words = observed.split()
    ctx = (jnp.full((P, LCTX), PAD, jnp.int32), jnp.zeros(P, jnp.int32))
    log_w = jnp.zeros(P)
    ess_hist = []

    @jax.jit
    def extend(key, ctx, obs_ids):
        keys = jax.random.split(key, P)
        constraint = ChoiceMap.d({"obs": obs_ids})
        carry_in = ctx

        def one(k, c):
            tr, w = word_step.importance(k, constraint, (c,))
            return tr.get_retval(), tr.get_choices()["w"], w

        new_carry, widx, w = jax.vmap(one)(keys, carry_in)
        return new_carry, widx, w

    for t, ow in enumerate(obs_words):
        obs_ids, _ = encode(ow)
        key, sub = jax.random.split(key)
        ctx, widx, w = extend(sub, ctx, obs_ids)
        log_w = log_w + w
        ess_hist.append(float(_ess(log_w)))
        if resample:
            key, sub = jax.random.split(key)
            anc = jax.random.categorical(sub, log_w, shape=(P,))
            ctx = jax.tree_util.tree_map(lambda a: a[anc], ctx)
            log_w = jnp.zeros(P)

    return ctx, log_w, ess_hist


def decode(ctx, log_w, top=3):
    """Most frequent intended-word trajectory across the (post-resampling) particle cloud."""
    ctx_buf, ctx_len = ctx
    n = int(ctx_len[0])
    trajs = [tuple(int(i) for i in row[:n]) for row in ctx_buf]
    counts = Counter(trajs)
    out = []
    for traj, c in counts.most_common(top):
        out.append((" ".join(VOCAB[i] for i in traj), c / len(trajs)))
    return out


def main():
    key = jax.random.PRNGKey(0)
    trials = [
        ("teh cat sta on teh mat", "the cat sat on the mat"),
        ("the daug ran", "the dog ran"),
        ("a smll dog sat", "a small dog sat"),
    ]
    for noisy, truth in trials:
        key, k1, k2 = jax.random.split(key, 3)
        ctx, log_w, ess = run_smc(noisy, k1, resample=True)
        _, _, ess_noresamp = run_smc(noisy, k2, resample=False)
        post = decode(ctx, log_w)
        print(f"\nobserved : {noisy!r}")
        print(f"truth    : {truth!r}")
        print(f"SMC MAP  : {post[0][0]!r}  (p={post[0][1]:.2f})")
        if len(post) > 1:
            print(f"  runners-up: " + ", ".join(f"{s!r} {p:.2f}" for s, p in post[1:]))
        print(f"  ESS/particle  with resampling: min {min(ess)/2000:.3f}  "
              f"|  no resampling (final): {ess_noresamp[-1]/2000:.4f}")


if __name__ == "__main__":
    main()
