"""Proof-of-concept #3: full-word insertions and deletions via a NESTED pair-HMM.

Builds on poc_pairhmm_channel.py (char-level alignment DP) and poc_word_smc.py (sequential SMC
over words with a Pythia-interface toy LM). Those two assumed one intended word per observed word.
This script lifts the SAME pair-HMM recurrence one level up, to the sentence, so the model can
infer that an observed word was SPURIOUS (inserted) or that an intended word is MISSING (deleted).

The nesting, in one table:

    char-level (inside a word)        word-level (across the sentence)
    --------------------------        --------------------------------
    copy / substitute a char    <->   align intended word w to observed word k
                                        cost = channel(obs_k | w)  [the char-level DP score]
    delete a char               <->   intended word with no observation  = MISSING word
    insert a char               <->   observed word with no intended src  = SPURIOUS word

Each particle no longer carries a scalar "words consumed" counter; it carries the word-level
forward vector alpha[k] = P(intended prefix so far, exactly k observed words consumed), summed
over word alignments. That vector IS the Rao-Blackwellized word alignment, and it is fixed shape
(length M+1), so per-particle state stays vmap-clean. The intended sentence length becomes a
latent the filter discovers (it samples words until EOS); an intended sentence shorter/longer
than the observed one is precisely an inferred insertion / deletion.

Because we marginalize the alignment, the SMC weight is the increment in total forward mass dZ,
with a terminal correction at EOS forcing full consumption alpha[M] (cf. reading grid[n_x,n_o] in
PoC #1). We inject dZ through a `factor` so GenJAX's importance still returns the weight.

Validation here:
  * indel DISABLED (missing/spurious costs -> -inf) must reproduce the 1:1 PoC #2 behaviour;
  * a SPURIOUS-word sentence is corrected to a SHORTER intended sentence;
  * a MISSING-word sentence is corrected to a LONGER intended sentence.

Run:  python -m genjax_port.tests.toy_bigram
"""

from collections import Counter

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import genjax
from genjax import ChoiceMap, exact_density

from genjax_port.tests.toy_channel import channel_logpdf, encode, PAD
from genjax_port.tests.toy_vocab import (VOCAB, V, VOCAB_IDS, VOCAB_LEN, WORD2IDX,
                                         CORPUS, LCTX)
# DP recurrences live in the core now (shared with the live filter); re-import for this demo.
from genjax_port.word_dp import _word_row_update, _wins_only_row, _ess

BOS = V   # context row id for start-of-sentence
EOS = V   # next-token column id for end-of-sentence

# --- EOS-aware bigram LM with the heavy model's interface: logits(ctx_buf, ctx_len) -> [V+1] ---
def _build_bigram(corpus):
    counts = jnp.ones((V + 1, V + 1))  # rows: prev word (BOS=V); cols: next word (EOS=V)
    for sent in corpus:
        prev = BOS
        for w in sent.split():
            cur = WORD2IDX[w]
            counts = counts.at[prev, cur].add(1.0)
            prev = cur
        counts = counts.at[prev, EOS].add(1.0)
    return jnp.log(counts / counts.sum(axis=1, keepdims=True))


LOG_BIGRAM = _build_bigram(CORPUS)  # (V+1, V+1)


def lm_logits(ctx_buf, ctx_len):
    prev = jnp.where(ctx_len > 0, ctx_buf[jnp.maximum(ctx_len - 1, 0)], BOS)
    return LOG_BIGRAM[prev]


# deterministic log-weight injector (same trick as genjax_model.factor)
factor = exact_density(lambda key, lw: jnp.float32(0.0), lambda v, lw: lw, "factor")


# _word_row_update / _wins_only_row / _ess moved to genjax_port.word_dp (core); imported above.


def run(observed, key, P=16000, enable_indel=True, wdel=jnp.log(0.1), wins=jnp.log(0.05),
        slack=3):
    """Sequential RB-SMC over intended words; alpha (word alignment) marginalized per particle."""
    obs_words = observed.split()
    M = len(obs_words)
    obs_ids = jnp.stack([encode(w)[0] for w in obs_words])               # (M, Lchar)
    # word-level emission table: EMIT[k, w] = char-level channel score of obs word k from intended w
    EMIT = jax.vmap(jax.vmap(channel_logpdf, in_axes=(None, 0, 0)),
                    in_axes=(0, None, None))(obs_ids, VOCAB_IDS, VOCAB_LEN)  # (M, V)
    WDEL = wdel if enable_indel else -jnp.inf
    WINS = wins if enable_indel else -jnp.inf

    @genjax.gen
    def kernel(state):
        ctx_buf, ctx_len, log_alpha, done = state
        s = genjax.categorical(lm_logits(ctx_buf, ctx_len)) @ "s"     # next intended word or EOS
        is_eos = s == EOS
        w = jnp.where(is_eos, 0, s)
        advance = (~done) & (~is_eos)                                  # actually emit a word?
        Z_old = logsumexp(log_alpha)
        new_alpha = _word_row_update(log_alpha, EMIT[:, w], WDEL, WINS)
        # RB-SMC intermediate weight: increment in total forward mass Z (bootstrap proposal cancels
        # the LM prior). The terminal full-consumption correction alpha[M]/Z is applied ONCE at the
        # end, not here -- otherwise resampling discards it. A frozen/EOS'd particle adds nothing.
        incr = jnp.where(advance, logsumexp(new_alpha) - Z_old, 0.0)
        incr = jnp.where(jnp.isnan(incr), -jnp.inf, incr)  # -inf - -inf (dead particle) -> -inf
        _ = factor(incr) @ "ev"
        ctx_buf2 = jnp.where(advance, ctx_buf.at[ctx_len].set(w.astype(jnp.int32)), ctx_buf)
        return (ctx_buf2,
                jnp.where(advance, ctx_len + 1, ctx_len),
                jnp.where(advance, new_alpha, log_alpha),
                done | is_eos)

    a0 = jnp.where(jnp.arange(M + 1) == 0, 0.0, jnp.arange(M + 1) * WINS)  # leading spurious words
    state = (jnp.full((P, LCTX), PAD, jnp.int32), jnp.zeros(P, jnp.int32),
             jnp.broadcast_to(a0, (P, M + 1)), jnp.zeros(P, bool))
    log_w = jnp.zeros(P)
    constraint = ChoiceMap.d({"ev": jnp.float32(0.0)})

    @jax.jit
    def extend(key, state):
        keys = jax.random.split(key, P)
        def one(k, st):
            tr, w = kernel.importance(k, constraint, (st,))
            return tr.get_retval(), w
        return jax.vmap(one)(keys, state)

    for _ in range(M + slack):
        key, sub = jax.random.split(key)
        state, w = extend(sub, state)
        log_w = log_w + w
        # Resample only when the cloud has degenerated (ESS < P/2), not every step. Resampling
        # every step is myopic: it prunes a particle that just paid a deletion cost before the
        # later words reveal that the deletion was right. ESS-triggering keeps that particle alive.
        if _ess(log_w) < 0.5 * P:
            key, sub = jax.random.split(key)
            anc = jax.random.categorical(sub, log_w, shape=(P,))
            state = jax.tree_util.tree_map(lambda a: a[anc], state)
            log_w = jnp.zeros(P)

    # Terminal correction (applied once): reweight to the full-consumption target alpha[M], and
    # add any residual importance weight not yet absorbed by a resample. This kills particles whose
    # intended sentence does not explain ALL observed words; cf. reading grid[n_x, n_o] at the end
    # of the char-level DP in PoC #1.
    _, _, log_alpha, _ = state
    decode_w = log_w + log_alpha[:, M] - logsumexp(log_alpha, axis=1)
    decode_w = jnp.where(jnp.isnan(decode_w), -jnp.inf, decode_w)
    return state, decode_w


def decode(state, decode_w, key=jax.random.PRNGKey(0), top=3):
    ctx_buf, ctx_len, _, _ = state
    anc = jax.random.categorical(key, decode_w, shape=(ctx_buf.shape[0],))
    trajs = [tuple(int(i) for i in ctx_buf[a][:int(ctx_len[a])]) for a in anc]
    counts = Counter(trajs)
    n = len(trajs)
    return [(" ".join(VOCAB[i] for i in t), c / n) for t, c in counts.most_common(top)]


def main():
    key = jax.random.PRNGKey(0)

    print("=== check: indel DISABLED reduces to 1:1 (PoC #2 behaviour) ===")
    key, sub = jax.random.split(key)
    st, dw = run("teh cat sat on teh mat", sub, enable_indel=False)
    print(f"  'teh cat sat on teh mat' -> {decode(st, dw)[0][0]!r}  (expect: the cat sat on the mat)")

    print("\n=== SPURIOUS word: observed has an extra word -> SHORTER intended ===")
    key, sub = jax.random.split(key)
    st, dw = run("the cat cat sat", sub)
    for s, p in decode(st, dw):
        print(f"  {s!r:34} p={p:.2f}")
    print("  (observed 4 words; correct intended is 3: 'the cat sat')")

    print("\n=== MISSING word: observed dropped a word -> LONGER intended ===")
    key, sub = jax.random.split(key)
    st, dw = run("the cat on the mat", sub)
    for s, p in decode(st, dw):
        print(f"  {s!r:34} p={p:.2f}")
    print("  (observed 5 words; correct intended is 6: 'the cat sat on the mat')")


if __name__ == "__main__":
    main()
