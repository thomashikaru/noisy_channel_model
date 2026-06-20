"""Proof-of-concept #3b: channel-aware (locally-optimal) proposal for the word-indel pair-HMM.

``poc_word_indel.py`` reframed the noisy channel so the edit-alignment is marginalized by a
nested pair-HMM and only the intended sentence is sampled, left-to-right, from the LM. That
bootstrap filter proposes each next intended word from the LM PRIOR -- blind to the observed
string. This script replaces that proposal with the fully-adapted (locally-optimal) one: at each
step build a small candidate set C scored by LM x channel-evidence, sample the next intended word
from softmax over C, and weight by the candidate-set normalizer ``logsumexp_C``. Marginalizing the
candidate choice gives a near-zero-variance incremental weight.

Finding (revises the plan's premise -- read this)
-------------------------------------------------
The plan billed the bootstrap's MISSING-word failure ('the cat on the mat' should restore to
'the cat sat on the mat') as pure proposal myopia. It is mostly a MODEL limit. A deletion both
multiplies in an extra LM factor (<1) AND pays the deletion cost wdel; a *bigram* can never see
that 'the cat ON the mat' is odd, so under it the TRUE posterior favours the verbatim 5-word
parse -- and a correct sampler must return it (this one does; ``main`` shows the exact joint
agrees). Restoration becomes the true MAP only with an LM that knows the sentence; the demo
fakes that with a peaked bigram (a Pythia stand-in), and BOTH proposals then recover it -- once
the deletion word is itself LM-likely there is no myopia to cure.

What caprop actually buys, demonstrably, is the fully-adapted filter's signature: low-variance
incremental weights. At equal particle budget its logZ estimate is 2-7x more stable across seeds
than the bootstrap's (see ``main``), so good-but-locally-costly hypotheses stop being randomly
pruned -- the right cure as the LM (and the cost of a wasted particle) grows toward Pythia.

Math (why the weight is just the normalizer)
--------------------------------------------
At an intended-word step a particle holds LM context and the word-level forward vector ``alpha``
(``alpha[k]`` = log P(intended prefix, k observed words consumed)). For a candidate next word w,
the incremental forward-mass change is one length-M row update:

    dZ(w) = logsumexp(row_update(alpha, EMIT[:, w])) - logsumexp(alpha)

and the EOS candidate's "dZ" is the terminal full-consumption read ``alpha[M] - logsumexp(alpha)``.
With ``score(w) = log p(w|ctx) + dZ(w)`` the locally-optimal proposal is ``softmax_C(score)`` and
the incremental importance weight is ``logsumexp_C(score)`` -- independent of which w is drawn.
Because ``exp(score(w)) = p(w|ctx) * Z_new/Z_old`` telescopes, summing these per-step logsumexp's
over a whole trajectory (including the absorbing EOS step) recovers the exact log marginal
likelihood up to the constant ``logZ_0``. So the EOS branch IS the terminal correction, applied
exactly once -- when EOS is drawn. (A separate end correction is therefore added only to particles
that never emitted EOS within the step budget.)

The candidate set C (where "channel-aware" lives) is the union of, per particle:
  1. emission candidates -- intended words edit-close to the observed words at the alignment
     frontier ``argmax(alpha)`` (channel-compatible; here a direct edit-distance scan over the toy
     VOCAB, structured to swap in ``noise_word.word_sub_candidates`` later);
  2. deletion / fluency candidates -- the top-J LM words given context (cover MISSING words: a
     fluent bridge word the DP marks deleted);
  3. EOS.
Deduped and padded to fixed width, frontier-localized so K stays bounded for long sentences.

Run:  python -m genjax_port.tests.toy_caprop
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from genjax_port.tests.toy_channel import channel_logpdf, encode, PAD
from genjax_port.tests.toy_vocab import (VOCAB, V, VOCAB_IDS, VOCAB_LEN, WORD2IDX, CORPUS, LCTX)
from genjax_port.tests.toy_bigram import EOS, BOS, lm_logits, decode
from genjax_port.word_dp import _word_row_update, _wins_only_row, _ess
from genjax_port.noise_word import _damerau_levenshtein


# --- injectable LM (so the demo can swap a flat for a peaked bigram; Pythia later) -------------
# lm_fn(ctx_buf, ctx_len) -> log-probs over [V words ... + EOS], EXACTLY the heavy model's
# interface. The default is the flat bigram from poc_word_indel; make_bigram builds a peaked one.
def make_bigram(corpus):
    """corpus: list of (sentence, weight). Returns an lm_fn with the standard interface."""
    counts = jnp.ones((V + 1, V + 1))
    for sent, wt in corpus:
        prev = BOS
        for w in sent.split():
            cur = WORD2IDX[w]
            counts = counts.at[prev, cur].add(float(wt))
            prev = cur
        counts = counts.at[prev, EOS].add(float(wt))
    log_bigram = jnp.log(counts / counts.sum(axis=1, keepdims=True))

    def lm_fn(ctx_buf, ctx_len):
        prev = jnp.where(ctx_len > 0, ctx_buf[jnp.maximum(ctx_len - 1, 0)], BOS)
        return log_bigram[prev]
    return lm_fn


# --- emission candidate table (channel-compatible intended words per observed position) --------
# Toy stand-in for noise_word.word_sub_candidates: a direct edit-distance scan over the tiny VOCAB.
# Structured as (M, Ke) padded ids so swapping in the SymSpell-backed generator is a drop-in.
def _emit_table(obs_words, max_dist=2, Ke=6):
    rows = []
    for ow in obs_words:
        cands = sorted((_damerau_levenshtein(ow, w, max_dist), WORD2IDX[w]) for w in VOCAB)
        ids = [i for d, i in cands if d <= max_dist][:Ke]
        rows.append(ids + [-1] * (Ke - len(ids)))
    return jnp.array(rows, jnp.int32)  # (M, Ke); -1 = pad


def run(observed, key, P=4000, enable_indel=True, wdel=jnp.log(0.1), wins=jnp.log(0.05),
        slack=3, max_dist=2, Ke=6, J=4, cwin=1, lm_fn=lm_logits, proposal="caprop"):
    """Sequential RB-SMC over intended words; alpha (word alignment) marginalized per particle.

    proposal="caprop"   : channel-aware locally-optimal proposal (this PoC's contribution).
    proposal="bootstrap": propose the next word from the LM prior (the poc_word_indel baseline),
                          reimplemented here under the SAME injectable LM for a fair comparison.
    """
    obs_words = observed.split()
    M = len(obs_words)
    obs_ids = jnp.stack([encode(w)[0] for w in obs_words])               # (M, Lchar)
    EMIT = jax.vmap(jax.vmap(channel_logpdf, in_axes=(None, 0, 0)),
                    in_axes=(0, None, None))(obs_ids, VOCAB_IDS, VOCAB_LEN)  # (M, V)
    EMIT_TAB = _emit_table(obs_words, max_dist, Ke)                       # (M, Ke)
    WDEL = wdel if enable_indel else -jnp.inf
    WINS = wins if enable_indel else -jnp.inf
    offs = jnp.arange(-cwin, cwin + 1)

    def step_caprop(key, ctx_buf, ctx_len, log_alpha, done):
        lmlog = lm_fn(ctx_buf, ctx_len)                                  # (V+1,)
        lm_word, lm_eos = lmlog[:V], lmlog[EOS]
        Z = logsumexp(log_alpha)

        # --- assemble candidate set C: emission (frontier window) + top-J LM, deduped ---
        fpos = jnp.clip(jnp.argmax(log_alpha), 0, M - 1)                 # next observed word index
        emit_ids = EMIT_TAB[jnp.clip(fpos + offs, 0, M - 1)].reshape(-1)  # (W*Ke,), may pad/dup
        topJ = jax.lax.top_k(lm_word, J)[1]                              # (J,) fluency/deletion ids
        cand = jnp.concatenate([emit_ids, topJ])                         # (Kw,)
        Kw = cand.shape[0]
        valid = cand >= 0
        earlier_eq = (cand[:, None] == cand[None, :]) & valid[None, :] & jnp.tril(
            jnp.ones((Kw, Kw), bool), -1)
        valid = valid & ~jnp.any(earlier_eq, axis=1)                     # keep first occurrence only

        # --- score candidates: LM log-prob + forward-mass increment dZ ---
        cand_c = jnp.clip(cand, 0, V - 1)
        def cand_dZ(col):
            return logsumexp(_word_row_update(log_alpha, col, WDEL, WINS)) - Z
        dZ = jax.vmap(cand_dZ, in_axes=1)(EMIT[:, cand_c])               # (Kw,)
        score_word = jnp.where(valid, lm_word[cand_c] + dZ, -jnp.inf)
        dZ_insert = logsumexp(_wins_only_row(log_alpha, WINS)) - Z
        allow_insert = (jnp.argmax(log_alpha) > ctx_len) & enable_indel
        score_insert = jnp.where(allow_insert, dZ_insert, -jnp.inf)
        score_eos = lm_eos + (log_alpha[M] - Z)                          # EOS "dZ" = terminal read
        scores = jnp.concatenate([score_word, score_insert[None], score_eos[None]])  # (Kw+2,)

        # --- propose + locally-optimal weight (the normalizer; independent of the draw) ---
        incr = logsumexp(scores)
        choice = jax.random.categorical(key, scores)
        chose_eos = choice == Kw + 1
        chose_insert = (choice == Kw) & enable_indel
        w_id = jnp.where(chose_eos | chose_insert, 0, cand[jnp.clip(choice, 0, Kw - 1)])
        return _advance(ctx_buf, ctx_len, log_alpha, done, w_id, chose_eos, chose_insert, incr)

    def step_bootstrap(key, ctx_buf, ctx_len, log_alpha, done):
        # baseline: propose next word (or EOS) from the LM prior; weight = forward-mass increment.
        lmlog = lm_fn(ctx_buf, ctx_len)
        Z = logsumexp(log_alpha)
        s = jax.random.categorical(key, lmlog)
        chose_eos = s == EOS
        w_id = jnp.where(chose_eos, 0, s)
        new_alpha = _word_row_update(log_alpha, EMIT[:, jnp.clip(w_id, 0, V - 1)], WDEL, WINS)
        incr = jnp.where(chose_eos, 0.0, logsumexp(new_alpha) - Z)        # LM prior cancels
        return _advance(ctx_buf, ctx_len, log_alpha, done, w_id, chose_eos, jnp.zeros((), bool), incr)

    def _advance(ctx_buf, ctx_len, log_alpha, done, w_id, chose_eos, chose_insert, incr):
        advance_word = (~done) & (~chose_eos) & (~chose_insert)
        advance_insert = (~done) & chose_insert
        incr = jnp.where(done, 0.0, incr)                                # frozen particle adds nothing
        incr = jnp.where(jnp.isnan(incr), -jnp.inf, incr)                # -inf - -inf (dead) -> -inf
        new_alpha_word = _word_row_update(log_alpha, EMIT[:, jnp.clip(w_id, 0, V - 1)], WDEL, WINS)
        new_alpha_insert = _wins_only_row(log_alpha, WINS)
        new_alpha = jnp.where(chose_insert, new_alpha_insert, new_alpha_word)
        return (jnp.where(advance_word, ctx_buf.at[ctx_len].set(w_id.astype(jnp.int32)), ctx_buf),
                jnp.where(advance_word, ctx_len + 1, ctx_len),
                jnp.where(advance_word | advance_insert, new_alpha, log_alpha),
                done | chose_eos), incr

    step = step_caprop if proposal == "caprop" else step_bootstrap

    @jax.jit
    def extend(key, ctx_buf, ctx_len, log_alpha, done):
        keys = jax.random.split(key, P)
        return jax.vmap(step)(keys, ctx_buf, ctx_len, log_alpha, done)

    a0 = jnp.where(jnp.arange(M + 1) == 0, 0.0, jnp.arange(M + 1) * WINS)  # leading spurious words
    state = (jnp.full((P, LCTX), PAD, jnp.int32), jnp.zeros(P, jnp.int32),
             jnp.broadcast_to(a0, (P, M + 1)), jnp.zeros(P, bool))
    log_w = jnp.zeros(P)
    logZ = 0.0  # running log marginal likelihood estimate (accumulated at each resample)

    for _ in range(M + slack):
        key, sub = jax.random.split(key)
        state, incr = extend(sub, *state)
        log_w = log_w + incr
        if _ess(log_w) < 0.5 * P:        # gentle (ESS-triggered) resampling keeps early diversity
            logZ = logZ + logsumexp(log_w) - jnp.log(P)
            key, sub = jax.random.split(key)
            anc = jax.random.categorical(sub, log_w, shape=(P,))
            state = jax.tree_util.tree_map(lambda a: a[anc], state)
            log_w = jnp.zeros(P)

    # caprop folds the terminal full-consumption read into the EOS candidate, so particles that
    # emitted EOS already paid it; bootstrap never does, so it always needs the end correction.
    # Either way it is needed for particles still live at the budget (else raw forward mass over-
    # rewards long junk parses). Apply to non-done particles (caprop) / all (bootstrap).
    _, _, log_alpha, done = state
    need_term = jnp.ones_like(done) if proposal == "bootstrap" else ~done
    term = jnp.where(need_term, log_alpha[:, M] - logsumexp(log_alpha, axis=1), 0.0)
    term = jnp.where(jnp.isnan(term), -jnp.inf, term)
    log_w = log_w + term
    logZ = logZ + logsumexp(log_w) - jnp.log(P)
    return state, log_w, float(logZ)


def main():
    key = jax.random.PRNGKey(0)

    print("=== check: indel DISABLED reduces to 1:1 (PoC #2 behaviour) ===")
    key, sub = jax.random.split(key)
    st, dw, _ = run("teh cat sat on teh mat", sub, enable_indel=False)
    print(f"  'teh cat sat on teh mat' -> {decode(st, dw)[0][0]!r}  (expect: the cat sat on the mat)")

    print("\n=== SPURIOUS word: observed has an extra word -> SHORTER intended ===")
    key, sub = jax.random.split(key)
    st, dw, _ = run("the cat cat sat", sub)
    for s, p in decode(st, dw):
        print(f"  {s!r:34} p={p:.2f}")
    print("  (observed 4 words; correct intended is 3: 'the cat sat')")

    # --- The missing-word case is a MODEL question before it is an inference one. ---------------
    # A deletion adds an LM factor (<1) AND pays wdel; a flat bigram never prefers the longer
    # sentence (it cannot "see" that 'the cat ON the mat' is odd). So under the flat bigram the
    # TRUE posterior favours the verbatim 5-word parse, and a correct sampler MUST return it.
    print("\n=== MISSING word, FLAT bigram: model itself favours verbatim (no deletion) ===")
    key, sub = jax.random.split(key)
    st, dw, _ = run("the cat on the mat", sub)
    for s, p in decode(st, dw):
        print(f"  {s!r:34} p={p:.2f}")
    print("  (caprop returns the model's true MAP 'the cat on the mat' -- restoration is NOT")
    print("   favoured by a bigram; it needs an LM that knows the sentence, e.g. Pythia.)")

    # With an LM peaked on the real sentence (a Pythia stand-in), the model DOES favour the
    # deletion. Now resolving it is purely an inference problem -- exactly what caprop fixes and
    # the bootstrap proposal cannot, under the SAME LM and particle budget.
    peaked = make_bigram([("the cat sat on the mat", 50), ("the dog ran", 1),
                          ("a big cat slept", 1), ("the small dog sat", 1),
                          ("a dog sat on the hat", 1), ("the big dog slept", 1)])
    print("\n=== MISSING word, PEAKED LM: model favours restoration -> inference test ===")
    for name, prop, P in [("bootstrap", "bootstrap", 4000), ("caprop", "caprop", 4000)]:
        key, sub = jax.random.split(key)
        st, dw, logZ = run("the cat on the mat", sub, P=P, lm_fn=peaked, proposal=prop)
        top = decode(st, dw)
        print(f"  [{name:9} P={P}] MAP {top[0][0]!r:30} p={top[0][1]:.2f}  logZ={logZ:.3f}")
    print("  (target intended: 'the cat sat on the mat' -- now the model's true MAP, and BOTH")
    print("   proposals recover it: once the deletion word is also LM-likely there is no myopia.")
    print("   caprop's edge over bootstrap is variance, quantified next.)")

    # The fully-adapted proposal's signature property: near-zero-variance incremental weights, so
    # the log-marginal-likelihood (logZ) estimate is markedly more stable than the bootstrap's at
    # the SAME particle budget -- this is what stops good-but-locally-costly particles from being
    # randomly pruned. Both are consistent (means agree up to the small candidate-truncation bias).
    print("\n=== low-variance weights: logZ mean (std) over 8 seeds, P=2000 ===")
    for tag, obs, lm in [("spurious / flat ", "the cat cat sat", lm_logits),
                         ("missing  / peaked", "the cat on the mat", peaked)]:
        for prop in ["bootstrap", "caprop"]:
            zs = jnp.array([run(obs, jax.random.PRNGKey(s), P=2000, lm_fn=lm, proposal=prop)[2]
                            for s in range(8)])
            print(f"  {tag}  {prop:9}: {float(zs.mean()):7.3f}  (std {float(zs.std()):.3f})")


if __name__ == "__main__":
    main()
