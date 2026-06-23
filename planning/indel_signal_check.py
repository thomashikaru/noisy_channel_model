"""Inference-limited vs signal-limited classifier for the failing insertion/deletion battery items.

For each item we compare the model's JOINT logπ of the LITERAL reading (= well-formed observed, all
copies) vs the TARGET reading (= intended; restores a dropped word OR removes a spurious one), under the
deployment config (align channel, lm_temp=1.0, prime='.').

  logπ(intended | observed) = lm_temp * logP_LM(intended | prime) + logP_channel(observed | intended)

The LM term we compute exactly (full-sentence teacher forcing, seeded as the live filter). The channel term
differs between the two readings only by ONE action: a deletion-restoration adds a DEL action (the intended
word with no observed token); a spurious-insertion removal adds an INS action (the observed word with no
intended token). We report the LM difference and a Dirichlet-multinomial estimate of the action-prior cost
of that one extra action, then classify:

  LM_gain(target) - action_penalty  > 0  => true posterior PREFERS the edit => INFERENCE-limited (SMC misses it)
                                     < 0  => true posterior prefers literal  => SIGNAL-limited (70m's fault)
"""
import math
import jax, jax.numpy as jnp
from genjax_port import lm_penzai, tokenizer
from genjax_port.pythia_word_caprop import PRIME
from genjax_port.calibration_word_action_smc import _wellform
from genjax_port.unigram import unigram_surprisal

LM_TEMP = 1.0
# align action prior Dirichlet(align=200, ins=2, del=2)
A_ALIGN, A_INS, A_DEL = 200.0, 2.0, 2.0


def lm_logprob(sentence):
    """log P_LM(sentence | prime) seeded exactly as the live filter ([EOS] + encode(PRIME)), incl. final EOS."""
    seed = [lm_penzai.EOS_ID] + tokenizer.encode(PRIME)
    sent = tokenizer.encode(" " + sentence)
    buf = seed + sent + [lm_penzai.EOS_ID]
    logits = lm_penzai._raw_logits(jnp.array([buf]))
    all_lp = jax.nn.log_softmax(logits, axis=-1)[0]
    nxt = jnp.array(buf[1:])
    lp = jnp.take_along_axis(all_lp[:-1], nxt[:, None], axis=1)[:, 0]
    return float(jnp.sum(lp[len(seed) - 1:]))


def dm_action_cost(n_align, n_ins, n_del):
    """log marginal of an action multiset under Dirichlet-multinomial(A_ALIGN,A_INS,A_DEL) (theta integrated out)."""
    N = n_align + n_ins + n_del
    A0 = A_ALIGN + A_INS + A_DEL
    logcoef = math.lgamma(N + 1) - math.lgamma(n_align + 1) - math.lgamma(n_ins + 1) - math.lgamma(n_del + 1)
    num = (math.lgamma(A_ALIGN + n_align) - math.lgamma(A_ALIGN)
           + math.lgamma(A_INS + n_ins) - math.lgamma(A_INS)
           + math.lgamma(A_DEL + n_del) - math.lgamma(A_DEL))
    den = math.lgamma(A0 + N) - math.lgamma(A0)
    return logcoef + num - den


# (id, observed, intended, kind)  kind: 'del'=restore a dropped word, 'ins'=remove a spurious word
ITEMS = [
    ("DEL-the-01a", "we went to store", "we went to the store", "del"),
    ("DEL-a-01a", "he is good man", "he is a good man", "del"),
    ("DEL-a-02a", "she lives in big house", "she lives in a big house", "del"),
    ("DELTO-01a", "The mother gave the candle a daughter", "The mother gave the candle to a daughter", "del"),
    ("DELTO-02a", "The waiter served the soup the customers", "The waiter served the soup to the customers", "del"),
    ("DELFROM-01a", "The businessman benefited the tax law", "The businessman benefited from the tax law", "del"),
    ("DELFROM-02a", "The patient slowly recovered the illness", "The patient slowly recovered from the illness", "del"),
    ("DELFOR-01a", "The tailor sewed the dress the bride", "The tailor sewed the dress for the bride", "del"),
    ("LADDER-give-2", "The volunteer gave the shelter the children", "The volunteer gave the shelter to the children", "del"),
    ("LADDER-send-2", "The clerk sent the branch the manager", "The clerk sent the branch to the manager", "del"),
    ("DEL-of-01a", "this is one the best", "this is one of the best", "del"),   # WORKS (E=0.70) -- calibration
    ("INS-02a", "the cat sat on on the mat", "the cat sat on the mat", "ins"),  # WORKS (E=0.95) -- calibration
    ("INS-to-01a", "The mother gave the daughter to the candle", "The mother gave the daughter the candle", "ins"),
    ("INS-to-02a", "The waiter served the customers to the soup", "The waiter served the customers the soup", "ins"),
    ("INS-01a", "the boy handed handed the pencil to the girl", "the boy handed the pencil to the girl", "ins"),
]


def main():
    lm_penzai.load_model()
    print(f"LM={lm_penzai.MODEL_NAME}  lm_temp={LM_TEMP}  prime={PRIME!r}  align_alpha=(200,2,2)\n")
    print(f"{'item':14s} {'kind':4s} {'nWords':>6s}  {'LM_lit':>9s} {'LM_tgt':>9s} {'LM_gain':>8s}  "
          f"{'actPen':>7s}  {'JOINTΔ':>8s}  verdict")
    for iid, obs, intt, kind in ITEMS:
        lit, tgt = _wellform(obs), _wellform(intt)
        lm_lit, lm_tgt = lm_logprob(lit), lm_logprob(tgt)
        lm_gain = LM_TEMP * (lm_tgt - lm_lit)
        n = len(obs.split())
        if kind == "del":       # target restores a word: literal=(n align), target=(n align + 1 del).
            act_pen = dm_action_cost(n, 0, 0) - dm_action_cost(n, 0, 1)   # literal - target  (>0 = target costs more)
            # a DELETION has no content cost: the restored word's identity is in the intended (scored by the LM).
            content = 0.0
        else:                   # target removes a spurious word: literal=(n align), target=(n-1 align + 1 ins).
            act_pen = dm_action_cost(n, 0, 0) - dm_action_cost(n - 1, 1, 0)
            # an INSERTION pays a CONTENT cost -unigram_surprisal(removed_word): the observed still contains the
            # word, so the channel must have inserted it (omitting this is why a content-word dedup looks winnable
            # but is not -- 'handed' is rare so calling it an insertion is expensive).
            from collections import Counter
            diff = Counter(obs.split()) - Counter(intt.split())   # multiset diff -> handles duplicate removal
            removed = next(iter(diff.elements()))
            content = -unigram_surprisal(removed)
        joint_delta = lm_gain - act_pen + content   # logπ(target) - logπ(literal)
        verdict = "INFERENCE-limited (edit should win)" if joint_delta > 0 else "signal-limited (70m prefers literal)"
        print(f"{iid:14s} {kind:4s} {n:>6d}  {lm_lit:>9.2f} {lm_tgt:>9.2f} {lm_gain:>+8.2f}  "
              f"{act_pen:>7.2f}  {joint_delta:>+8.2f}  {verdict}")


if __name__ == "__main__":
    main()
