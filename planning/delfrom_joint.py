"""Does the TRUE joint prefer the target restoration ('from') over the junk insertions the bd move found?

If yes (target joint >> junk joint) but the move picks junk, the problem is the PROPOSAL (uniform gap inserts
at locally-fluent wrong positions), fixable by an informed gap. If the junk genuinely scores higher, it's an
LM/model issue. Joint = lm_temp*LM(intended) + channel-action cost, where each intended word with no observed
correspondent is a DELETE action (cost from the Dirichlet-multinomial action prior; no content cost)."""
import math, jax, jax.numpy as jnp
from genjax_port import lm_penzai, tokenizer
from genjax_port.pythia_word_caprop import PRIME
from genjax_port.calibration_word_action_smc import _wellform

A_ALIGN, A_INS, A_DEL = 200.0, 2.0, 2.0
OBS_N = 6   # 'the businessman benefited the tax law'


def lm_logprob(sentence):
    seed = [lm_penzai.EOS_ID] + tokenizer.encode(PRIME)
    sent = tokenizer.encode(" " + sentence)
    buf = seed + sent + [lm_penzai.EOS_ID]
    lp = jax.nn.log_softmax(lm_penzai._raw_logits(jnp.array([buf]))[0], axis=-1)
    nxt = jnp.array(buf[1:])
    return float(jnp.sum(jnp.take_along_axis(lp[:-1], nxt[:, None], axis=1)[len(seed) - 1:, 0]))


def dm(n_align, n_ins, n_del):
    N, A0 = n_align + n_ins + n_del, A_ALIGN + A_INS + A_DEL
    lc = math.lgamma(N + 1) - math.lgamma(n_align + 1) - math.lgamma(n_ins + 1) - math.lgamma(n_del + 1)
    num = (math.lgamma(A_ALIGN + n_align) - math.lgamma(A_ALIGN) + math.lgamma(A_INS + n_ins) - math.lgamma(A_INS)
           + math.lgamma(A_DEL + n_del) - math.lgamma(A_DEL))
    return lc + num - math.lgamma(A0 + N) + math.lgamma(A0)


# (label, intended sentence, n_align, n_ins, n_del) relative to the 6-word observed.
CANDS = [
    ("literal",       "the businessman benefited the tax law",          6, 0, 0),
    ("TARGET from",   "the businessman benefited from the tax law",     6, 0, 1),
    ("junk +father",  "the businessman's father benefited the tax law", 5, 0, 2),   # 's,father del; businessman~sub
    ("junk has the",  "the businessman has the benefited the tax law",  6, 0, 2),
    ("junk +'s",      "the businessman's benefited the tax law",        5, 0, 1),
]


def main():
    lm_penzai.load_model()
    print(f"LM={lm_penzai.MODEL_NAME}  obs='the businessman benefited the tax law'\n")
    print(f"{'cand':16s} {'LM':>9s} {'actCost':>8s} {'JOINT':>9s}")
    base = None
    for lab, s, na, ni, nd in CANDS:
        lm = lm_logprob(_wellform(s))
        act = dm(na, ni, nd)
        joint = lm + act
        if base is None:
            base = joint
        print(f"{lab:16s} {lm:>9.2f} {act:>8.2f} {joint:>9.2f}   Δvs literal={joint - base:+.2f}")


if __name__ == "__main__":
    main()
