"""Proof-of-concept: noisy-channel spelling correction with the alignment MARGINALIZED.

This script de-risks the reframing discussed in planning/FRUSTRATIONS.md. The pain in the
current port comes from *sampling* the alignment between the intended and observed strings
(which characters are copied / substituted / inserted / deleted) with trans-dimensional moves.
That is exactly the latent that dynamic programming sums over exactly and in fixed shape.

The idea, in one line: don't sample the alignment -- marginalize it with a pair-HMM forward
pass, and expose that forward score as a GenJAX ``exact_density`` channel distribution. The
model is then pure ``@gen``:

    intended ~ Categorical(prior over a dictionary)        # the only latent we sample
    observed ~ EditChannel(intended)                       # logpdf = sum over alignments (DP)

Because ``intended`` is the only sampled latent, GenJAX's *built-in* importance sampling
(``Target`` + ``ImportanceK``) computes the posterior for us -- no hand-rolled weights, no MH
ratios, no rejuvenation. We check it against the exact enumerated posterior.

Everything is character-level over a toy alphabet so the example is self-contained and runs in
a second; nothing here needs Pythia/penzai. The same skeleton scales up by (a) swapping the
unigram dictionary prior for an autoregressive LM written as a ``@gen`` scan, and (b) running
the built-in SMC sequentially over words instead of one-shot importance sampling.

Run:  python -m genjax_port.poc_pairhmm_channel
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import genjax
from genjax import ChoiceMap, Target, exact_density
from genjax.inference import smc

# --- toy alphabet -----------------------------------------------------------------------------
# id 0 = PAD; 1..26 = 'a'..'z'; 27 = space. Fixed buffer width L so every string is one shape.
PAD = 0
L = 12


def encode(word):
    ids = [ord(c) - ord("a") + 1 if c != " " else 27 for c in word]
    n = len(ids)
    ids = ids + [PAD] * (L - n)
    return jnp.array(ids, jnp.int32), jnp.int32(n)


# --- channel parameters (per-edit log-probabilities) ------------------------------------------
# A character is copied with prob COPY; otherwise substituted to one of the other 26 letters.
# Independently, the writer may delete an intended char (omission) or insert a spurious one.
# These need not sum to a perfectly normalized path measure for a PoC -- they just have to make
# a copy cheaper than an error, which is what drives the posterior the right way.
COPY_LP = jnp.log(0.90)
SUB_LP = jnp.log(0.10 / 26.0)  # substitute to a *specific* wrong letter
DEL_LP = jnp.log(0.05)         # intended char produced no observation
INS_LP = jnp.log(0.05)         # observed char came from nowhere


def _emit_lp(o_char, x_char):
    """log P(observe o_char | intend x_char) for an aligned (substitute/copy) step."""
    return jnp.where(o_char == x_char, COPY_LP, SUB_LP)


def channel_logpdf(observed_ids, intended_ids, n_x):
    """log P(observed | intended), summed over ALL monotone alignments (pair-HMM forward).

    ``observed_ids``/``intended_ids`` are length-L padded char buffers; ``n_x`` is the intended
    length. The observed length ``n_o`` is read off the value itself (count of non-PAD), so the
    observed string is fully data. Builds the (L+1)x(L+1) log-alpha grid and returns the cell
    ``[n_x, n_o]`` -- the marginal likelihood of the observation under the edit process.
    """
    n_o = jnp.sum(observed_ids != PAD)

    # Row 0: only insertions can have happened (no intended char consumed yet).
    row0 = jnp.arange(L + 1, dtype=jnp.float32) * INS_LP

    def fill_row(prev_row, x_char):
        # cur[0]: only a deletion leads here (consume x_char, emit nothing).
        cur0 = prev_row[0] + DEL_LP

        def step(left, cols):
            o_char, prev_diag, prev_up = cols
            sub = prev_diag + _emit_lp(o_char, x_char)  # align x_char <-> o_char
            dele = prev_up + DEL_LP                      # drop x_char
            ins = left + INS_LP                          # spurious o_char
            cell = logsumexp(jnp.stack([sub, dele, ins]))
            return cell, cell

        cols = (observed_ids, prev_row[:-1], prev_row[1:])  # k = 1..L
        _, rest = jax.lax.scan(step, cur0, cols)
        cur_row = jnp.concatenate([cur0[None], rest])
        return cur_row, cur_row

    _, rows = jax.lax.scan(fill_row, row0, intended_ids)  # rows for j = 1..L
    grid = jnp.concatenate([row0[None], rows])            # (L+1, L+1)
    return grid[n_x, n_o]


# GenJAX distribution: sample is a never-used stub (observed is always constrained data); the
# logpdf is the DP forward score. This mirrors the existing obs_dist pattern in genjax_model.py.
edit_channel = exact_density(
    lambda key, intended_ids, n_x: jnp.zeros(L, jnp.int32),  # stub
    channel_logpdf,
    "edit_channel",
)


# --- dictionary + unigram prior ---------------------------------------------------------------
WORDS = [
    "the", "tea", "ten", "then", "he", "her", "here", "three", "there",
    "receive", "recover", "relieve", "deceive", "ceiling",
    "world", "word", "would", "work", "wood",
    "hello", "help", "held", "hold",
]
FREQ = {"the": 1000, "he": 300, "there": 120, "would": 110, "world": 90,
        "word": 40, "work": 80, "here": 70, "then": 60, "help": 50}

_enc = [encode(w) for w in WORDS]
DICT_IDS = jnp.stack([e[0] for e in _enc])          # (V, L)
DICT_LEN = jnp.stack([e[1] for e in _enc])          # (V,)
_counts = jnp.array([FREQ.get(w, 5) for w in WORDS], jnp.float32)
PRIOR_LOGITS = jnp.log(_counts / _counts.sum())     # (V,)
V = len(WORDS)


# --- the generative model: pure @gen ----------------------------------------------------------
@genjax.gen
def spelling_model():
    """intended ~ unigram prior over the dictionary; observed ~ edit channel of intended."""
    idx = genjax.categorical(PRIOR_LOGITS) @ "intended"
    x_ids = DICT_IDS[idx]
    x_len = DICT_LEN[idx]
    observed = edit_channel(x_ids, x_len) @ "observed"
    return observed


# --- inference --------------------------------------------------------------------------------
def exact_posterior(obs_ids):
    """Ground truth: enumerate the (single) latent. P(intended=w | observed) over the dictionary."""
    joint = jax.vmap(channel_logpdf, in_axes=(None, 0, 0))(obs_ids, DICT_IDS, DICT_LEN)
    joint = joint + PRIOR_LOGITS
    return jax.nn.softmax(joint)


def builtin_posterior(obs_ids, key, k_particles=20000):
    """GenJAX-native: build a Target for the observation, run ImportanceK, read weighted particles.

    GenJAX computes each particle's weight = log P(observed | intended) for us; we never write a
    weight or ratio by hand. Returns the self-normalized posterior over dictionary indices.
    """
    target = Target(spelling_model, (), ChoiceMap.d({"observed": obs_ids}))
    pc = smc.ImportanceK(target, k_particles=k_particles).run_smc(key)
    idxs = pc.particles.get_choices()["intended"]   # (K,) sampled intended indices
    lw = pc.get_log_weights()                        # (K,) importance log-weights
    w = jax.nn.softmax(lw)
    return jax.ops.segment_sum(w, idxs, num_segments=V)


def show(typo, key):
    obs_ids, _ = encode(typo)
    exact = exact_posterior(obs_ids)
    approx = builtin_posterior(obs_ids, key)
    order = jnp.argsort(exact)[::-1][:3]
    print(f"\nobserved '{typo}'  ->  posterior over intended word")
    print(f"    {'word':10} {'exact':>8} {'ImportanceK':>12}")
    for i in order:
        print(f"    {WORDS[int(i)]:10} {float(exact[i]):8.3f} {float(approx[i]):12.3f}")


def main():
    key = jax.random.PRNGKey(0)
    for typo in ["teh", "recieve", "wrold", "wurld", "ther", "hlep"]:
        key, sub = jax.random.split(key)
        show(typo, sub)


if __name__ == "__main__":
    main()
