"""Behavioral eval: does aligned conditional rejuvenation improve over the forward-only filter?

Both arms run the SAME forward filter (copy / substitution / deletion / insertion). The ONLY
difference is that the rejuv arm interleaves surprisal-gated SUBSTITUTION rejuvenation after each
word's resample (``run_smc_conditional_rejuv_aligned``), while the baseline does not
(``run_smc_substitution`` with identical forward settings).

The suite targets *reanalysis* cases: a literal-valid word the forward filter commits early (its
channel weight ~1 makes the correct substitution candidate die in the every-word resample) that
only later context reveals as wrong (e.g. ``too`` -> ``to`` once ``the store`` arrives). Rejuvenation
re-proposes the word with the suffix in view, so it can recover from the depletion. We report the
posterior mass on the intended target sentence for each arm and the delta.

    PYTHONPATH=. python -m src.genjax_port.tests.eval_rejuv

Runs on the default LM (pythia-70m). Set NC_LM=EleutherAI/pythia-410m for the stronger LM.
"""

import os
from collections import Counter

import jax
import jax.numpy as jnp

from src.genjax_port import lm_penzai as L
from src.genjax_port.rejuv_bridge import run_smc_conditional_rejuv_aligned
from src.genjax_port.smc_substitution import run_smc_substitution
from src.genjax_port.tokenizer import encode

SEED = 0
PARTICLES = int(os.environ.get("NC_P", "64"))
MAX_DIST = 2
MAX_DELETIONS = 1
# rejuvenation config (mirrors run_example_native.sh). The gate fires on (contextual - unigram)
# surprisal, so the center is ~0 (Gen.jl default), not the old raw-surprisal ~4.
LOOKBACK = 2
LOGPROB_THRESH = 0.0
LOGPROB_SPREAD = 1.0
REJUV_SWEEPS = 2

# (observed, intended target, note). Reanalysis cases: the target word is literal-valid-but-wrong
# until the suffix arrives.
CASES = [
    ("he went too the store", "he went to the store", "too->to (suffix 'the store')"),
    ("i need too sleep now", "i need to sleep now", "too->to (suffix 'sleep now')"),
    ("we have too win this", "we have to win this", "too->to (suffix 'win this')"),
    ("i got it form the store", "i got it from the store", "form->from (suffix 'the store')"),
    ("he jumped of the cliff", "he jumped off the cliff", "of->off (suffix 'the cliff')"),
    # control: a clean sentence should stay put under both arms (rejuv must not hurt it)
    ("the boy did an experiment today", "the boy did an experiment today", "clean control"),
]


def _mass(sentences, target):
    norm = " ".join(target.split())
    c = Counter(sentences)
    return c.get(norm, 0) / len(sentences), c.most_common(3)


def main():
    L.load_model()
    print(f"eval: P={PARTICLES} max_dist={MAX_DIST} max_deletions={MAX_DELETIONS} insertion=on "
          f"LM={os.environ.get('NC_LM', 'default')}")
    print(f"rejuv: lookback={LOOKBACK} thresh={LOGPROB_THRESH} spread={LOGPROB_SPREAD} "
          f"sweeps={REJUV_SWEEPS}\n")

    rows = []
    for observed, target, note in CASES:
        obs = jnp.asarray(encode(observed))

        # baseline: forward filter only (no interleaved rejuvenation)
        base_sents, _, base_ess = run_smc_substitution(
            jax.random.key(SEED), obs, num_particles=PARTICLES, max_dist=MAX_DIST,
            max_deletions=MAX_DELETIONS, allow_insertion=True)
        base_mass, base_top = _mass(base_sents, target)

        # rejuv: same forward filter + interleaved surprisal-gated sub-flip
        rej_sents, _, rej_ess, acc = run_smc_conditional_rejuv_aligned(
            jax.random.key(SEED), obs, num_particles=PARTICLES, max_dist=MAX_DIST,
            lookback=LOOKBACK, logprob_thresh=LOGPROB_THRESH, logprob_spread=LOGPROB_SPREAD,
            n_sweeps=REJUV_SWEEPS, max_deletions=MAX_DELETIONS, allow_insertion=True)
        rej_mass, rej_top = _mass(rej_sents, target)

        rows.append((observed, target, note, base_mass, rej_mass, acc))
        print(f"observed : {observed}")
        print(f"target   : {target}    [{note}]")
        print(f"  forward-only : target={base_mass:6.1%}  minESS={base_ess:4.1f}   top={_fmt(base_top, len(base_sents))}")
        print(f"  +rejuv       : target={rej_mass:6.1%}  minESS={rej_ess:4.1f}  acc={acc:5.1%}  top={_fmt(rej_top, len(rej_sents))}")
        delta = rej_mass - base_mass
        print(f"  delta        : {delta:+6.1%}\n")

    print("=" * 70)
    print("summary (target posterior mass):")
    print(f"  {'case':<32} {'fwd':>7} {'+rejuv':>7} {'delta':>7}")
    for observed, _, _, b, r, _ in rows:
        print(f"  {observed[:32]:<32} {b:7.1%} {r:7.1%} {r-b:+7.1%}")
    mean_delta = sum(r - b for _, _, _, b, r, _ in rows) / len(rows)
    print(f"\n  mean delta: {mean_delta:+.1%}")


def _fmt(top, total):
    return ", ".join(f"{c/total:.0%} {s!r}" for s, c in top[:2])


if __name__ == "__main__":
    main()
