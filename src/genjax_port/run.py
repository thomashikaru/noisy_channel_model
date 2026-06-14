"""Run the token-level Genjax noisy-channel particle filter on a sentence.

Usage (from repo root, in the `ncgenjax` conda env):

    python -m src.genjax_port.run
    python -m src.genjax_port.run --sentence "the boy handed the pencil too the girl" \\
        --particles 64 --seed 0

Given an observed (possibly noisy) sentence -- now *arbitrary* text via the Pythia BPE
tokenizer -- it prints the posterior over inferred *intended* sentences, ranked by particle
frequency.
"""

import argparse
from collections import Counter

import jax

from .particle_filter_unified import run_particle_filter_unified
from .tokenizer import encode

DEFAULT_SENTENCE = "the boy handed the pencil to the girl"


def summarize(sentences, top_k=5):
    counts = Counter(sentences)
    total = len(sentences)
    return [(s, c, c / total) for s, c in counts.most_common(top_k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--max_dist", type=int, default=2,
                        help="max character edit distance for word-substitution candidates "
                             "(SymSpell). Distance is not hard-capped at modeling time -- "
                             "SUB_PARAM**d down-weights far candidates; this only bounds "
                             "candidate retrieval.")
    args = parser.parse_args()

    obs = encode(args.sentence)
    key = jax.random.key(args.seed)
    # Unified word-scan filter: copy / substitution (incl. BPE-token-count typos) / insertion /
    # deletion in one model, with dedup LM forwards by default.
    sentences, log_marginal, min_ess = run_particle_filter_unified(
        key, jax.numpy.array(obs), num_particles=args.particles,
        max_dist=args.max_dist, progress=True,
    )

    norm_observed = " ".join(args.sentence.split())
    print(f"observed : {args.sentence}")
    print(f"particles: {args.particles}   log P(observed) ~= {log_marginal:.3f}"
          f"   min ESS = {min_ess:.1f}/{args.particles}\n")
    print("inferred intended sentences (posterior over alternatives):")
    for sent, count, frac in summarize(sentences, top_k=args.top_k):
        marker = "  <- matches observed" if sent == norm_observed else ""
        print(f"  {frac:5.1%}  ({count:>3d})  {sent}{marker}")


if __name__ == "__main__":
    main()
