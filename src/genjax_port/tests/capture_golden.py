"""Capture golden behavioral targets from the hand-rolled unified filter (the migration ref).

Runs ``run_particle_filter_unified`` on the §7 suite at a fixed seed/P and records each
posterior to ``golden_targets.json``. These are **idealized intuitive behaviors, not strict
binary correctness checks** -- M1's native filter should reproduce them within Monte-Carlo
noise, not bit-for-bit. The recorded ``ideal`` field is the human intuition for each case; the
``posterior`` field is what the hand-rolled filter actually produced at this seed/P.

    NC_LM=EleutherAI/pythia-410m PYTHONPATH=. python -m src.genjax_port.tests.capture_golden
"""

import json
import os
from collections import Counter

import jax

from src.genjax_port import lm_penzai as L
from src.genjax_port.particle_filter_unified import run_particle_filter_unified
from src.genjax_port.tokenizer import encode

SEED = 0
PARTICLES = 64
MAX_DIST = 2

# (observed sentence, idealized intuitive behavior). The fractions are soft expectations.
SUITE = [
    ("the boy did an experimemt today",
     "obvious typo: experimemt -> experiment (~1.0)"),
    ("did you recieve the message",
     "borderline typo: recieve -> receive, weak at 410m (~0.1)"),
    ("he wants go home",
     "deletion: reconstruct omitted 'to' (~0.5, high-variance ESS)"),
    ("the boy handed handed the pencil to the girl",
     "insertion: remove the doubled 'handed' (~0.5)"),
    ("the boy did an experiment today",
     "clean control: stay literal, no edit (~1.0)"),
    ("The medics treated the wound to prevent an inflection.",
     "inflection -> infection; period kept as EOS; no spurious 'who' insertion"),
]


def summarize(sentences, top_k=6):
    counts = Counter(sentences)
    total = len(sentences)
    return [{"sent": s, "count": c, "frac": round(c / total, 4)}
            for s, c in counts.most_common(top_k)]


def main():
    L.load_model()
    out = {"model": L.MODEL_NAME, "seed": SEED, "particles": PARTICLES,
           "max_dist": MAX_DIST, "cases": []}
    for observed, ideal in SUITE:
        obs = jax.numpy.array(encode(observed))
        sentences, log_marginal, min_ess = run_particle_filter_unified(
            jax.random.key(SEED), obs, num_particles=PARTICLES,
            max_dist=MAX_DIST, progress=False,
        )
        posterior = summarize(sentences)
        out["cases"].append({
            "observed": observed,
            "ideal": ideal,
            "log_marginal": round(log_marginal, 3),
            "min_ess": round(min_ess, 2),
            "posterior": posterior,
        })
        print(f"\nobserved : {observed}")
        print(f"ideal    : {ideal}")
        print(f"logP~={log_marginal:.1f}  minESS={min_ess:.1f}/{PARTICLES}")
        for row in posterior:
            print(f"  {row['frac']:6.1%}  ({row['count']:>3d})  {row['sent']}")

    path = os.path.join(os.path.dirname(__file__), "golden_targets.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
