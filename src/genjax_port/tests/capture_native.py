"""M4 parity: run the full README suite through the genjax-native filter (all ops) vs the golden.

Uses the native word-scan SMC (substitution + deletion + insertion) at a single fixed bucket via
``run_smc_batch`` so the whole suite shares one compiled forward. Prints each native posterior
beside the golden (``golden_targets.json``, from the hand-rolled unified filter). Golden targets
are soft idealized behaviors, not bit-parity -- the gate is matching behavior within MC noise.

``golden_targets.json`` is a **410m reference** artifact (captured by ``capture_golden.py``). The
default LM is now 70m, so run this with ``NC_LM=EleutherAI/pythia-410m`` to compare like-for-like;
on 70m the posteriors legitimately differ (70m mangles short words) and the comparison is only
illustrative.

    NC_LM=EleutherAI/pythia-410m PYTHONPATH=. python -m src.genjax_port.tests.capture_native
"""

import json
import os
from collections import Counter

import jax
import jax.numpy as jnp

from src.genjax_port import lm_penzai as L
from src.genjax_port.smc_substitution import run_smc_batch, required_buffer_size
from src.genjax_port.tokenizer import encode

SEED = 0
PARTICLES = 64
MAX_DIST = 2
MAX_DELETIONS = 1


def _golden():
    path = os.path.join(os.path.dirname(__file__), "golden_targets.json")
    with open(path) as f:
        data = json.load(f)
    by_obs = {c["observed"]: c for c in data["cases"]}
    return data["particles"], by_obs


def main():
    L.load_model()
    g_particles, golden = _golden()
    observed = list(golden.keys())
    obs_ids = [jnp.asarray(encode(s)) for s in observed]
    bucket = max(required_buffer_size(o, MAX_DELETIONS) for o in obs_ids)
    print(f"native filter: P={PARTICLES} max_deletions={MAX_DELETIONS} insertion=on bucket={bucket}\n")

    results = run_smc_batch(jax.random.key(SEED), obs_ids, bucket=bucket, num_particles=PARTICLES,
                            max_dist=MAX_DIST, max_deletions=MAX_DELETIONS, allow_insertion=True,
                            progress=True)

    for s, (sents, logm, ess) in zip(observed, results):
        g = golden[s]
        print(f"\nobserved : {s}")
        print(f"ideal    : {g['ideal']}")
        total = len(sents)
        top = Counter(sents).most_common(4)
        print(f"  native  (logP~={logm:.1f} minESS={ess:.1f}/{PARTICLES}):")
        for sent, c in top:
            print(f"      {c/total:6.1%}  ({c:>3d})  {sent}")
        print(f"  golden  (logP~={g['log_marginal']:.1f} minESS={g['min_ess']:.1f}/{g_particles}):")
        for row in g["posterior"][:4]:
            print(f"      {row['frac']:6.1%}  ({row['count']:>3d})  {row['sent']}")


if __name__ == "__main__":
    main()
