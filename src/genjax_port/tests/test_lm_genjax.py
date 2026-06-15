"""Spike 1 as a regression test: penzai LM as a genjax distribution in a @gen Scan model.

Proves the three facts the native port stands on (MIGRATION_PLAN.md §2, "Spike 1"):
  - the @gen Scan model with ``lm_token`` simulates and the trace auto-records a "tok" choice
    per step (the addressable structure rejuvenation needs);
  - ``importance`` with constrained tokens equals the direct LM chain-rule log-density;
  - ``vmap`` runs P independent particle traces (the penzai forward batches under vmap).
"""

import time

import jax
import jax.numpy as jnp
from genjax import ChoiceMap as C

from src.genjax_port import lm_penzai as L
from src.genjax_port.lm_genjax import lm_logp
from src.genjax_port.genjax_model import make_lm_scan_model
from src.genjax_port.tokenizer import encode


def test_scan_simulate_auto_records_tokens():
    """simulate runs and the trace exposes one 'tok' choice per scan step."""
    T = 6
    model, M = make_lm_scan_model(T)
    init = (jnp.full(M, L.EOS_ID, jnp.int32), jnp.array(1, jnp.int32))
    tr = model.simulate(jax.random.key(0), (init, None))
    toks = tr.get_choices()[:, "tok"]
    assert toks.shape == (T,)
    assert toks.dtype == jnp.int32


def test_importance_matches_lm_chain_rule():
    """importance weight on a constrained sentence == manual LM chain-rule (LM-independent)."""
    obs_ids = encode("the boy ran home")
    T = len(obs_ids)
    model, M = make_lm_scan_model(T)
    init = (jnp.full(M, L.EOS_ID, jnp.int32), jnp.array(1, jnp.int32))
    vchm = C.empty().at[:, "tok"].set(jnp.asarray(obs_ids, jnp.int32))
    _, w = model.importance(jax.random.key(1), vchm, (init, None))

    buf = jnp.full(M, L.EOS_ID, jnp.int32)
    ilen = 1
    direct = 0.0
    for tid in obs_ids:
        direct += float(lm_logp(buf, ilen)[tid])
        buf = buf.at[ilen].set(tid)
        ilen += 1
    assert abs(float(w) - direct) < 1e-2, (float(w), direct)


def test_vmap_runs_independent_particles():
    """vmap simulate produces P distinct particle traces of the right shape."""
    T = 6
    model, M = make_lm_scan_model(T)
    init = (jnp.full(M, L.EOS_ID, jnp.int32), jnp.array(1, jnp.int32))
    sim = jax.jit(jax.vmap(lambda k: model.simulate(k, (init, None))))
    P = 8
    keys = jax.random.split(jax.random.key(2), P)
    t0 = time.time()
    tr = sim(keys)
    toks = tr.get_choices()[:, "tok"]
    jax.block_until_ready(toks)
    dt = time.time() - t0
    assert toks.shape == (P, T)
    # particles are seeded by distinct keys, so not all rows are identical
    assert int(jnp.sum(jnp.any(toks != toks[0], axis=1))) > 0
    print(f"  vmap P={P} compile+run={dt:.2f}s")


if __name__ == "__main__":
    L.load_model()
    test_scan_simulate_auto_records_tokens()
    print("OK  scan simulate auto-records tokens")
    test_importance_matches_lm_chain_rule()
    print("OK  importance matches LM chain-rule")
    test_vmap_runs_independent_particles()
    print("OK  vmap runs independent particles")
