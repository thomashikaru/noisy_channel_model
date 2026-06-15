"""Phase 2b: the masked autoregressive chain model (`rejuv_model.make_masked_chain_model`).

The keystone for the masked carrier: its importance weight must equal the model joint computed by
hand -- specifically, the LM buffer must thread over **active slots only** (an inactive slot neither
scores nor shifts later context). This is the genuinely new piece vs Phase 2a (which had no
autoregression). Plus a vmap check (the move/materialization must batch over particles).

    NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python -m src.genjax_port.tests.test_rejuv_model
"""

import math

import jax
import jax.numpy as jnp

from src.genjax_port import lm_penzai as L
from src.genjax_port.lm_genjax import lm_logp
from src.genjax_port.rejuv_model import (
    make_masked_chain_model, chain_constraints, active_tokens, P_PRESENT_DEFAULT,
)


def _manual_joint(present, tokens, buf0, ilen0, p_present):
    """Hand-computed model joint: sum_k log P(present_k) + sum_{active k} lm(x_k | active prefix)."""
    lp_present = math.log(p_present)
    lp_absent = math.log(1.0 - p_present)
    total = 0.0
    buf = buf0
    il = int(ilen0)
    for p, t in zip(present, tokens):
        total += lp_present if p else lp_absent
        if p:
            total += float(lm_logp(buf, jnp.int32(il))[t])   # score under the ACTIVE prefix
            buf = buf.at[il].set(jnp.int32(t))
            il += 1
    return total


def test_masked_chain_importance_matches_active_prefix_joint():
    """Importance over a mixed present/absent pattern == the hand joint, proving active-only
    autoregression (inactive slots are skipped in both score and LM context)."""
    K = 4
    M = 1 + K + 4
    buf0 = jnp.full(M, L.EOS_ID, jnp.int32)
    ilen0 = jnp.array(1, jnp.int32)
    p = P_PRESENT_DEFAULT
    model = make_masked_chain_model(K, p_present=p)

    toks = [10, 20, 30, 40]                       # arbitrary valid token ids
    for present in ([True, True, True, True],
                    [True, False, True, False],
                    [False, True, True, True],
                    [False, False, False, False]):
        chm = chain_constraints(present, toks)
        _, w = model.importance(jax.random.key(0), chm, (buf0, ilen0))
        manual = _manual_joint(present, toks, buf0, ilen0, p)
        assert abs(float(w) - manual) < 1e-2, (present, float(w), manual)


def test_masked_chain_importance_vmaps_over_particles():
    """Importance vmaps over particles (fixed mask pattern, per-particle tokens) and == the loop."""
    K = 3
    M = 1 + K + 4
    buf0 = jnp.full(M, L.EOS_ID, jnp.int32)
    ilen0 = jnp.array(1, jnp.int32)
    model = make_masked_chain_model(K)
    P = 8
    present_pat = [True, False, True]              # fixed across particles (real mask, one absent)
    tok_rows = jnp.array([[5 + i, 6 + i, 7 + i] for i in range(P)], jnp.int32)

    def imp(key, tok_row):
        chm = chain_constraints(present_pat, [tok_row[k] for k in range(K)])
        _, w = model.importance(key, chm, (buf0, ilen0))
        return w

    keys = jax.random.split(jax.random.key(1), P)
    ws_vmap = jax.vmap(imp)(keys, tok_rows)
    ws_loop = jnp.stack([imp(keys[i], tok_rows[i]) for i in range(P)])
    assert jnp.allclose(ws_vmap, ws_loop, atol=1e-3), (ws_vmap, ws_loop)


if __name__ == "__main__":
    L.load_model()
    test_masked_chain_importance_matches_active_prefix_joint()
    print("OK  masked chain importance == active-prefix joint")
    test_masked_chain_importance_vmaps_over_particles()
    print("OK  masked chain importance vmaps over particles")
