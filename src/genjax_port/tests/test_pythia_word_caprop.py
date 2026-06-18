"""Smoke tests for the pair-HMM Pythia channel-aware filter."""

import os

import jax

os.environ.setdefault("NC_LM", "EleutherAI/pythia-70m")

from genjax_port import lm_penzai
from genjax_port.poc_word_indel import _word_row_update, _wins_only_row
from genjax_port.pythia_word_caprop import run, decode, _norm
import jax.numpy as jnp


def test_wins_only_row():
    a = jnp.array([0.0, -1.0, -2.0], dtype=jnp.float32)
    out = _wins_only_row(a, jnp.float32(-0.5))
    assert float(out[0]) == float(-jnp.inf)
    assert abs(float(out[1]) - (-0.5)) < 1e-5


def test_kv_tail_parity():
    """Cached tail logprobs match uncached within ~5e-3 (opt-in via NC_USE_KV=1)."""
    import os
    if os.environ.get("NC_USE_KV", "0") != "1":
        return
    lm_penzai.load_model()
    ctx = jnp.array([[lm_penzai.EOS_ID, 253, 4814, 0, 0]], jnp.int32)
    ilen = jnp.array([3], jnp.int32)
    tails = jnp.array([[[253, 4814, 0, 0]]], jnp.int32)
    tlens = jnp.array([[2]], jnp.int32)
    u = lm_penzai.batch_tail_logprobs(ctx, ilen, tails, tlens, use_kv=False)
    c = lm_penzai.batch_tail_logprobs(ctx, ilen, tails, tlens, use_kv=True)
    assert float(jnp.max(jnp.abs(u - c))) < 5e-3


def test_missing_word_smoke():
    # P=128 is the validated budget; P=4 decodes pure noise (resample-and-count on 4 particles).
    lm_penzai.load_model()
    st, lw, _, sl = run("i want go home", jax.random.PRNGKey(0), P=128, Ke=8, J=8)
    top = decode(st, lw, skip=sl, top=1)[0][0]
    assert _norm(top) == _norm("i want to go home")


if __name__ == "__main__":
    test_wins_only_row()
    test_kv_tail_parity()
    test_missing_word_smoke()
    print("OK")
