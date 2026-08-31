"""Smoke tests for the pair-HMM Pythia channel-aware filter."""

import os

import jax

os.environ.setdefault("NC_LM", "EleutherAI/pythia-70m")

from genjax_port import lm_penzai
from genjax_port.word_dp import _word_row_update, _wins_only_row
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
    # rejuv="off" is explicit because run() has no default (see REJUV_CHOICES): this asserts the
    # forward-only filter restores a dropped function word on its own, with no rejuvenation help.
    lm_penzai.load_model()
    st, lw, _, sl = run("i want go home", jax.random.PRNGKey(0), P=128, Ke=8, J=8, rejuv="off")
    top = decode(st, lw, skip=sl, top=1)[0][0]
    assert _norm(top) == _norm("i want to go home")


def test_word_stats_smoke():
    """Phase-2 §3.6(5): finite per-word outputs on the live Pythia path from ONE cheap run --
    surprisal_nc finite with sum(S)+S_end == −logZ, alignment posteriors summing to 1 per unit,
    and the plain-LM baseline finite on the SAME copy spans. Positional-vs-DP P(error) agreement
    is reported, not asserted (positional is exact only when no indel shifted the alignment)."""
    import numpy as np
    from genjax_port import word_stats
    from genjax_port.pythia_word_caprop import lm_word_surprisals
    lm_penzai.load_model()
    obs = "teh cat sat on teh mat"
    ws, dg = {}, {}
    st, lw, logZ, _sl = run(obs, jax.random.PRNGKey(0), P=64, Ke=8, J=8, rejuv="off",
                            word_stats=ws, diag=dg)
    S = np.asarray(ws["surprisal_nc"])
    assert np.all(np.isfinite(S)), f"non-finite surprisal_nc: {S}"
    assert abs(float(S.sum()) + ws["surprisal_end_nc"] + logZ) < 1e-4, \
        "sum(S_k) + S_end != -logZ"
    post = word_stats.alignment_posteriors(st, lw, dg)
    tot = post["p_copy"] + post["p_sub"] + post["p_ins"]
    assert np.allclose(tot, 1.0, atol=1e-6), f"unit posteriors sum to {tot}"
    base = lm_word_surprisals(obs)
    assert np.all(np.isfinite(base["surprisal_lm"])) and np.isfinite(base["surprisal_end_lm"])
    assert base["units"] == post["units"]
    print("word_stats smoke:"
          f"  S_nc={np.round(S, 2).tolist()}"
          f"  S_lm={np.round(base['surprisal_lm'], 2).tolist()}"
          f"  p_err_dp={np.round(1.0 - post['p_copy'], 3).tolist()}"
          f"  p_err_positional={np.round(post['p_err_positional'], 3).tolist()}")


if __name__ == "__main__":
    test_wins_only_row()
    test_kv_tail_parity()
    test_missing_word_smoke()
    test_word_stats_smoke()
    print("OK")
