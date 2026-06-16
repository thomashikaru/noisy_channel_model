"""M1 representation test: the per-word Switch model (copy multi-token vs sub single-token).

Locks spike 3 (planning/MIGRATION_PLAN.md §6.3 -- the N:1 word emission risk, resolved via Switch). The
importance == manual-joint assertions are LM-independent, so they hold for any NC_LM. Uses a
real multi-candidate word ('experimemt' -> 4 tokens, 3 sub candidates) so the C-way switch (not
just 2-way) is exercised.
"""

import jax
import jax.numpy as jnp

from src.genjax_port import lm_penzai as L
from src.genjax_port import noise_word as NW
from src.genjax_port.config import ACTION_ALPHAS
from src.genjax_port.model import COPY, SUB
from src.genjax_port.lm_genjax import lm_logp
from src.genjax_port.genjax_model import make_word_model, word_constraints
from src.genjax_port.tokenizer import encode


def _setup():
    ctx = encode("the boy did an")
    obs_word_ids = encode(" experimemt")
    n = len(obs_word_ids)
    M = len(ctx) + n + 4
    buf0 = jnp.full(M, L.EOS_ID, jnp.int32).at[jnp.arange(1, 1 + len(ctx))].set(jnp.asarray(ctx))
    ilen0 = 1 + len(ctx)
    subs = NW.word_sub_candidates("experimemt", max_dist=2)
    ap = jnp.log(jnp.asarray(ACTION_ALPHAS, jnp.float32) / sum(ACTION_ALPHAS))
    return ctx, obs_word_ids, n, buf0, ilen0, subs, ap


def _branch_args(buf0, ilen0, ap, subs):
    copy_args = (buf0, ilen0, ap[COPY])
    sub_args = [(buf0, ilen0, ap[SUB] + NW.word_sub_loglik(d)) for _, d in subs]
    return copy_args, sub_args


def test_copy_branch_importance_matches_manual():
    """COPY branch importance == action prior + LM chain-rule over the word's n tokens."""
    _, obs_word_ids, n, buf0, ilen0, subs, ap = _setup()
    model = make_word_model(n, len(subs))
    copy_args, sub_args = _branch_args(buf0, ilen0, ap, subs)
    chm = word_constraints(0, obs_word_ids, None)
    _, w = model.importance(jax.random.key(0), chm, (0, copy_args, *sub_args))

    b, il, manual = buf0, ilen0, float(ap[COPY])
    for t in obs_word_ids:
        manual += float(lm_logp(b, il)[t])
        b = b.at[il].set(t); il += 1
    assert abs(float(w) - manual) < 1e-2, (float(w), manual)


def test_each_sub_branch_importance_matches_manual():
    """Every SUB branch (C-way switch) importance == action prior + LM(x) + word_sub_loglik."""
    _, obs_word_ids, n, buf0, ilen0, subs, ap = _setup()
    assert len(subs) >= 2, "need >=2 sub candidates to exercise the C-way (>2 branch) switch"
    model = make_word_model(n, len(subs))
    copy_args, sub_args = _branch_args(buf0, ilen0, ap, subs)
    for k, (x, d) in enumerate(subs):
        chm = word_constraints(k + 1, obs_word_ids, x)
        _, w = model.importance(jax.random.key(k), chm, (k + 1, copy_args, *sub_args))
        manual = float(ap[SUB]) + float(NW.word_sub_loglik(d)) + float(lm_logp(buf0, ilen0)[x])
        assert abs(float(w) - manual) < 1e-2, (k, float(w), manual)


def test_sub_correction_beats_literal_copy():
    """The single-token SUB 'experiment' reading outscores the 4-token COPY 'experimemt'."""
    _, obs_word_ids, n, buf0, ilen0, subs, ap = _setup()
    model = make_word_model(n, len(subs))
    copy_args, sub_args = _branch_args(buf0, ilen0, ap, subs)
    exp_id = encode(" experiment")[0]
    k_exp = next(k for k, (x, _) in enumerate(subs) if x == exp_id)

    _, w_copy = model.importance(
        jax.random.key(0), word_constraints(0, obs_word_ids, None), (0, copy_args, *sub_args))
    _, w_sub = model.importance(
        jax.random.key(1), word_constraints(k_exp + 1, obs_word_ids, exp_id),
        (k_exp + 1, copy_args, *sub_args))
    assert float(w_sub) > float(w_copy), (float(w_sub), float(w_copy))


if __name__ == "__main__":
    L.load_model()
    test_copy_branch_importance_matches_manual()
    print("OK  copy branch importance matches manual")
    test_each_sub_branch_importance_matches_manual()
    print("OK  each sub branch (C-way switch) importance matches manual")
    test_sub_correction_beats_literal_copy()
    print("OK  sub correction beats literal copy")
