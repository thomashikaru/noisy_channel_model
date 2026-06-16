"""Spike 2 as a regression test: substitution noisy channel + native Rejuvenate reanalysis.

Proves (planning/MIGRATION_PLAN.md §2, "Spike 2"):
  - a @gen noisy-channel model (x ~ LM, o ~ table-channel(x)) has a joint log-density that
    matches a manual computation (LM-independent identity);
  - native ``Rejuvenate`` performs the reanalysis flip -- a trace started at the LITERAL
    reading (x = the observed typo " too") moves to the higher-posterior intended " to".

The two-step model is unrolled for clarity (scan composition is covered by test_lm_genjax);
it reuses the channel building blocks from genjax_model.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import genjax
from genjax import ChoiceMap as C

from src.genjax_port import lm_penzai as L
from src.genjax_port.lm_genjax import lm_logp, lm_token
from src.genjax_port.genjax_model import obs_dist, token_candidates, _obs_logpdf
from src.genjax_port.tokenizer import encode, surface

try:
    from genjax import exact_density
except ImportError:
    from genjax._src.generative_functions.distributions.distribution import exact_density


def _setup():
    """Prefix 'he wants', observed ' too go' (typo too->to). Returns model args + ids."""
    prefix = encode("he wants")
    o0 = encode(" too")[0]
    o1 = encode(" go")[0]
    to_id = encode(" to")[0]
    M = len(prefix) + 6
    buf0 = jnp.full(M, L.EOS_ID, jnp.int32).at[jnp.arange(1, 1 + len(prefix))].set(
        jnp.asarray(prefix)
    )
    ilen0 = 1 + len(prefix)
    cx0, cl0 = token_candidates(o0)
    cx1, cl1 = token_candidates(o1)
    return buf0, ilen0, o0, o1, to_id, cx0, cl0, cx1, cl1


@genjax.gen
def _channel_model(buf0, ilen0, cx0, cl0, cx1, cl1):
    x0 = lm_token(buf0, ilen0) @ "x0"
    _ = obs_dist(x0, cx0, cl0) @ "o0"
    buf1 = buf0.at[ilen0].set(x0.astype(jnp.int32))
    x1 = lm_token(buf1, ilen0 + 1) @ "x1"
    _ = obs_dist(x1, cx1, cl1) @ "o1"
    return (x0, x1)


def test_channel_joint_matches_manual():
    """Fully-constrained importance weight == manual joint log-density (LM-independent)."""
    buf0, ilen0, o0, o1, to_id, cx0, cl0, cx1, cl1 = _setup()
    margs = (buf0, ilen0, cx0, cl0, cx1, cl1)
    # ' to' must be a candidate for the observed ' too', else the example is malformed
    assert bool((cx0 == to_id).any())

    chm = C.d({"x0": jnp.int32(to_id), "o0": jnp.int32(o0),
               "x1": jnp.int32(o1), "o1": jnp.int32(o1)})
    _, w = _channel_model.importance(jax.random.key(0), chm, margs)
    manual = (float(lm_logp(buf0, ilen0)[to_id]) + float(_obs_logpdf(o0, to_id, cx0, cl0))
              + float(lm_logp(buf0.at[ilen0].set(to_id), ilen0 + 1)[o1])
              + float(_obs_logpdf(o1, o1, cx1, cl1)))
    assert abs(float(w) - manual) < 1e-2, (float(w), manual)


def test_rejuvenate_reanalysis_flips_literal_to_posterior():
    """Start at the literal reading x0=' too'; Rejuvenate flips it to the posterior ' to'.

    Validated on pythia-70m and pythia-410m. LM-dependent (the LM must prefer ' to'); run the
    suite on a Pythia LM. If a future LM made this borderline, gate on NC_LM rather than delete.
    """
    from genjax._src.generative_functions.static import StaticRequest
    from genjax.inference.requests import Rejuvenate

    buf0, ilen0, o0, o1, to_id, cx0, cl0, cx1, cl1 = _setup()
    margs = (buf0, ilen0, cx0, cl0, cx1, cl1)

    # local-posterior proposal over x0's candidate set ~ LM * channel
    def _prop_sample(key, buf, il, cx, cl):
        sc = lm_logp(buf, il)[cx] + cl
        return cx[jax.random.categorical(key, sc)].astype(jnp.int32)

    def _prop_logpdf(x, buf, il, cx, cl):
        sc = lm_logp(buf, il)[cx] + cl
        m = cx == x
        return jnp.where(jnp.any(m), jnp.max(jnp.where(m, sc, -jnp.inf)) - logsumexp(sc), -jnp.inf)

    cand_prop = exact_density(_prop_sample, _prop_logpdf, "cand_prop")

    chm0 = C.d({"x0": jnp.int32(o0), "o0": jnp.int32(o0),
                "x1": jnp.int32(o1), "o1": jnp.int32(o1)})
    tr, _ = _channel_model.importance(jax.random.key(1), chm0, margs)
    assert int(tr.get_choices()["x0"]) == int(o0)  # starts literal

    req = StaticRequest({"x0": Rejuvenate(cand_prop, lambda chm: (buf0, ilen0, cx0, cl0))})
    key = jax.random.key(2)
    flipped = False
    for _ in range(30):
        key, k1, k2 = jax.random.split(key, 3)
        new_tr, w, _, _ = req.edit(k1, tr, genjax.Diff.no_change(tr.get_args()))
        accept = jnp.log(jax.random.uniform(k2)) < w
        tr = jax.tree_util.tree_map(lambda a, b: jnp.where(accept, a, b), new_tr, tr)
        if int(tr.get_choices()["x0"]) == int(to_id):
            flipped = True
            break
    assert flipped, f"x0 stayed {surface(int(tr.get_choices()['x0']))!r}, expected ' to'"


if __name__ == "__main__":
    L.load_model()
    test_channel_joint_matches_manual()
    print("OK  channel joint matches manual")
    test_rejuvenate_reanalysis_flips_literal_to_posterior()
    print("OK  Rejuvenate reanalysis flips literal -> posterior")
