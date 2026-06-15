"""M5/R1 tests: native Rejuvenate substitution-flip (incremental reanalysis).

Promotes spike 4. Three checks:
  - reanalysis: a literal trace ("he wants too go home") is corrected too->to by a flip sweep;
  - the suffix participates: the MH weight to flip an early word is larger when later words are in
    the trace (later context votes via the Update's suffix re-scoring);
  - detailed balance: on a 2-word toy the MH stationary histogram matches the brute-force exact
    posterior -- the move provably samples the posterior (docs/model.tex Thm 2), empirically.

LM-dependent; validated on pythia-70m. Run as a script (or via tests/run.py).
"""

import math
from collections import Counter

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from src.genjax_port import lm_penzai as L
from src.genjax_port.lm_genjax import lm_logp
from src.genjax_port.rejuvenation import (
    chain_inputs, literal_trace, flip_request, rejuv_step, rejuv_sweep, decoded_intended,
)
from src.genjax_port.tokenizer import encode, surface
import genjax


def _ids(*words):
    return [encode(w)[0] for w in words]


def test_reanalysis_flip_too_to():
    """A literal 'he wants too go home' is corrected too->to by a rejuvenation sweep."""
    obs = _ids(" he", " wants", " too", " go", " home")
    to_id = encode(" to")[0]
    model, obs, buf0, ilen0, cxs, cls = chain_inputs(obs)
    tr = literal_trace(jax.random.key(0), model, obs, buf0, ilen0, cxs, cls)
    assert decoded_intended(tr, len(obs))[2] == obs[2]           # starts literal ("too")
    tr = rejuv_sweep(jax.random.key(1), tr, buf0, ilen0, cxs, cls, n_sweeps=5)
    assert int(tr.get_choices()["x2"]) == to_id, surface(int(tr.get_choices()["x2"]))


def test_suffix_participates_in_reanalysis():
    """The MH weight to flip the 'too' word is larger with the later context present than without."""
    def mean_flip_weight(words, k):
        obs = _ids(*words)
        model, obs, buf0, ilen0, cxs, cls = chain_inputs(obs)
        tr = literal_trace(jax.random.key(0), model, obs, buf0, ilen0, cxs, cls)
        key, ws = jax.random.key(7), []
        for _ in range(15):
            key, k1 = jax.random.split(key)
            req = flip_request(tr, k, buf0, ilen0, cxs, cls)
            _, w, _, _ = req.edit(k1, tr, genjax.Diff.no_change(tr.get_args()))
            ws.append(float(w))
        return sum(ws) / len(ws)

    w_short = mean_flip_weight([" he", " wants", " too"], 2)              # no suffix
    w_full = mean_flip_weight([" he", " wants", " too", " go", " home"], 2)  # suffix present
    assert w_full > w_short + 1.0, (w_short, w_full)


def test_detailed_balance_matches_exact_posterior():
    """MH substitution-flip stationary histogram matches the enumerated exact posterior (2-word)."""
    obs = _ids(" he", " too")
    model, obs, buf0, ilen0, cxs, cls = chain_inputs(obs)
    W = len(obs)

    def cands(t):
        return [(int(cxs[t][i]), float(cls[t][i])) for i in range(cxs.shape[1])
                if float(cls[t][i]) > -1e20]
    logp = {}
    for x0, l0 in cands(0):
        b1 = buf0.at[ilen0].set(x0)
        for x1, l1 in cands(1):
            logp[(x0, x1)] = (float(lm_logp(buf0, ilen0)[x0]) + l0
                              + float(lm_logp(b1, ilen0 + 1)[x1]) + l1)
    Z = float(logsumexp(jnp.array(list(logp.values()))))
    exact = {k: math.exp(v - Z) for k, v in logp.items()}

    tr = literal_trace(jax.random.key(0), model, obs, buf0, ilen0, cxs, cls)
    key, counts = jax.random.key(3), Counter()
    n_sweeps = 400
    for _ in range(n_sweeps):
        key, sk = jax.random.split(key)
        tr = rejuv_sweep(sk, tr, buf0, ilen0, cxs, cls, n_sweeps=1)
        counts[tuple(decoded_intended(tr, W))] += 1
    n = sum(counts.values())
    max_err = max(abs(exact[s] - counts.get(s, 0) / n) for s in exact)
    assert max_err < 0.07, (max_err, {surface(s[0]) + surface(s[1]): (round(exact[s], 3),
                            round(counts.get(s, 0) / n, 3)) for s in exact})


if __name__ == "__main__":
    L.load_model()
    test_reanalysis_flip_too_to()
    print("OK  reanalysis flip too->to")
    test_suffix_participates_in_reanalysis()
    print("OK  suffix participates in the reanalysis weight")
    test_detailed_balance_matches_exact_posterior()
    print("OK  detailed balance: MH histogram matches exact posterior")
