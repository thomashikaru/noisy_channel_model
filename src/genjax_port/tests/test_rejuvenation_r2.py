"""M5/R2 tests: add/delete reversible-jump rejuvenation (trans-dimensional reanalysis).

Three checks, mirroring the R1 suite (``test_rejuvenation.py``):
  - reanalysis: a literal "he wants go home" (the word "to" omitted) is repaired by an add sweep,
    which inserts "to" at the gap before "go" -> "he wants to go home";
  - the suffix participates: the add move's MH weight at the gap is larger when the later word is in
    the trace (later context votes via the Update's suffix re-scoring);
  - detailed balance: on a 1-word toy with one toggleable deletion gap, the MH stationary histogram
    over the reachable states {off} u {on with a top-k omitted token} matches the enumerated exact
    posterior -- the trans-dimensional add/delete move provably samples the posterior (docs/model.tex
    Thm 2 / R2), empirically.

LM-dependent; validated on pythia-70m. Run as a script (or via tests/run.py).
"""

import math
from collections import Counter

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import genjax
from genjax import ChoiceMap as C

from src.genjax_port import lm_penzai as L
from src.genjax_port.lm_genjax import lm_logp
from src.genjax_port.rejuvenation_r2 import (
    gap_chain_inputs, literal_trace, add_delete_step, add_delete_sweep,
    vmapped_add_delete, gap_config, decoded_gap_chain,
)
from src.genjax_port.tokenizer import encode, surface


def _ids(*words):
    return [encode(w)[0] for w in words]


def test_reanalysis_add_recovers_omitted():
    """A literal 'he wants go home' (the word 'to' dropped) is repaired by an add sweep."""
    obs = _ids(" he", " wants", " go", " home")        # intended: "he wants TO go home"
    to_id = encode(" to")[0]
    model, obs, buf0, ilen0, cxs, cls = gap_chain_inputs(obs)
    W = len(obs)
    tr = literal_trace(jax.random.key(0), model, obs, buf0, ilen0, cxs, cls)
    assert all(not on for on, _ in gap_config(tr, W))                # starts with no deletions
    tr = add_delete_sweep(jax.random.key(5), tr, buf0, ilen0, cxs, cls, obs, n_sweeps=8)
    cfg = gap_config(tr, W)
    assert cfg[2] == (True, to_id), [surface(i) for i in decoded_gap_chain(tr, W)]   # gap before "go" = "to"


def test_suffix_participates_in_add_weight():
    """The add move's weight at the gap before 'go' is larger with the trailing 'home' present."""
    def mean_add_weight(words, k):
        obs = _ids(*words)
        model, obs, buf0, ilen0, cxs, cls = gap_chain_inputs(obs)
        tr = literal_trace(jax.random.key(0), model, obs, buf0, ilen0, cxs, cls)
        key, ws = jax.random.key(7), []
        for _ in range(20):
            key, sk = jax.random.split(key)
            _, w, _ = add_delete_step(sk, tr, k, buf0, ilen0, cxs, cls, obs)
            ws.append(float(w))
        return sum(ws) / len(ws)

    w_short = mean_add_weight([" he", " wants", " go"], 2)            # no suffix after "go"
    w_full = mean_add_weight([" he", " wants", " go", " home"], 2)    # "home" suffix present
    assert w_full > w_short + 0.5, (w_short, w_full)


def test_detailed_balance_add_delete():
    """MH add/delete histogram matches the enumerated exact posterior over the reachable states."""
    K = 4                                       # toy proposal support (and # of reachable on-states)
    p_del = 0.3                                 # inflated so on-states carry real mass (DB is p_del-free)
    obs = _ids(" go")
    model, obs, buf0, ilen0, cxs, cls = gap_chain_inputs(obs, p_del=p_del)
    W = 1
    args = (buf0, ilen0, cxs, cls)
    x0, o0 = obs[0], obs[0]

    # reachable on-states: the top-K LM tokens at the gap context (q's support)
    topk = [int(i) for i in jax.lax.top_k(lm_logp(buf0, ilen0), K)[1]]

    def state_logp(del0, xd):
        chm = C.d({"del0": jnp.bool_(del0), "gap0": C.d({"xd": jnp.int32(xd)}),
                   "x0": jnp.int32(x0), "o0": jnp.int32(o0)})
        tr, _ = model.importance(jax.random.key(0), chm, args)
        return float(tr.get_score())

    states = [("off", None)] + [("on", xd) for xd in topk]
    logps = [state_logp(False, x0)] + [state_logp(True, xd) for xd in topk]
    Z = float(logsumexp(jnp.array(logps)))
    exact = {s: math.exp(lp - Z) for s, lp in zip(states, logps)}

    def state_of(tr):
        on, tok = gap_config(tr, W)[0]
        return ("on", int(tok)) if on else ("off", None)

    tr = literal_trace(jax.random.key(0), model, obs, buf0, ilen0, cxs, cls)
    key, counts = jax.random.key(3), Counter()
    n_steps = 1500
    for _ in range(n_steps):
        key, sk = jax.random.split(key)
        tr = add_delete_sweep(sk, tr, buf0, ilen0, cxs, cls, obs, positions=[0], lookahead_k=K)
        counts[state_of(tr)] += 1
    n = sum(counts.values())
    max_err = max(abs(exact[s] - counts.get(s, 0) / n) for s in exact)
    assert max_err < 0.07, (max_err, {(s[0], surface(s[1]) if s[1] else "-"):
                            (round(exact[s], 3), round(counts.get(s, 0) / n, 3)) for s in exact})


def test_vmapped_move_matches_loop_with_per_particle_masks():
    """The vmapped add/delete move (Phase 2) == the per-particle loop, with DIFFERENT deletion
    configs per particle -- proving the MaskCombinator vmaps over per-particle del{t} flags and the
    LM forward batches. Parity to XLA float-tiling noise (~1e-2), accepts identical."""
    obs = _ids(" he", " wants", " go", " home")
    to_id = encode(" to")[0]
    model, obs, buf0, ilen0, cxs, cls = gap_chain_inputs(obs)
    W = len(obs)
    args = (buf0, ilen0, cxs, cls)
    P = 6

    def particle(i):                               # particle i: one gap on, at a per-particle slot
        on = i % W
        d = {}
        for t in range(W):
            d[f"del{t}"] = jnp.bool_(t == on)
            d[f"gap{t}"] = C.d({"xd": jnp.int32(to_id)})
            d[f"x{t}"] = jnp.int32(obs[t])
            d[f"o{t}"] = jnp.int32(obs[t])
        tr, _ = model.importance(jax.random.key(100 + i), C.d(d), args)
        return tr

    trs = [particle(i) for i in range(P)]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trs)
    keys = jax.random.split(jax.random.key(0), P)
    k = 1

    _, vW, vacc = vmapped_add_delete(keys, batched, k, buf0, ilen0, cxs, cls, obs)
    loop = [add_delete_step(keys[i], trs[i], k, buf0, ilen0, cxs, cls, obs) for i in range(P)]
    lW = jnp.stack([o[1] for o in loop])
    lacc = jnp.stack([o[2] for o in loop])
    finite = jnp.isfinite(vW) & jnp.isfinite(lW)
    assert jnp.all(vacc == lacc), (vacc, lacc)
    assert jnp.all(jnp.abs(jnp.where(finite, vW - lW, 0.0)) < 1e-2), (vW, lW)
    assert jnp.all(jnp.isfinite(vW) == jnp.isfinite(lW)), (vW, lW)   # -inf pattern matches


if __name__ == "__main__":
    L.load_model()
    test_reanalysis_add_recovers_omitted()
    print("OK  reanalysis: add recovers omitted 'to' -> he wants to go home")
    test_suffix_participates_in_add_weight()
    print("OK  suffix participates in the add weight")
    test_detailed_balance_add_delete()
    print("OK  detailed balance: add/delete histogram matches exact posterior")
    test_vmapped_move_matches_loop_with_per_particle_masks()
    print("OK  vmapped move == per-particle loop (per-particle masks, batched forward)")
