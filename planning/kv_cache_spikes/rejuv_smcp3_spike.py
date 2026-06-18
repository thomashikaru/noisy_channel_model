"""Spike: get the genjax SMCP3 (Rejuvenate) weight for a single-address full-conditional move,
both via the real `Rejuvenate` class (single particle, closure) and via an inline propose/Update/
assess (vmappable over P). Goal: confirm (a) full-conditional weight == 0, (b) inline == class,
(c) it runs under jax.vmap. This de-risks R1's genjax-weight wiring."""
import jax
import jax.numpy as jnp

import genjax
from genjax import ChoiceMap, Update, Diff, StaticRequest
from genjax_port.genjax_model import factor

from genjax._src.inference.requests.rejuvenate import Rejuvenate


def slot_model_fn(K):
    @genjax.gen
    def slot_model(target_lp):
        s = genjax.categorical(jnp.zeros((K,))) @ "slot"
        _ = factor(target_lp[s]) @ "ev"
        return s
    return slot_model


def slot_proposal_fn(K):
    @genjax.gen
    def slot_proposal(target_lp):
        s = genjax.categorical(target_lp) @ "slot"
        return s
    return slot_proposal


def probe_single():
    K = 5
    target_lp = jnp.array([0.5, -1.0, 2.0, 0.3, -0.7])
    model = slot_model_fn(K)
    prop = slot_proposal_fn(K)
    key = jax.random.PRNGKey(0)

    # --- inline propose/Update/assess ---
    s_cur = jnp.int32(2)
    k1, k2, k3 = jax.random.split(key, 3)
    tr, _ = model.importance(k1, ChoiceMap.d({"slot": s_cur, "ev": jnp.float32(0.0)}), (target_lp,))
    print("trace score:", float(tr.get_score()))

    proposed, fwd, _ = prop.propose(k2, (target_lp,))
    print("proposed chm:", proposed, "fwd:", float(fwd))

    argdiffs = (Diff.no_change(target_lp),)
    new_tr, w, retdiff, bwd_req = Update(proposed).edit(k3, tr, argdiffs)
    bwd_chm = bwd_req.constraint
    print("bwd_chm:", bwd_chm)
    bwd, _ = prop.assess(bwd_chm, (target_lp,))
    weight = w + bwd - fwd
    s_new = new_tr.get_choices()["slot"]
    print(f"inline: s_new={int(s_new)} w={float(w):.4f} fwd={float(fwd):.4f} "
          f"bwd={float(bwd):.4f} weight={float(weight):.6f}")

    # --- real Rejuvenate class (single particle, closure over target_lp) ---
    try:
        req = StaticRequest({"slot": Rejuvenate(prop, lambda chm: (target_lp,))})
        ntr, rw, _, _ = req.edit(key, tr, (Diff.no_change(target_lp),))
        print(f"Rejuvenate class: s_new={int(ntr.get_choices()['slot'])} weight={float(rw):.6f}")
    except Exception as e:
        print(f"Rejuvenate class FAILED (proposal re-addresses 'slot'): {type(e).__name__}: {e}")


def probe_vmap():
    K = 5
    P = 4
    model = slot_model_fn(K)
    prop = slot_proposal_fn(K)
    target_lp = jax.random.normal(jax.random.PRNGKey(1), (P, K))
    s_cur = jnp.arange(P) % K
    keys = jax.random.split(jax.random.PRNGKey(2), P)

    def move(key, target_lp, s_cur):
        k1, k2, k3 = jax.random.split(key, 3)
        tr, _ = model.importance(k1, ChoiceMap.d({"slot": s_cur, "ev": jnp.float32(0.0)}),
                                 (target_lp,))
        proposed, fwd, _ = prop.propose(k2, (target_lp,))
        new_tr, w, _, bwd_req = Update(proposed).edit(k3, tr, (Diff.no_change(target_lp),))
        bwd, _ = prop.assess(bwd_req.constraint, (target_lp,))
        return new_tr.get_choices()["slot"], w + bwd - fwd

    s_new, weight = jax.jit(jax.vmap(move))(keys, target_lp, s_cur)
    print("vmap s_new:", s_new, "weight:", weight)
    print("max |weight|:", float(jnp.max(jnp.abs(weight))))


if __name__ == "__main__":
    print("=== single ===")
    probe_single()
    print("\n=== vmap ===")
    probe_vmap()
