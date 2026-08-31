"""Phase-2 per-word model outputs (planning/NOISY_CHANNEL_HARNESS_IMPLEMENTATION_PLAN.md §3).

Everything here is an OBSERVER of the certified filter: ``pairhmm_smc.run(word_stats=, diag=)``
threads arrays it already computes into these host-side estimators, consumes no extra RNG, and adds
no arithmetic inside the certified jitted code -- ``word_stats=None`` is bit-identical (gated by
``tests/test_word_stats.py``).

Three outputs:

* **Noisy-channel per-observed-word surprisal** (§3.1): ``S_k = −log(Q_k / Q_{k−1})`` where ``Q_k``
  is the total mass of generative paths whose LAST event emitted observed unit ``o_k`` -- by an
  intended word (the diag arc of ``word_dp._word_row_update``) or as a spurious insertion (that
  row's insertion sweep). Each path emits ``o_k`` exactly once, so these events partition the paths;
  deletion-terminated states are excluded (they would double-count the state before the deletion).
  Estimated from ONE run by :class:`PrefixAccumulator`: per SMC step, :func:`emission_masses` (the
  emission-terminated part of the SAME row update the kernel applied) is accumulated PRE-resample
  with the properly-weighted cloud. ``sum(S_k) + S_end == −logZ`` holds by construction.

* **Per-observed-word alignment posteriors / P(error)** (§3.3): :func:`alignment_posteriors`, a
  host-side forward-backward over the final cloud on the UNMASKED lattice with the band applied
  only to row-to-row arcs (mirroring the kernel's sweep-then-mask), giving per-unit
  ``p_copy/p_sub/p_ins`` (sum to 1) and per-gap expected deletions.

* **Per-word rejuvenation statistics** (§3.4): aggregation helpers for the substitution sweep's
  ``(s_new, target)`` and the Gibbs indel move's ``(logits, idx)`` -- quantities the jitted moves
  already compute and previously discarded.

**Plain-English guide for analysis-time readers: ``planning/WORD_STATS.md``.**

Conventions (also emitted as the ``convention`` field): the FORM channel is an unnormalized edit
kernel (sum_o e^{K*d} ~ 1.05-1.4 per word), so ``S_k`` carries a small near-constant offset -- the
same convention ``logZ`` already uses; requires ``lm_temp == 1`` (asserted); ``prefix_logq[k] =
−inf`` means unreachable under the band and must serialize as null, never as infinite surprisal.
"""

import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jlse
from scipy.special import logsumexp as _lse


CONVENTION = {
    "surprisal_nc": "prefix-mass estimator; form channel unnormalized (near-constant per-word "
                    "offset, same convention as logZ); requires lm_temp == 1",
    "unreachable": "prefix_logq[k] = -inf means unreachable under the band; serialize as null, "
                   "never as an infinite surprisal",
}


# --------------------------------------------------------------------------------------------------
# §3.1 -- the emission-terminated row and the prefix-mass accumulator.
# --------------------------------------------------------------------------------------------------
def _emission_row(log_alpha, emit_col, wdel, wins):
    """Second output of the SAME scan as ``word_dp._word_row_update``: the part of the new row whose
    LAST event emitted the observed unit -- the diag arc (this intended word aligned to observed
    word k), or an insertion into this cell. ``em[0] = -inf`` (cell 0 consumed nothing). Identity
    (tested): ``logaddexp(em[1:], log_alpha[1:] + wdel) == _word_row_update(log_alpha, ...)[1:]`` --
    the full new cell adds only the deletion arc, whose last event emitted nothing."""
    M = emit_col.shape[0]
    diag = log_alpha[0:M] + emit_col
    up = log_alpha[1:M + 1] + wdel
    beta_rest = jlse(jnp.stack([diag, up]), axis=0)
    beta0 = log_alpha[0] + wdel
    beta = jnp.concatenate([beta0[None], beta_rest])
    wins_vec = jnp.broadcast_to(jnp.asarray(wins, beta.dtype), (M,))

    def step(left, x):
        b, w, d = x
        cell = jlse(jnp.stack([b, left + w]))          # identical to _word_row_update's ins_step
        em = jlse(jnp.stack([d, left + w]))            # diag + insertion into this cell
        return cell, em

    _, em_rest = jax.lax.scan(step, beta[0], (beta[1:], wins_vec, diag))
    return jnp.concatenate([jnp.full((1,), -jnp.inf, beta.dtype), em_rest])


@functools.partial(jax.jit, static_argnames=("band",))
def emission_masses(alpha_prev, word_surf_post, n_words_prev, emit_full, copy_mask,
                    lp_copy, lp_sub, wdel_p, wins_p, band=None):
    """Per-particle emission-terminated row (P, M+1) for the word THIS step appended.

    ``alpha_prev``/``n_words_prev`` are the LOOP-TOP (pre-extension) state; ``word_surf_post`` is the
    post-extension ``word_surf`` (the appended word sits at slot ``n_words_prev``). The emission
    column is rebuilt with the SAME formula the kernel used (``emit_full[:, surf]`` plus the
    word-action offset), with the step's per-particle costs, and band-masked AFTER the sweep at
    ``t = n_words_prev + 1`` -- exactly like the kernel. Rows of done / EOS-choosing particles are
    garbage here; the accumulator masks them via ``done_post``."""
    P, Mp1 = alpha_prev.shape
    M = Mp1 - 1
    Wmax = word_surf_post.shape[1]
    surf = word_surf_post[jnp.arange(P), jnp.clip(n_words_prev, 0, Wmax - 1)]
    surf_c = jnp.clip(surf, 0, emit_full.shape[1] - 1)
    cols = (emit_full[:, surf_c].T + lp_sub[:, None]
            + (lp_copy - lp_sub)[:, None] * copy_mask[:, surf_c].T)          # (P, M)
    em = jax.vmap(_emission_row)(alpha_prev, cols, wdel_p, wins_p)
    if band is not None:
        ks = jnp.arange(M + 1)
        em = jnp.where(jnp.abs(ks[None, :] - (n_words_prev + 1)[:, None]) <= band, em, -jnp.inf)
    return em


class PrefixAccumulator:
    """Host-side (float64) accumulator for the prefix masses ``Q_k`` (§3.1).

    ``logq[k]`` estimates ``log Q_k`` RELATIVE to the leading-insertion init's total mass (the same
    relative convention as the filter's ``logZ``); ``logq[0] = 0`` exactly (the empty observed
    prefix), so ``sum(S_k) + S_end == −logZ`` holds by construction. Initialized from the (masked)
    per-particle ``a0p`` -- the s=0 term: the whole prefix explained as leading insertions."""

    def __init__(self, a0p):
        a = np.asarray(a0p, np.float64)
        self.P = a.shape[0]
        self.logq = np.full(a.shape[1], -np.inf)
        self.logq[0] = 0.0
        if a.shape[1] > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = a[:, 1:] - _lse(a, axis=1)[:, None]
                self.logq[1:] = _lse(rel, axis=0) - np.log(self.P)

    def add(self, em, alpha_post, done_post, log_w, logZ_acc):
        """One step's contribution, computed right after ``log_w += incr`` and BEFORE the resample
        test (the particle mapping is the identity there and the cloud is properly weighted for the
        step target). ``g`` is the per-particle emission fraction; done particles (including the
        ones that chose EOS this step -- EOS is not an observed-unit emission) contribute nothing."""
        em = np.asarray(em, np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = em - _lse(np.asarray(alpha_post, np.float64), axis=1)[:, None]
        g[np.asarray(done_post)] = -np.inf
        contrib = float(logZ_acc) + np.asarray(log_w, np.float64)[:, None] - np.log(self.P) + g
        contrib = np.where(np.isnan(contrib), -np.inf, contrib)     # dead particles (-inf - -inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.logq = np.logaddexp(self.logq, _lse(contrib, axis=0))

    def finish(self, logZ_final):
        """-> dict(prefix_logq, surprisal_nc [S_1..S_M], surprisal_end_nc). ``-inf`` prefix cells
        (unreachable under the band) make the adjacent S entries non-finite -- serialize those as
        null downstream, never as infinite surprisal."""
        S = self.logq[:-1] - self.logq[1:]
        return {"prefix_logq": self.logq.copy(), "surprisal_nc": S,
                "surprisal_end_nc": float(self.logq[-1] - logZ_final)}


# --------------------------------------------------------------------------------------------------
# §3.3 -- host-side forward-backward on the final cloud: per-unit alignment posteriors / P(error).
# --------------------------------------------------------------------------------------------------
def alignment_posteriors(state, log_w, diag, check_tol=1e-6):
    """Per-observed-unit action posteriors from the FINAL cloud, exact per particle.

    ``state``/``log_w`` are ``pairhmm_smc.run``'s returns (``log_w`` terminal-corrected); ``diag``
    is the dict ``run(diag={})`` filled (``emit_full, copy_mask, lp_copy, lp_sub, wdel_p, wins_p,
    band, M, obs_words``). Per particle (no dedup -- theta differs per particle after a refresh) a
    float64 forward stores every row and a backward runs on the UNMASKED lattice with the band
    applied only to row-to-row arcs (the insertion sweep is unmasked within a row and the mask is
    applied after it -- mirroring the kernel; a naive masked backward does not sum to 1).

    Returns per-unit ``p_copy/p_sub/p_ins`` (sum to 1; asserted) and per-gap ``e_del_gap``
    (expected deletions at gap k = before unit k+1), averaged over particles with
    ``softmax(log_w)``, plus the cheap POSITIONAL approximation ``p_err_positional``
    (``1 − copy_mask[i, word_surf[p, i]]``, what ``_action_counts`` uses; exact only when no indel
    shifted the alignment) and the per-particle arrays. Particles with zero weight or an impossible
    parse (−inf total) are excluded (their weight is 0 after the terminal correction anyway)."""
    _cb, _cl, n_words, word_len, word_surf, log_alpha, _done = state
    M, band = int(diag["M"]), diag["band"]
    emit_full = np.asarray(diag["emit_full"], np.float64)
    copy_mask = np.asarray(diag["copy_mask"], bool)
    lp_copy = np.asarray(diag["lp_copy"], np.float64)
    lp_sub = np.asarray(diag["lp_sub"], np.float64)
    wdel = np.asarray(diag["wdel_p"], np.float64)                     # (P,)
    wins = np.asarray(diag["wins_p"], np.float64)                    # (P, M)
    if wins.ndim == 1:
        wins = np.broadcast_to(wins[None, :] if wins.shape[0] == M else wins[:, None],
                               (wdel.shape[0], M)).copy()
    ws_ = np.asarray(word_surf)
    n = np.asarray(n_words).astype(int)
    P, Wmax = ws_.shape
    wsc = np.clip(ws_, 0, emit_full.shape[1] - 1)

    # Per-particle emission columns: cols[p, i, :] = channel column of intended word i+1 (slot i).
    flat = wsc.reshape(-1)
    cols = (emit_full[:, flat].T.reshape(P, Wmax, M)
            + lp_sub[:, None, None]
            + (lp_copy - lp_sub)[:, None, None] * copy_mask[:, flat].T.reshape(P, Wmax, M))

    ks = np.arange(M + 1)

    def inband(i):
        return np.ones(M + 1, bool) if band is None else (np.abs(ks - i) <= band)

    err = np.errstate(divide="ignore", invalid="ignore")
    with err:
        # ---- forward: store every row, masked (A) and unmasked-within-row (Au) ----
        a0u = np.concatenate([np.zeros((P, 1)), np.cumsum(wins, axis=1)], axis=1)   # UNMASKED a0
        A = np.full((P, Wmax + 1, M + 1), -np.inf)
        Au = np.full((P, Wmax + 1, M + 1), -np.inf)
        Au[:, 0] = a0u
        A[:, 0] = np.where(inband(0)[None, :], a0u, -np.inf)
        for i in range(1, Wmax + 1):
            col = cols[:, i - 1, :]
            prev = A[:, i - 1]
            au = np.empty((P, M + 1))
            au[:, 0] = prev[:, 0] + wdel
            beta = np.logaddexp(prev[:, :M] + col, prev[:, 1:] + wdel[:, None])
            for k in range(1, M + 1):
                au[:, k] = np.logaddexp(beta[:, k - 1], au[:, k - 1] + wins[:, k - 1])
            Au[:, i] = au
            A[:, i] = np.where(inband(i)[None, :], au, -np.inf)
        total = A[np.arange(P), n, M]                                # per-particle terminal mass

        # ---- backward on the unmasked lattice; band only on row-to-row arcs ----
        base = np.zeros((P, M + 1))
        for k in range(M - 1, -1, -1):
            base[:, k] = base[:, k + 1] + wins[:, k]
        if band is not None:                                          # terminal read requires inband(n, M)
            base[np.abs(M - n) > band] = -np.inf
        Bu = np.full((P, Wmax + 1, M + 1), -np.inf)
        outs = np.full((P, Wmax + 1, M + 1), -np.inf)                # row-exit continuation (i < n)
        for i in range(Wmax, -1, -1):
            if i < Wmax:
                nxt = Bu[:, i + 1]
                col = cols[:, i, :]                                  # word i+1 lives at slot i
                out = np.empty((P, M + 1))
                out[:, :M] = np.logaddexp(nxt[:, 1:] + col, nxt[:, :M] + wdel[:, None])
                out[:, M] = nxt[:, M] + wdel
                out = np.where(inband(i)[None, :], out, -np.inf)     # band on the row-to-row arcs
                bu = np.empty((P, M + 1))
                bu[:, M] = out[:, M]
                for k in range(M - 1, -1, -1):
                    bu[:, k] = np.logaddexp(out[:, k], bu[:, k + 1] + wins[:, k])
            else:
                out = np.full((P, M + 1), -np.inf)
                bu = out
            is_base = (n == i)[:, None]
            Bu[:, i] = np.where(is_base, base, np.where((n > i)[:, None], bu, -np.inf))
            outs[:, i] = np.where((n > i)[:, None], out, -np.inf)

        # ---- arc posteriors (relative to total) ----
        use = np.isfinite(total)
        tot = np.where(use, total, 0.0)[:, None]
        p_copy_pk = np.zeros((P, M)); p_sub_pk = np.zeros((P, M)); p_ins_pk = np.zeros((P, M))
        e_del_pk = np.zeros((P, M + 1))
        for i in range(1, Wmax + 1):
            act = (i <= n)[:, None]
            col = cols[:, i - 1, :]
            pd = np.exp(np.where(act, A[:, i - 1, :M] + col + Bu[:, i, 1:] - tot, -np.inf))
            pu = np.exp(np.where(act, A[:, i - 1, :] + wdel[:, None] + Bu[:, i, :] - tot, -np.inf))
            is_cp = copy_mask[:, wsc[:, i - 1]].T                    # (P, M): unit k a COPY of word i?
            p_copy_pk += pd * is_cp
            p_sub_pk += pd * ~is_cp
            e_del_pk += pu
        for i in range(0, Wmax + 1):
            act = (i <= n)[:, None]
            p_ins_pk += np.exp(np.where(act, Au[:, i, :M] + wins + Bu[:, i, 1:] - tot, -np.inf))

        # ---- invariants (per used particle) ----
        unit_tot = p_copy_pk + p_sub_pk + p_ins_pk                   # each unit emitted exactly once
        bad = use & (np.abs(unit_tot - 1.0).max(axis=1) > max(check_tol, 1e-6))
        assert not bad.any(), \
            f"alignment_posteriors: unit posteriors do not sum to 1 (max err " \
            f"{np.abs(unit_tot[use] - 1.0).max():.2e})"
        for p in np.nonzero(use)[0]:
            for i in range(int(n[p])):                               # every path crosses each row exit once
                cross = _lse(A[p, i] + outs[p, i])
                assert abs(cross - total[p]) < 1e-6 + 1e-9 * abs(total[p]), \
                    f"alignment_posteriors: row-exit invariant broken at particle {p}, row {i}: " \
                    f"{cross:.9f} vs total {total[p]:.9f}"

    # ---- weighted average over the cloud (terminal-corrected weights) ----
    w = np.asarray(jax.nn.softmax(jnp.asarray(log_w)), np.float64)
    w = np.where(use, w, 0.0)
    if w.sum() <= 0:
        raise ValueError("alignment_posteriors: no usable particle (all zero-weight or impossible)")
    w = w / w.sum()

    def avg(a):
        return (w[:, None] * a).sum(axis=0)

    # Positional approximation: slot i <-> unit i; a slot past n_words counts as an error (the unit
    # is positionally unmatched).
    pos_err = np.ones((P, M))
    for m in range(M):
        act = m < n
        pos_err[act, m] = 1.0 - copy_mask[m, wsc[act, m]]

    return {
        "units": list(diag.get("obs_words", [])),
        "p_copy": avg(p_copy_pk), "p_sub": avg(p_sub_pk), "p_ins": avg(p_ins_pk),
        "e_del_gap": avg(e_del_pk),
        "p_err_positional": avg(pos_err),
        "per_particle": {"p_copy": p_copy_pk, "p_sub": p_sub_pk, "p_ins": p_ins_pk,
                         "e_del_gap": e_del_pk, "total": total, "weight": w},
        "n_particles_used": int((w > 0).sum()),
    }


# --------------------------------------------------------------------------------------------------
# §3.4 -- rejuvenation statistics (aggregation of quantities the jitted moves already compute).
# --------------------------------------------------------------------------------------------------
def sub_sweep_event_summary(records):
    """Per-word summary of ONE sweep event's records ``[{w, s_new, target, active}, ...]``:
    ``change_rate`` = fraction of ACTIVE particles whose slot moved off the current word (slot 0 =
    keep), ``stay_prob`` = mean full-conditional probability of keeping it (``softmax(target)[:, 0]``
    over active particles), ``n`` = active count. Slot w <-> observed unit w positionally."""
    out = {}
    for r in records:
        act = np.asarray(r["active"])
        if not act.any():
            continue
        tgt = np.asarray(r["target"], np.float64)[act]
        with np.errstate(invalid="ignore"):
            stay = np.exp(tgt[:, 0] - _lse(tgt, axis=1))
        out[int(r["w"])] = {
            "n": int(act.sum()),
            "change_rate": float((np.asarray(r["s_new"])[act] != 0).mean()),
            "stay_prob": float(np.nanmean(stay)),
        }
    return out


def accumulate_sub_events(acc, records):
    """Fold one event's records into the running per-word accumulator (dict w -> sums)."""
    acc = {} if acc is None else acc
    for r in records:
        act = np.asarray(r["active"])
        if not act.any():
            continue
        w = int(r["w"])
        tgt = np.asarray(r["target"], np.float64)[act]
        with np.errstate(invalid="ignore"):
            stay = np.exp(tgt[:, 0] - _lse(tgt, axis=1))
        a = acc.setdefault(w, {"n_events": 0, "n_active": 0, "n_changed": 0, "stay_sum": 0.0})
        a["n_events"] += 1
        a["n_active"] += int(act.sum())
        a["n_changed"] += int((np.asarray(r["s_new"])[act] != 0).sum())
        a["stay_sum"] += float(np.nansum(stay))
    return acc


def finalize_sub(acc):
    """-> {word slot: {n_events, n_active, n_changed, stay_sum, change_rate, stay_prob}} over all
    accumulated sweep events. The raw counts are kept alongside the rates so a multi-seed merge
    can POOL them exactly (sum counts, recompute rates) instead of averaging rates."""
    return {w: {"n_events": a["n_events"], "n_active": a["n_active"],
                "n_changed": a["n_changed"], "stay_sum": a["stay_sum"],
                "change_rate": a["n_changed"] / max(a["n_active"], 1),
                "stay_prob": a["stay_sum"] / max(a["n_active"], 1)}
            for w, a in sorted(acc.items())}


def indel_summary(records, weights):
    """Per-attempt summary of the Gibbs indel move's conditional and chosen edits.

    Each record carries the host-visible ``logits`` (P, 1 + Wmax*Kc + Wmax) over
    ``{no-op} ∪ {insert c@g} ∪ {delete i}`` and the sampled ``idx``; probabilities are averaged over
    DONE particles with the caller-supplied ``weights`` (softmax(log_w) at the call site), and the
    chosen edits are decoded with the move's own arithmetic. Returns a list, one dict per attempt."""
    out = []
    w_all = np.asarray(weights, np.float64)
    for r in records:
        logits = np.asarray(r["logits"], np.float64)
        idx = np.asarray(r["idx"])
        done = np.asarray(r["done"], bool)
        Kc, Wmax = int(r["Kc"]), int(r["Wmax"])
        n_ins = Wmax * Kc
        w = np.where(done, w_all, 0.0)
        wsum = w.sum()
        w = w / wsum if wsum > 0 else w
        with np.errstate(invalid="ignore"):
            p = np.exp(logits - _lse(logits, axis=1)[:, None])
        p = np.nan_to_num(p, nan=0.0)
        p_ins = p[:, 1:1 + n_ins].reshape(-1, Wmax, Kc)
        is_ins = (idx >= 1) & (idx <= n_ins) & done
        is_del = (idx > n_ins) & done
        ins_flat = np.clip(idx - 1, 0, n_ins - 1)
        g_sel, c_sel = ins_flat // Kc, ins_flat % Kc                # same arithmetic as _indel_apply
        del_i = np.clip(idx - 1 - n_ins, 0, Wmax - 1)
        ins_count = np.zeros(Wmax, int); del_count = np.zeros(Wmax, int)
        cand_count = np.zeros(Kc, int)
        np.add.at(ins_count, g_sel[is_ins], 1)
        np.add.at(cand_count, c_sel[is_ins], 1)
        np.add.at(del_count, del_i[is_del], 1)
        out.append({
            "p_noop": float((w * p[:, 0]).sum()),
            "p_ins_gap": (w[:, None] * p_ins.sum(axis=2)).sum(axis=0),
            "p_del_word": (w[:, None] * p[:, 1 + n_ins:]).sum(axis=0),
            "n_done": int(done.sum()),
            "chosen": {"n_noop": int((done & (idx == 0)).sum()),
                       "n_ins": int(is_ins.sum()), "n_del": int(is_del.sum()),
                       "ins_count_gap": ins_count, "del_count_word": del_count,
                       "ins_count_cand": cand_count},
        })
    return out
