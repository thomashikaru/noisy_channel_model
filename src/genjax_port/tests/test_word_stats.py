"""Phase-2 word_stats gates (planning/NOISY_CHANNEL_HARNESS_IMPLEMENTATION_PLAN.md §3.6).

Certifies the per-word output hooks on the toy, against exact enumeration / brute force:

1. ``_emission_row`` is the emission-terminated part of ``_word_row_update`` (algebraic identity).
2. The one-run prefix-mass estimator reproduces the EXACT per-prefix masses (and S_end) on the
   banded toy, at the logZ gate's tolerance.
3. ``alignment_posteriors``'s forward-backward matches a brute-force enumeration of alignment
   paths (band=None and band=1; the band checked at end-of-row exactly as the kernel), on both
   the char-copy and the word-action channel; its internal invariants are exercised throughout.
4. The hooks are PURE OBSERVERS: ``word_stats=``/``diag=`` runs are bit-identical to hook-less
   runs for the char_copy/off, word-action gibbs, and gibbs+bd configs (this also covers the
   extra return values threaded out of the jitted rejuvenation closures).

Run as a test:  pytest src/genjax_port/tests/test_word_stats.py
"""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp as slse

from genjax_port import pairhmm_smc, word_stats
from genjax_port.word_dp import _word_row_update
from genjax_port.tests.test_pairhmm_exact import (_toy_model, _emit_table, _peaked, _bigram_table,
                                                  _wa_emit_copymask_costs, _WA_THETA, WDEL, WINS)
from genjax_port.tests.toy_bigram import BOS, EOS
from genjax_port.tests.toy_vocab import V, WORD2IDX

_OBS = "teh cat sat"


# --------------------------------------------------------------------------------------------------
# 1. Row identity: em ++ deletion arc == the full row update.
# --------------------------------------------------------------------------------------------------
def test_emission_row_identity():
    """logaddexp(em[1:], alpha[1:] + wdel) == _word_row_update(alpha, ...)[1:] on random inputs
    (including -inf band-masked cells), and em[0] == -inf."""
    rng = np.random.default_rng(0)
    for M in (1, 3, 6):
        for trial in range(6):
            alpha = rng.normal(size=M + 1) * 3.0
            if trial >= 3:                                   # band-masked-looking inputs
                alpha[rng.random(M + 1) < 0.4] = -np.inf
            col = rng.normal(size=M) * 2.0
            wdel = float(rng.normal())
            wins = rng.normal(size=M)
            a, c, w = jnp.asarray(alpha), jnp.asarray(col), jnp.asarray(wins)
            em = np.asarray(word_stats._emission_row(a, c, jnp.float32(wdel), w), np.float64)
            new = np.asarray(_word_row_update(a, c, jnp.float32(wdel), w), np.float64)
            assert em[0] == -np.inf
            lhs = np.logaddexp(em[1:], alpha[1:] + wdel)
            assert np.allclose(lhs, new[1:], atol=1e-4), \
                f"M={M} trial={trial}: {lhs} vs {new[1:]}"


# --------------------------------------------------------------------------------------------------
# 2. Exact prefix masses vs the SMC estimator on the banded toy.
# --------------------------------------------------------------------------------------------------
def _mask_np(a, t, band):
    if band is None:
        return a
    ks = jnp.arange(a.shape[0])
    return jnp.where(jnp.abs(ks - t) <= band, a, -jnp.inf)


def _exact_prefix_masses(lm, obs, band, Lmax):
    """Exact (log Q_k − a0_const, logZ − a0_const) by enumerating intended PREFIXES up to Lmax:
    Q_k = a0[k] (leading insertions only) + sum over prefixes x_{1:n} of
    LM_prefix(x) * em_n(x)[k], with the LM prefix probability carrying NO EOS term and every row
    band-masked exactly as the kernel. With band=1 and Lmax = M+1 every contributing row is
    enumerated (row n reaches only k in [n−band, n+band])."""
    M = len(obs.split())
    emit = _emit_table(obs)
    lb = _bigram_table(lm)
    a0 = _mask_np(jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.full((M,), WINS))]), 0, band)
    logQ = np.asarray(a0, np.float64).copy()
    logQ[0] = -np.inf                                        # row 0 emits nothing into k=0
    logZ_terms = [float(lb[BOS, EOS]) + float(a0[M])]        # the n=0 full sentence
    for n in range(1, Lmax + 1):
        seqs = jnp.asarray(list(itertools.product(range(V), repeat=n)), jnp.int32).reshape(-1, n)
        frm = jnp.concatenate([jnp.full((seqs.shape[0], 1), BOS, jnp.int32), seqs[:, :-1]], axis=1)
        lm_prefix = jnp.sum(lb[frm, seqs], axis=1)           # NO EOS term

        def last_row(seq):
            alpha = a0
            for i in range(n - 1):
                alpha = _mask_np(_word_row_update(alpha, emit[:, seq[i]], WDEL, WINS), i + 1, band)
            em = _mask_np(word_stats._emission_row(alpha, emit[:, seq[n - 1]], WDEL, WINS), n, band)
            alpha_n = _mask_np(_word_row_update(alpha, emit[:, seq[n - 1]], WDEL, WINS), n, band)
            return em, alpha_n

        em, alpha_n = jax.vmap(last_row)(seqs)
        lmp = np.asarray(lm_prefix, np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            contrib = lmp[:, None] + np.asarray(em, np.float64)
            contrib = np.where(np.isnan(contrib), -np.inf, contrib)
            logQ = np.logaddexp(logQ, slse(contrib, axis=0))
            eos_lp = np.asarray(lb[seqs[:, -1], EOS], np.float64)
            fin = lmp + eos_lp + np.asarray(alpha_n, np.float64)[:, M]
            logZ_terms.append(float(slse(np.where(np.isnan(fin), -np.inf, fin))))
    a0c = float(slse(np.asarray(a0, np.float64)))
    return logQ - a0c, float(slse(np.asarray(logZ_terms))) - a0c


def test_prefix_masses_match_exact():
    """Mean over 4 seeds of the one-run estimator's prefix_logq[k] matches the exact banded value
    at the logZ gate's tolerance (P=6000, band=1); so does S_end; and sum(S)+S_end == −logZ holds
    per run by construction."""
    lm = _peaked()
    model = _toy_model(lm)
    M = len(_OBS.split())
    exact_logq, exact_logZ = _exact_prefix_masses(lm, _OBS, band=1, Lmax=M + 1)
    logqs, ends = [], []
    for s in range(4):
        ws = {}
        _st, _lw, logZ, _sl = pairhmm_smc.run(_OBS, jax.random.PRNGKey(s), model, P=6000,
                                              wdel=WDEL, wins=WINS, band=1, word_stats=ws)
        assert abs(float(np.sum(ws["surprisal_nc"])) + ws["surprisal_end_nc"] + logZ) < 1e-5, \
            "sum(S_k) + S_end != -logZ (must hold by construction)"
        logqs.append(ws["prefix_logq"])
        ends.append(ws["surprisal_end_nc"])
    mean_logq = np.mean(np.stack(logqs), axis=0)
    for k in range(1, M + 1):
        assert abs(mean_logq[k] - exact_logq[k]) < 0.08, \
            f"prefix_logq[{k}] {mean_logq[k]:.4f} != exact {exact_logq[k]:.4f}"
    exact_end = exact_logq[M] - exact_logZ
    assert abs(float(np.mean(ends)) - exact_end) < 0.08, \
        f"S_end {np.mean(ends):.4f} != exact {exact_end:.4f}"


# --------------------------------------------------------------------------------------------------
# 3. alignment_posteriors vs brute-force alignment enumeration.
# --------------------------------------------------------------------------------------------------
def _brute_arc_posteriors(cols, is_copy, wdel, wins, n, M, band):
    """Enumerate every alignment path from (0,0) to (n,M) -- j insertions within a row, then a
    diag/del exit gated by inband(row, k) (the band checked at END-OF-ROW, exactly as the kernel;
    termination requires inband(n, M)) -- and read off exact per-unit p_copy/p_sub/p_ins and
    per-gap E[deletions]. cols[i][k] = emission of o_{k+1} by word i+1; is_copy[i][k] says whether
    that diag arc is a (case-insensitive) COPY."""
    p_copy = np.zeros(M); p_sub = np.zeros(M); p_ins = np.zeros(M)
    e_del = np.zeros(M + 1)
    total = 0.0

    def inband(i, k):
        return band is None or abs(k - i) <= band

    def record(events, logw):
        nonlocal total
        w = math.exp(logw)
        total += w
        for ev in events:
            kind, i, k = ev
            if kind == "diag":
                (p_copy if is_copy[i - 1][k - 1] else p_sub)[k - 1] += w
            elif kind == "del":
                e_del[k] += w
            else:
                p_ins[k - 1] += w

    def go(i, k, logw, events):
        while True:
            if i == n:
                if k == M and inband(n, M):
                    record(events, logw)
            elif inband(i, k):
                if k < M:
                    go(i + 1, k + 1, logw + cols[i][k], events + [("diag", i + 1, k + 1)])
                go(i + 1, k, logw + wdel, events + [("del", i + 1, k)])
            if k == M:
                return
            logw += wins[k]
            events = events + [("ins", i, k + 1)]
            k += 1

    go(0, 0, 0.0, [])
    assert total > 0, "brute force: no valid alignment path"
    return p_copy / total, p_sub / total, p_ins / total, e_del / total, math.log(total)


def _posterior_case(obs, sentences, weights, band, wa):
    """Run alignment_posteriors on a hand-built cloud (one intended sentence per particle) and
    the matching brute-force enumeration; compare per-particle arrays and the weighted average."""
    lm = _peaked()
    M = len(obs.split())
    if wa:
        _model, emit_aug, copy_mask_f, (lp_c, lp_s, wd, wi_) = _wa_emit_copymask_costs(
            obs, _WA_THETA, lm)
        emit = np.asarray(emit_aug, np.float64)
        cm = np.asarray(copy_mask_f) > 0.5
        lp_copy, lp_sub, wdel, wins_s = float(lp_c), float(lp_s), float(wd), float(wi_)
    else:
        emit = np.asarray(_emit_table(obs), np.float64)
        cm = np.zeros((M, V), bool)
        lp_copy = lp_sub = 0.0
        wdel, wins_s = float(WDEL), float(WINS)

    P = len(sentences)
    Wmax = max(len(s.split()) for s in sentences) + 1
    word_surf = np.zeros((P, Wmax), np.int32)
    word_len = np.zeros((P, Wmax), np.int32)
    n_words = np.zeros(P, np.int32)
    for p, s in enumerate(sentences):
        ids = [WORD2IDX[w] for w in s.split()]
        n_words[p] = len(ids)
        word_len[p, :len(ids)] = 1
        word_surf[p, :len(ids)] = ids
    state = (None, None, jnp.asarray(n_words), jnp.asarray(word_len), jnp.asarray(word_surf),
             None, None)
    diag = {"emit_full": emit, "copy_mask": cm, "lp_copy": np.full(P, lp_copy),
            "lp_sub": np.full(P, lp_sub), "wdel_p": np.full(P, wdel),
            "wins_p": np.full((P, M), wins_s), "band": band, "M": M, "obs_words": obs.split()}
    post = word_stats.alignment_posteriors(state, jnp.log(jnp.asarray(weights)), diag)

    exp_copy = np.zeros((P, M)); exp_sub = np.zeros((P, M)); exp_ins = np.zeros((P, M))
    exp_del = np.zeros((P, M + 1))
    for p, s in enumerate(sentences):
        ids = [WORD2IDX[w] for w in s.split()]
        cols = [emit[:, t] + lp_sub + (lp_copy - lp_sub) * cm[:, t] for t in ids]
        isc = [cm[:, t] for t in ids]
        pc, ps, pi, ed, _ = _brute_arc_posteriors(
            [c.tolist() for c in cols], [r.tolist() for r in isc], wdel,
            [wins_s] * M, len(ids), M, band)
        exp_copy[p], exp_sub[p], exp_ins[p], exp_del[p] = pc, ps, pi, ed
    pp = post["per_particle"]
    assert np.allclose(pp["p_copy"], exp_copy, atol=1e-8), f"p_copy: {pp['p_copy']} vs {exp_copy}"
    assert np.allclose(pp["p_sub"], exp_sub, atol=1e-8)
    assert np.allclose(pp["p_ins"], exp_ins, atol=1e-8)
    assert np.allclose(pp["e_del_gap"], exp_del, atol=1e-8)
    wn = np.asarray(weights, np.float64); wn = wn / wn.sum()
    assert np.allclose(post["p_copy"], (wn[:, None] * exp_copy).sum(axis=0), atol=1e-8)
    assert np.allclose(post["e_del_gap"], (wn[:, None] * exp_del).sum(axis=0), atol=1e-8)


def test_alignment_posteriors_match_brute_force():
    """Forward-backward == brute-force path enumeration for clouds mixing n < M, n == M, n > M
    particles, unbanded and banded, char-copy and word-action (the copy/sub split rides the same
    copy_mask as the kernel). The internal sum-to-1 and row-exit invariants run on every case."""
    sentences = ["the cat sat", "the cat sat on", "the cat"]      # n = M, M+1, M-1
    weights = [0.5, 0.3, 0.2]
    for band in (None, 1):
        for wa in (False, True):
            _posterior_case(_OBS, sentences, weights, band, wa)


# --------------------------------------------------------------------------------------------------
# 4. Hooks are pure observers: bit-identical with and without word_stats/diag.
# --------------------------------------------------------------------------------------------------
def _run_pair(channel_kwargs, rejuv, P, seed=0):
    lm = _peaked()
    model = _toy_model(lm)
    base = dict(P=P, wdel=WDEL, wins=WINS, band=2, rejuv=rejuv, **channel_kwargs)
    st1, lw1, z1, _ = pairhmm_smc.run(_OBS, jax.random.PRNGKey(seed), model, **base)
    ws, dg = {}, {}
    st2, lw2, z2, _ = pairhmm_smc.run(_OBS, jax.random.PRNGKey(seed), model,
                                      word_stats=ws, diag=dg, **base)
    for i, (a, b) in enumerate(zip(st1, st2)):
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"state leaf {i} differs ({rejuv})"
    assert np.array_equal(np.asarray(lw1), np.asarray(lw2)), f"log_w differs ({rejuv})"
    assert z1 == z2, f"logZ differs ({rejuv}): {z1} vs {z2}"
    assert abs(float(np.sum(ws["surprisal_nc"])) + ws["surprisal_end_nc"] + z2) < 1e-5
    assert dg["M"] == len(_OBS.split()) and dg["emit_full"] is not None
    return ws, dg


def test_hooks_are_pure_observers():
    """word_stats=/diag= change NOTHING in the certified outputs (state, log_w, logZ) -- char_copy
    off, word-action gibbs, and word-action gibbs+bd (which also exercises the extra outputs now
    threaded out of the jitted sweep and indel-move closures)."""
    _run_pair(dict(), "off", 512)
    _run_pair(dict(action_alpha=[8.5, 0.5, 0.5, 0.5]), "gibbs", 512)
    ws, _dg = _run_pair(dict(action_alpha=[8.5, 0.5, 0.5, 0.5], bd_mode="gibbs"), "gibbs+bd", 512)
    assert "rejuv" in ws and "indel" in ws["rejuv"], \
        "gibbs+bd with word_stats must record the indel-move statistics"
    ind = ws["rejuv"]["indel"][0]
    assert 0.0 <= ind["p_noop"] <= 1.0 + 1e-9
    assert ind["chosen"]["n_noop"] + ind["chosen"]["n_ins"] + ind["chosen"]["n_del"] == ind["n_done"]


if __name__ == "__main__":
    test_emission_row_identity()
    test_prefix_masses_match_exact()
    test_alignment_posteriors_match_brute_force()
    test_hooks_are_pure_observers()
    print("word_stats gates: 4/4 PASS")
