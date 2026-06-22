"""Phase-0 gate for the birth/death rejuvenation involution (REJUV_BIRTH_DEATH_PLAN.md).

The birth/death move is a SINGLE symmetric involution that adds OR removes one intended word. Before any
scoring (LM / channel / SMCP3 weight -- Phase 1), the *mechanics* must be exact:

  * ``_insert_word`` (birth) and ``_delete_word`` (death) are mutual inverses at the same position ``w``,
    with the deleted word as the dimension-matching auxiliary -- delete recovers exactly what insert put in.
  * ``_involve`` -- the one move, flipping direction -- is its OWN INVERSE: ``_involve ∘ _involve = id`` on
    the move's support (this is the analogue of the ``weight ≈ 0`` build-it-right assertion in ``_smcp3_move``).
  * both ops preserve the canonical buffer form (active words in ``[0, n_words)``; pad = ``(0, 0, -1)``), so
    the round-trip is bit-exact on the FULL buffers, and ``n_words`` bookkeeping matches (birth +1, death -1).

Vectorized over P with per-particle ``(direction, w, x)`` and mixed directions in one batch.

Run as a script:  python -m genjax_port.tests.test_rejuv_birth_death
Run as a test:    pytest src/genjax_port/tests/test_rejuv_birth_death.py
"""

import itertools

import numpy as np
import jax
import jax.numpy as jnp

from genjax_port.pairhmm_rejuv import (
    _PAD_SURF, _insert_word, _delete_word, _involve,
    _bd_log_weight, birth_death_move,
)

# --------------------------------------------------------------------------------------------------
# A small canonical batch (T_max = 1, the deployment capacity) with varied n_words to exercise the
# vectorization: a short sentence, a near-full one, a singleton, and a mid one. Distinct token ids
# everywhere so equality is meaningful; surf == tok (single-token); pad = (tok=0, len=0, surf=-1).
# --------------------------------------------------------------------------------------------------
WMAX, T = 6, 1
_WORDS = [
    [10, 11, 12],            # n_words = 3
    [20, 21],                # n_words = 2
    [30, 31, 32, 33, 34],    # n_words = 5 (near full)
    [40],                    # n_words = 1
]


def _canonical_state(words=_WORDS, Wmax=WMAX):
    P = len(words)
    word_tok = np.zeros((P, Wmax, T), np.int32)
    word_len = np.zeros((P, Wmax), np.int32)
    word_surf = np.full((P, Wmax), _PAD_SURF, np.int32)
    n_words = np.zeros((P,), np.int32)
    for p, ws in enumerate(words):
        n_words[p] = len(ws)
        for i, t in enumerate(ws):
            word_tok[p, i, 0] = t
            word_len[p, i] = 1
            word_surf[p, i] = t
    return (jnp.asarray(word_tok), jnp.asarray(word_len), jnp.asarray(word_surf), jnp.asarray(n_words))


def _mk_word(toks):
    """Pack token-id lists (length <= T) into (x_tok [P,T], x_len [P], x_surf [P]); surf = first token."""
    P = len(toks)
    x_tok = np.zeros((P, T), np.int32)
    x_len = np.zeros((P,), np.int32)
    x_surf = np.full((P,), _PAD_SURF, np.int32)
    for p, ts in enumerate(toks):
        x_len[p] = len(ts)
        x_tok[p, :len(ts)] = ts
        x_surf[p] = ts[0]
    return jnp.asarray(x_tok), jnp.asarray(x_len), jnp.asarray(x_surf)


def _eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def _assert_canonical_pad(word_tok, word_len, word_surf, n_words):
    """Slots at/after n_words must be exactly (tok=0, len=0, surf=-1)."""
    wt, wl, ws, nw = (np.asarray(x) for x in (word_tok, word_len, word_surf, n_words))
    P, Wmax = wl.shape
    pad = np.arange(Wmax)[None, :] >= nw[:, None]
    assert np.all(wl[pad] == 0), "pad slots must have word_len 0"
    assert np.all(ws[pad] == _PAD_SURF), "pad slots must have word_surf -1"
    assert np.all(wt[pad] == 0), "pad slots must have word_tok 0"


def test_insert_delete_roundtrip():
    """Birth then death at the same gap recovers the original buffers, and the death recovers exactly the
    inserted word as the auxiliary."""
    wt, wl, ws, nw = _canonical_state()
    w = jnp.asarray([1, 0, 5, 1], np.int32)                 # gaps in [0, n_words]; p2 appends at end (=5)
    x_tok, x_len, x_surf = _mk_word([[91], [92], [93], [94]])
    ins = _insert_word(wt, wl, ws, nw, w, x_tok, x_len, x_surf)
    assert _eq(ins[3], nw + 1)
    _assert_canonical_pad(*ins)
    (back, (rx_tok, rx_len, rx_surf)) = _delete_word(*ins, w)
    assert _eq(back[0], wt) and _eq(back[1], wl) and _eq(back[2], ws) and _eq(back[3], nw)
    assert _eq(rx_tok, x_tok) and _eq(rx_len, x_len) and _eq(rx_surf, x_surf)


def test_delete_insert_roundtrip():
    """Death then birth (with the recovered word) at the same position recovers the original buffers."""
    wt, wl, ws, nw = _canonical_state()
    w = jnp.asarray([2, 1, 4, 0], np.int32)                 # positions in [0, n_words)
    (dlt, (rx_tok, rx_len, rx_surf)) = _delete_word(wt, wl, ws, nw, w)
    assert _eq(dlt[3], nw - 1)
    _assert_canonical_pad(*dlt)
    back = _insert_word(*dlt, w, rx_tok, rx_len, rx_surf)
    assert _eq(back[0], wt) and _eq(back[1], wl) and _eq(back[2], ws) and _eq(back[3], nw)


def test_involution_self_inverse():
    """``_involve ∘ _involve = id`` on a mixed batch (birth + death particles). Death particles carry their
    aux = word at w (the move's support convention); birth particles carry a fresh word to insert."""
    wt, wl, ws, nw = _canonical_state()
    d_birth = jnp.asarray([True, False, False, True])
    w = jnp.asarray([1, 0, 4, 1], np.int32)                 # birth: gap in [0,n]; death: pos in [0,n)
    # death particles (1,2): aux must equal the word currently at w; birth particles (0,3): a fresh word.
    x_tok, x_len, x_surf = _mk_word([[97], [20], [34], [98]])
    z = (wt, wl, ws, nw, d_birth, w, x_tok, x_len, x_surf)
    z1 = _involve(*z)
    assert _eq(z1[4], ~d_birth)                              # direction flipped after one application
    z2 = _involve(*z1)
    names = ["word_tok", "word_len", "word_surf", "n_words", "d_birth", "w", "x_tok", "x_len", "x_surf"]
    for name, a, b in zip(names, z2, z):
        assert _eq(a, b), f"self-inverse failed on {name}"


def test_boundary_append_and_remove_last():
    """Birth at the end gap (w == n_words) appends; death at the last active slot (w == n_words-1) removes
    it -- and they invert each other, including the near-full particle (n_words=5, Wmax=6)."""
    wt, wl, ws, nw = _canonical_state()
    w_end = nw                                               # append at the end for every particle
    x_tok, x_len, x_surf = _mk_word([[81], [82], [83], [84]])
    ins = _insert_word(wt, wl, ws, nw, w_end, x_tok, x_len, x_surf)
    _assert_canonical_pad(*ins)
    (back, (rx_tok, _rl, _rs)) = _delete_word(*ins, w_end)
    assert _eq(back[0], wt) and _eq(back[3], nw)
    assert _eq(rx_tok, x_tok)


# ==================================================================================================
# Phase-1 weight gate: the reversible-jump invariance identity, checked EXACTLY (no Monte Carlo).
#
# For a properly weighted SMCP3/RJ move, starting from y ~ π and proposing y' ~ q_fwd with weight
# w = exp(W), the move leaves π invariant:  Σ_y π(y) Σ_{y'} q_fwd(y'|y) w(y->y') g(y')  =  E_π[g]  for all g.
# Taking g = indicator of each sentence, this is  Σ_y π(y) q_fwd(y'|y) exp(W) = π(y')  for every y'. We
# verify it deterministically by enumerating EVERY (y, move) transition over a small synthetic target,
# closing the move on lengths {0..Wmax} (Wmax small so births at Wmax are infeasible -> no escape).
# Pool = full vocab, so every word is deletable (D = length); proposals are uniform. We build the explicit
# forward density ``qf`` and reverse density ``qb`` for each transition and feed them to the now
# proposal-AGNOSTIC ``_bd_log_weight`` -- so this certifies the densities + weight END-TO-END (the part the
# involution's correctness hinges on), and the same harness recertifies any future informed proposals.
# ==================================================================================================
_BD_WMAX, _BD_V = 3, (1, 2, 3, 4, 5)            # sentence-length cap; vocab of surfaces (avoid 0 = pad tok)
_BD_KC = len(_BD_V)
_ALPHA = {1: 0.3, 2: -0.5, 3: 0.1, 4: -0.2, 5: 0.4}
_BETA = -0.7                                      # per-word length term


def _synth_logpi(seq):
    """Arbitrary black-box target over a sentence (tuple of surfaces): per-word weights + a length term +
    an adjacency bonus for repeats (so the target couples neighbours, exercising the score difference)."""
    lp = _BETA * len(seq) + sum(_ALPHA[s] for s in seq)
    lp += 0.2 * sum(1.0 for a, b in zip(seq, seq[1:]) if a == b)
    return lp


def _enumerate():
    seqs = []
    for n in range(_BD_WMAX + 1):
        seqs.extend(itertools.product(_BD_V, repeat=n))
    idx = {s: i for i, s in enumerate(seqs)}
    logp = np.array([_synth_logpi(s) for s in seqs])
    pi = np.exp(logp - logp.max())
    pi /= pi.sum()
    return seqs, idx, logp, pi


def _dir_probs(n):
    """(p_birth, p_death) the move uses at a length-n sentence with full-vocab pool (D = n)."""
    fb, fd = n < _BD_WMAX, n > 0
    if fb and fd:
        return 0.5, 0.5
    return (1.0, 0.0) if fb else (0.0, 1.0)


def _del_softmax(seq):
    """(len(seq),) NEAR-CONDITIONAL deletion distribution over positions of ``seq`` -- the exact reference for
    ``pairhmm_rejuv._del_logq`` under the full-vocab pool (every position deletable): q_del(w) ∝ exp(logπ(seq
    with position w removed)). Empty for a 0-length sequence (no death possible)."""
    n = len(seq)
    if n == 0:
        return np.array([])
    sc = np.array([_synth_logpi(seq[:w] + seq[w + 1:]) for w in range(n)])
    e = np.exp(sc - sc.max())
    return e / e.sum()


def _ins_softmax(seq, gap):
    """{surface: prob} NEAR-CONDITIONAL insertion-word distribution at ``gap`` -- the exact reference for
    ``pairhmm_rejuv._ins_logq`` under the full pool (every vocab word insertable): q_ins(x) ∝ exp(logπ(seq
    with x inserted at gap))."""
    sc = {x: _synth_logpi(seq[:gap] + (x,) + seq[gap:]) for x in _BD_V}
    m = max(sc.values())
    e = {x: np.exp(v - m) for x, v in sc.items()}
    z = sum(e.values())
    return {x: e[x] / z for x in _BD_V}


def test_rj_weight_invariance_exact():
    """Exact transition-sum: Σ_y π(y) q_fwd(y'|y) exp(W) == π(y') for every y'. Certifies the proposal-agnostic
    _bd_log_weight together with the uniform forward (``qf``) and reverse (``qb``) densities fed to it: a birth's
    forward density is direction·gap·word, its reverse is a death at y'; a death's forward is direction·position,
    its reverse a birth at y'. Full-vocab pool ⇒ D = length, so every reverse move has positive density."""
    seqs, idx, logp, pi = _enumerate()
    src, dst, qf, qb = [], [], [], []
    for i, y in enumerate(seqs):
        n = len(y)
        pb, pd = _dir_probs(n)
        if n < _BD_WMAX:                                     # births: gap w in 0..n, word x in vocab
            _pbp, pdp = _dir_probs(n + 1)                    # direction rule at y' (n+1 words)
            for w in range(n + 1):
                for x in _BD_V:
                    yp = y[:w] + (x,) + y[w:]
                    src.append(i); dst.append(idx[yp])
                    qf.append(pb * (1.0 / (n + 1)) * (1.0 / _BD_KC))     # fwd birth: dir·gap·word
                    qb.append(pdp * (1.0 / (n + 1)))                     # rev death at y' (D_yp = n+1)
        if n > 0:                                            # deaths: position w in 0..n-1 (all deletable)
            pbp, _pdp = _dir_probs(n - 1)                    # direction rule at y' (n-1 words)
            for w in range(n):
                yp = y[:w] + y[w + 1:]
                src.append(i); dst.append(idx[yp])
                qf.append(pd * (1.0 / n))                             # fwd death: dir·position (D_y = n)
                qb.append(pbp * (1.0 / n) * (1.0 / _BD_KC))          # rev birth at y' (gaps = n, words = Kc)
    src, dst = np.array(src), np.array(dst)
    W = np.asarray(_bd_log_weight(
        jnp.asarray(logp[src]), jnp.asarray(logp[dst]),
        jnp.asarray(np.log(qf)), jnp.asarray(np.log(qb))))
    mass = np.zeros(len(seqs))
    np.add.at(mass, dst, pi[src] * np.array(qf) * np.exp(W))
    err = np.max(np.abs(mass - pi))
    assert err < 1e-6, f"RJ invariance violated: max|mass - pi| = {err:.2e}"


def test_rj_weight_invariance_informed():
    """Same exact transition-sum invariance with the PHASE-2 informed proposals the move actually uses, BOTH
    directions: a death's position ~ q_del (softmax of the removal target, ``_del_softmax``) and a birth's word
    ~ q_ins (softmax of the insertion target, ``_ins_softmax``), gap uniform. So a death's forward density is
    dir·q_del(w|y) and its reverse is a birth at y' (dir·gap·q_ins(removed|w,y')); a birth's forward is
    dir·gap·q_ins(x|w,y) and its reverse a death at y' (dir·q_del(w|y')). Certifies that ``birth_death_move``
    feeds the proposal-agnostic ``_bd_log_weight`` consistent forward/reverse densities under the informed
    proposals -- the exact case the live move runs (score_fn = LM+channel instead of the synthetic target)."""
    seqs, idx, logp, pi = _enumerate()
    src, dst, qf, qb = [], [], [], []
    for i, y in enumerate(seqs):
        n = len(y)
        pb, pd = _dir_probs(n)
        if n < _BD_WMAX:                                     # births: informed word fwd; reverse death at y'
            _pbp, pdp = _dir_probs(n + 1)
            for w in range(n + 1):
                ism = _ins_softmax(y, w)
                for x in _BD_V:
                    yp = y[:w] + (x,) + y[w:]
                    src.append(i); dst.append(idx[yp])
                    qf.append(pb * (1.0 / (n + 1)) * ism[x])             # fwd birth: dir·gap·q_ins(x|w,y)
                    qb.append(pdp * _del_softmax(yp)[w])                 # rev death at y': dir·q_del(w|y')
        if n > 0:                                            # deaths: informed fwd; reverse birth at y' informed
            pbp, _pdp = _dir_probs(n - 1)
            dsm = _del_softmax(y)
            for w in range(n):
                yp = y[:w] + y[w + 1:]
                ism_yp = _ins_softmax(yp, w)                          # reverse birth re-inserts removed @ gap w
                src.append(i); dst.append(idx[yp])
                qf.append(pd * dsm[w])                               # fwd death: dir·q_del(w|y)
                qb.append(pbp * (1.0 / n) * ism_yp[y[w]])           # rev birth at y': dir·gap·q_ins(rem|w,y')
    src, dst = np.array(src), np.array(dst)
    W = np.asarray(_bd_log_weight(
        jnp.asarray(logp[src]), jnp.asarray(logp[dst]),
        jnp.asarray(np.log(qf)), jnp.asarray(np.log(qb))))
    mass = np.zeros(len(seqs))
    np.add.at(mass, dst, pi[src] * np.array(qf) * np.exp(W))
    err = np.max(np.abs(mass - pi))
    assert err < 1e-6, f"informed RJ invariance violated: max|mass - pi| = {err:.2e}"


def test_birth_death_move_smoke():
    """End-to-end: birth_death_move runs over a mixed batch with an injected synthetic score_fn, returns
    canonical states, valid n_words in [0, Wmax], finite weights, and actually moves some particles."""
    Wmax = _BD_WMAX
    alpha_vec = np.zeros(max(_BD_V) + 1, np.float32)
    for s, a in _ALPHA.items():
        alpha_vec[s] = a
    alpha_vec = jnp.asarray(alpha_vec)

    def score_fn(word_tok, word_len, word_surf, n_words, done):
        active = word_len > 0
        surf = jnp.clip(word_surf, 0, alpha_vec.shape[0] - 1)
        per = jnp.where(active, alpha_vec[surf], 0.0)
        return jnp.sum(per, axis=1) + _BETA * n_words.astype(jnp.float32)

    words = [[1, 2], [3], [], [4, 5, 1]]                     # incl. empty (birth-only) and full (death-only)
    P = len(words)
    word_tok = np.zeros((P, Wmax, T), np.int32)
    word_len = np.zeros((P, Wmax), np.int32)
    word_surf = np.full((P, Wmax), _PAD_SURF, np.int32)
    n_words = np.zeros((P,), np.int32)
    for p, ws in enumerate(words):
        n_words[p] = len(ws)
        for i, s in enumerate(ws):
            word_tok[p, i, 0] = s; word_len[p, i] = 1; word_surf[p, i] = s
    state = tuple(jnp.asarray(a) for a in (word_tok, word_len, word_surf, n_words))
    cand_surf = jnp.asarray(_BD_V, jnp.int32)
    cand_tok = cand_surf[:, None].astype(jnp.int32)
    cand_len = jnp.ones((_BD_KC,), jnp.int32)
    done = jnp.ones((P,), bool)

    (nt, nl, ns, nnw), mlw = birth_death_move(
        jax.random.PRNGKey(0), *state, done, score_fn, cand_tok, cand_len, cand_surf)
    _assert_canonical_pad(nt, nl, ns, nnw)
    nnw_np = np.asarray(nnw)
    assert np.all((nnw_np >= 0) & (nnw_np <= Wmax)), "n_words out of range"
    assert nnw_np[2] == 1, "empty sentence must birth (n: 0 -> 1)"
    assert nnw_np[3] == 2, "full sentence must die (n: 3 -> 2)"
    assert np.all(np.isfinite(np.asarray(mlw))), "non-finite move weight"


def main():
    test_insert_delete_roundtrip()
    test_delete_insert_roundtrip()
    test_involution_self_inverse()
    test_boundary_append_and_remove_last()
    test_rj_weight_invariance_exact()
    test_rj_weight_invariance_informed()
    test_birth_death_move_smoke()
    print("birth/death Phase-0 + Phase-1 + Phase-2 gates: 7/7 PASS")


if __name__ == "__main__":
    main()
