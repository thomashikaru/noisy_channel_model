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

import numpy as np
import jax.numpy as jnp

from genjax_port.pairhmm_rejuv import (
    _PAD_SURF, _insert_word, _delete_word, _involve,
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


def main():
    test_insert_delete_roundtrip()
    test_delete_insert_roundtrip()
    test_involution_self_inverse()
    test_boundary_append_and_remove_last()
    print("birth/death involution Phase-0 gates: 4/4 PASS")


if __name__ == "__main__":
    main()
