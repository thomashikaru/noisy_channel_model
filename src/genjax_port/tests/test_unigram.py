"""Unigram-relative surprisal gate tests (LM-independent).

Locks in the semantics of the conditional-rejuvenation gate's input, ``surprisal - unigram_surp``
(Gen.jl ``gen_inference.jl:407-409``): the gate should fire on a *common* word that is surprising
in context, but NOT on a *rare* word that is merely improbable out of context (a rare word is
expected to be surprising, so reanalysing it wastes effort). We feed synthetic contextual surprisals
so the test needs no LM.

    PYTHONPATH=. python -m src.genjax_port.tests.test_unigram
"""

from src.genjax_port.unigram import unigram_surprisal, FLOOR_FREQ, CEIL_FREQ, custom_sigmoid


def test_floors():
    """Common words < rare words < unknown words; punctuation/empty pinned to the low (ceil) floor."""
    import math

    assert unigram_surprisal("the") < unigram_surprisal("experiment")
    assert unigram_surprisal("experiment") < unigram_surprisal("zzqxkjv")  # unknown -> FLOOR_FREQ
    assert unigram_surprisal("zzqxkjv") == -math.log(FLOOR_FREQ)
    # punctuation and empty/EOS units are treated as very common (CEIL_FREQ)
    ceil_surp = -math.log(CEIL_FREQ)
    assert unigram_surprisal(".") == ceil_surp
    assert unigram_surprisal(",") == ceil_surp
    assert unigram_surprisal("") == ceil_surp


def test_gate_unigram_relative():
    """A common word that is surprising in context gets a HIGH gate prob; a rare word with the
    same contextual surprisal gets a LOW one (it is expected to be surprising)."""
    # Synthetic contextual surprisal shared by both words. With center=0, spread=1 (the new gate
    # defaults), the gate input is surprisal - unigram_surp.
    surprisal = 9.0
    center, spread = 0.0, 1.0

    common_surp = unigram_surprisal("the")          # ~2.9: surprisal - unigram >> 0 -> fires
    rare_surp = unigram_surprisal("experiment")      # ~10.6: surprisal - unigram < 0 -> fires little

    p_common = custom_sigmoid(surprisal - common_surp, center, spread)
    p_rare = custom_sigmoid(surprisal - rare_surp, center, spread)

    assert p_common > 0.9, p_common
    assert p_rare < 0.2, p_rare
    assert p_common > p_rare

    # Sanity: gating on RAW surprisal (the old behavior) would fire on BOTH equally -- the
    # unigram-relative input is exactly what separates them.
    p_raw = custom_sigmoid(surprisal, center, spread)
    assert p_raw > 0.9 and abs(p_raw - p_common) > 0.0  # raw can't distinguish common vs rare


if __name__ == "__main__":
    test_floors()
    test_gate_unigram_relative()
    print("all passed")
