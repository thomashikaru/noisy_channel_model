"""Word-frequency (unigram) surprisal for the conditional-rejuvenation gate.

The surprisal gate decides which words to reanalyse. Gating on raw contextual surprisal fires on
any word that is improbable in context -- including words that are simply *rare* (a rare word is
*expected* to be surprising, so reanalysing it wastes effort and can pull a legitimately-rare
literal toward a common substitute). Gen.jl instead gates on contextual surprisal *relative to*
the word's out-of-context (unigram) surprisal (``gen_inference.jl:407-409``)::

    cond_rejuv_p = custom_sigmoid(surprisal - unigram_surp, thresh, spread)

so the gate fires only when a word is *more* surprising in context than its base rate predicts --
the signature of a genuine garden-path reanalysis, not just a rare word.

This module is the port of Gen.jl's unigram table (``gen_lm.jl:90-92``), which is a normalized
word-frequency distribution with two floors. We use the ``wordfreq`` package as the frequency
source instead of a bundled CSV; the floors mirror Gen.jl:

- unknown / zero-frequency words -> ``FLOOR_FREQ`` (high surprisal). Gen.jl uses ``min_freq``; the
  effect is that an unrecognised word is treated as expected-to-be-surprising, so the gate fires
  *less* on it.
- punctuation-only units and empty / EOS -> ``CEIL_FREQ`` (low surprisal). Gen.jl uses
  ``max_freq``; punctuation is so common out of context that any in-context surprise it carries
  (e.g. a sentence-final period) is what should drive reanalysis of the preceding window.

The gate input is a *scalar per observed word* (shared across particles), so this is computed once
per word at hook-build time, never inside ``vmap``.
"""

import functools
import math

import wordfreq

from . import noise_word as NW

# Frequency floors (probabilities). Tuned from wordfreq's "en" distribution: real words bottom out
# around 1e-7 (e.g. "recieve"), so FLOOR_FREQ=1e-8 puts unknown words just below the rarest real
# word (surprisal ~18.4). CEIL_FREQ=1e-1 (surprisal ~2.3) is near the most common words ("the" is
# ~5e-2), marking punctuation/EOS as very common.
FLOOR_FREQ = 1e-8
CEIL_FREQ = 1e-1


@functools.lru_cache(maxsize=None)
def unigram_surprisal(word: str) -> float:
    """``-log`` of the word's unigram (out-of-context) probability, with Gen.jl's floors.

    ``word`` is a space-stripped surface (``noise_word.segment_words`` ``unit_str``). Punctuation
    units and empty/EOS map to ``CEIL_FREQ`` (low surprisal); unknown words to ``FLOOR_FREQ`` (high
    surprisal). Multi-token intended words pass their full surface to ``wordfreq`` unchanged.
    """
    w = word.strip()
    if not w or NW._is_punct(word):
        return -math.log(CEIL_FREQ)
    freq = wordfreq.word_frequency(w, "en")
    return -math.log(max(freq, FLOOR_FREQ))
