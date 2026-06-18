"""Word-span noise layer for BPE-token-count substitutions (e.g. "experimemt"->"experiment").

The per-token noise model (noise.py) is 1:1 token-aligned, so it cannot correct a typo whose
fix changes the BPE token count -- the correct intended word is never in its candidate set
(see the genjax-port-bpe-substitution memo + /tmp/bpe_diag.py). This module works at the
*word* level instead:

- ``segment_words`` groups an observed token sequence into words (GPT-NeoX marks word starts
  with a leading space). The segmentation is deterministic data, so a word-scan SMC stays in
  lockstep across particles (all consume the same observed word per step).
- ``word_sub_candidates`` proposes single-vocab-token intended words whose surface is within a
  (configurable, NOT fixed-at-1) character edit distance of the observed word string -- the N:1
  case (n observed tokens explained by 1 intended token), which Phase-0 showed covers the common
  typos because correct words are usually a single BPE token. Multi-token intended words (full
  M:N) are a later extension.

Candidate retrieval uses a **SymSpell deletion index** over the single-token, word-initial
vocab: the edit distance is a real (Damerau-Levenshtein) parameter, not capped at 1. The cap is
only a tractability bound on the index; the substitution likelihood ``SUB_PARAM**d`` already
down-weights far candidates (e.g. d=3 contributes ~SUB_PARAM**3), so the model -- not a hard
distance limit -- decides how far a substitution is worth positing.
"""

import functools
import math

from .tokenizer import surface, str_to_id, vocab_strings, encode
from .noise import _split_leading_space, SUB_PARAM
from .config import MAX_SUB_CANDIDATES

# Frequency-ranked dictionary size for MULTI-TOKEN substitution candidates (Phase D / D2). The
# single-token candidate pool is the ~27k word-initial vocab; multi-token intended words (which the
# vocab pool can't express) come from the top-N wordfreq English list, re-tokenized. N trades
# coverage (rarer correct words) against the SymSpell index build cost. 30k covers common multi-token
# words; raise for more proper-noun/rare coverage.
MULTITOKEN_DICT_N = 30000
MAX_MULTITOKEN_CANDIDATES = 8   # per observed word; multi-token tail-scoring is the cost knob


def _is_punct(surf):
    """A token is punctuation if its first non-space character is non-alphabetic."""
    body = surf.lstrip(" ")
    return bool(body) and not body[0].isalpha()


def segment_words(obs_ids):
    """Group observed token ids into units: alphabetic WORDS and PUNCTUATION runs.

    - A word unit is a run of word-piece tokens; a new word begins at a leading-space token
      (so "experimemt" = ' exper','im','em','t' is one unit).
    - Punctuation is kept as its OWN unit, never absorbed into the adjacent word. This matters
      twice: (1) a trailing "." is not folded into a substitution (so "inflection."->"infection"
      corrects only the word and keeps the period), and (2) the period stays a real intended
      token, so the LM scores it -- a sentence-final period behaves like EOS, up-ranking complete
      sentences over fragments and thus penalizing insertions/deletions that break sentencehood.

    Returns a list of ``(token_ids, unit_str)`` where ``unit_str`` is the space-stripped surface.
    """
    units = []
    cur_ids, cur_surf, cur_is_punct = [], "", None
    for tid in obs_ids:
        s = surface(int(tid))
        is_p = _is_punct(s)
        if cur_ids and (is_p != cur_is_punct or (not is_p and s.startswith(" "))):
            units.append((cur_ids, cur_surf))
            cur_ids, cur_surf = [], ""
        cur_ids.append(int(tid))
        cur_surf += s
        cur_is_punct = is_p
    if cur_ids:
        units.append((cur_ids, cur_surf))
    return [(ids, _split_leading_space(surf)[1]) for ids, surf in units]


@functools.lru_cache(maxsize=1)
def _word_initial_vocab():
    """Map ``body -> token_id`` for single tokens whose surface is a word-initial alphabetic
    string (leading space stripped). This is the intended-word candidate pool."""
    out = {}
    for i, s in enumerate(vocab_strings()):
        if s.startswith(" "):
            body = s[1:]
            if body and body.isalpha():
                out.setdefault(body, i)  # first id wins on collision
    return out


def _deletes(word, max_edit):
    """All strings obtainable by deleting up to ``max_edit`` characters from ``word``."""
    result = {word}
    frontier = {word}
    for _ in range(max_edit):
        nxt = set()
        for w in frontier:
            for i in range(len(w)):
                nxt.add(w[:i] + w[i + 1:])
        result |= nxt
        frontier = nxt
    return result


@functools.lru_cache(maxsize=4)
def _symspell_index(max_edit):
    """delete-variant -> tuple of candidate bodies. SymSpell: two strings are within
    Damerau-Levenshtein distance D iff they share a common delete-variant (up to D deletes on
    each side), so indexing dictionary deletes lets us retrieve all candidates within D by
    looking up the query's deletes. Built once per max_edit (lru-cached)."""
    index = {}
    for body in _word_initial_vocab():
        for dv in _deletes(body, max_edit):
            index.setdefault(dv, []).append(body)
    return {k: tuple(v) for k, v in index.items()}


def _damerau_levenshtein(a, b, cutoff):
    """Optimal-string-alignment (Damerau-Levenshtein with adjacent transpositions) distance,
    returning early as ``cutoff + 1`` once the whole row exceeds ``cutoff``."""
    la, lb = len(a), len(b)
    if abs(la - lb) > cutoff:
        return cutoff + 1
    prev2 = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
            if v < row_min:
                row_min = v
        if row_min > cutoff:
            return cutoff + 1
        prev2, prev = prev, cur
    return prev[lb]


def word_sub_candidates(word_str, max_dist=2, max_candidates=MAX_SUB_CANDIDATES):
    """Single-token intended words within Damerau-Levenshtein distance ``max_dist`` of
    ``word_str``, nearest first.

    Returns ``[(token_id, char_dist), ...]`` for word-initial single vocab tokens, distance NOT
    fixed at 1 (retrieved via the SymSpell index). Truncated to ``max_candidates`` (closest by
    distance; default :data:`config.MAX_SUB_CANDIDATES`) for tractability -- not by a hard distance
    cap; far candidates are kept rare by ``SUB_PARAM**d`` in the likelihood. The candidate count
    this sets, ``K``, is the dominant cost of the suffix-aware rejuvenation move (it forwards
    ``P*K`` buffers per revisited word), so the cap is a real speed knob. The literal word itself is
    excluded (that is COPY).
    """
    if not word_str or not word_str[0].isalpha():
        return []  # no substitutions for punctuation / non-word units
    pool = _word_initial_vocab()
    index = _symspell_index(max_dist)
    cand_bodies = set()
    for qv in _deletes(word_str, max_dist):
        cand_bodies.update(index.get(qv, ()))
    out = []
    for body in cand_bodies:
        if body == word_str:
            continue  # COPY branch, not a substitution
        d = _damerau_levenshtein(word_str, body, max_dist)
        if 1 <= d <= max_dist:
            out.append((pool[body], d))
    out.sort(key=lambda t: (t[1], t[0]))  # nearest first, deterministic
    return out[:max_candidates]


@functools.lru_cache(maxsize=1)
def _multitoken_dict(n=MULTITOKEN_DICT_N):
    """``body -> BPE token span`` for the top-``n`` wordfreq English words that tokenize to >= 2 BPE
    tokens. This is the M:N substitution pool (Phase D / D2): intended words whose surface spans >= 2
    tokens, which the single-token ``_word_initial_vocab`` cannot express. Single-token words are
    excluded (already covered by :func:`word_sub_candidates`). Built once (re-tokenizes ``n`` words)."""
    import wordfreq
    out = {}
    for w in wordfreq.top_n_list("en", n):
        if not w.isalpha():
            continue
        span = tuple(encode(" " + w))
        if len(span) >= 2:
            out[w] = span
    return out


@functools.lru_cache(maxsize=4)
def _multitoken_symspell(max_edit):
    """delete-variant -> candidate multi-token bodies (SymSpell index over :func:`_multitoken_dict`)."""
    index = {}
    for body in _multitoken_dict():
        for dv in _deletes(body, max_edit):
            index.setdefault(dv, []).append(body)
    return {k: tuple(v) for k, v in index.items()}


def word_sub_candidates_multitoken(word_str, max_dist=2, max_candidates=MAX_MULTITOKEN_CANDIDATES):
    """MULTI-TOKEN intended words within Damerau-Levenshtein ``max_dist`` of ``word_str`` (Phase D /
    D2). Returns ``[(token-span tuple, surface str, char_dist), ...]`` nearest first, for dictionary
    words that tokenize to >= 2 BPE tokens -- the substitution targets the single-token pool misses
    (e.g. a misspelling of 'kitten' = ' k'+'itten'). Same SymSpell retrieval + ``SUB_PARAM**d``
    down-weighting as :func:`word_sub_candidates`; the literal is excluded (that is COPY)."""
    if not word_str or not word_str[0].isalpha():
        return []
    pool = _multitoken_dict()
    index = _multitoken_symspell(max_dist)
    cand_bodies = set()
    for qv in _deletes(word_str, max_dist):
        cand_bodies.update(index.get(qv, ()))
    out = []
    for body in cand_bodies:
        if body == word_str:
            continue
        d = _damerau_levenshtein(word_str, body, max_dist)
        if 1 <= d <= max_dist:
            out.append((pool[body], body, d))
    out.sort(key=lambda t: (t[2], t[1]))  # nearest first, deterministic
    return out[:max_candidates]


def word_sub_loglik(char_dist):
    """log P(observed word | intended word, sub) for a character edit-distance.

    Channel approximation: each character edit costs a factor SUB_PARAM (the same sharpness as
    the token-level model). Unnormalized across candidates -- the copy/sub balance is governed
    by this together with the LM gradient (analogous to p_delete_prior as the edit-rate knob).
    """
    return char_dist * math.log(SUB_PARAM)
