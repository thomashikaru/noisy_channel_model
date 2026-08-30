"""Inflectional-variant edit class: a word and its number/agreement alternants.

The form channel scores an intended word by CHARACTER edit distance (``K * d`` in the align
channel). That is the right model for a typo and the wrong one for morphology. English
number agreement is partly suppletive -- ``is``/``are`` are three character edits apart and
``was``/``were`` three, while meaning-wise they are one step in a paradigm -- so a character
channel charges a suppletive alternation the same as an unrelated three-letter typo, which is
both linguistically wrong and, at the deployed ``K = -4.5``, about 13.5 nats of cost that no
realistic language-model gain can overcome.

This module defines the alternation relation. The channel change it enables is stated in one
line: **an inflectional alternation is ONE edit, whatever its character distance.** The cost is
:data:`MORPH_LP`, a flat log-probability that does NOT depend on ``d`` and is not derived from
``SUB_PARAM`` or the align slope; it is a separate, separately calibratable rate for a separate
kind of noise.

Scope is deliberately narrow and general: **number and agreement only**, the paradigm dimension
that noisy-channel agreement effects concern. Not tense, not derivation, not comparatives. The
relation is symmetric (if ``x`` alternates with ``y`` then ``y`` alternates with ``x``) and is a
property of English, not of any stimulus set -- it is generated from regular affixal rules plus a
short closed list of genuinely irregular paradigms, and never from a dataset's word inventory.

Deliberately NOT included, to keep the class honest:

* Anything that changes lexeme (``gift``/``present``) -- that is what the LM is for.
* Anything a character edit already handles cheaply. ``gift``/``gifts`` IS included even though
  it is one character, because membership must depend on the grammar rather than on whether the
  character channel happens to reach it; the two routes combine (see ``pairhmm_smc``), so a
  one-character alternation simply gets a slightly better score from two independent noise paths.
"""

from __future__ import annotations

import functools

#: Flat log-probability of an inflectional alternation, independent of character distance.
#: Defaults to the align channel's per-edit cost, so at the shipped operating point "one
#: morphological alternation" costs exactly what "one character edit" costs -- a neutral starting
#: point, not a claim that the two rates are equal. It is a separate constant precisely so it can
#: be swept without touching K, and it SHOULD be calibrated before being relied on quantitatively.
MORPH_LP = -4.5

#: Genuinely irregular / suppletive paradigms, as sets of forms that alternate for NUMBER or
#: subject agreement. Closed class, hand-listed because no rule generates them. Every pair within
#: a set alternates; sets are small on purpose.
IRREGULAR_SETS: tuple[frozenset[str], ...] = tuple(frozenset(s.split()) for s in (
    # verbal agreement (the suppletive core: this is what a character channel cannot see)
    "am is are",
    "was were",
    "has have",
    "does do",
    "goes go",
    "says say",
    "isn't aren't",
    "wasn't weren't",
    "hasn't haven't",
    "doesn't don't",
    # irregular noun plurals
    "child children", "man men", "woman women", "person people", "foot feet",
    "tooth teeth", "goose geese", "mouse mice", "louse lice", "ox oxen",
    "die dice", "penny pence", "datum data", "medium media", "criterion criteria",
    "phenomenon phenomena", "analysis analyses", "basis bases", "crisis crises",
    "thesis theses", "hypothesis hypotheses", "index indices", "appendix appendices",
    "matrix matrices", "vertex vertices", "axis axes", "cactus cacti", "fungus fungi",
    "nucleus nuclei", "radius radii", "stimulus stimuli", "syllabus syllabi",
    "alumnus alumni", "curriculum curricula", "bacterium bacteria",
))

#: Closed-class items with no number/agreement alternation at all. The affixal rules would happily
#: produce ``fors``, ``unders``, ``thes``; those are not English, and worse, several survive a
#: frequency check because they occur as surnames or abbreviations. Prepositions, determiners,
#: conjunctions, complementizers and pronouns are morphologically invariant for number, so they are
#: blocked outright. The auxiliaries that DO alternate (is/are, was/were, has/have, does/do) are not
#: here -- they are in IRREGULAR_SETS, which is consulted first.
INVARIANT: frozenset[str] = frozenset("""
    the a an this that these those each every some any no all both half either neither
    of to for from in on at by with about into over under above below between among through
    during before after since until against across behind beside beyond within without toward
    towards upon onto off out up down near per via
    and or but nor so yet as if then than because although though while whereas unless whether
    i you he she it we they me him her us them my your his its our their mine yours hers ours
    theirs myself yourself himself herself itself ourselves yourselves themselves who whom whose
    which what where when why how there here not very too also just only even still already
    can could will would shall should may might must ought
""".split())

#: Consonants for the -y -> -ies rule.
_VOWELS = frozenset("aeiou")

#: Stem endings that take -es rather than -s.
_ES_ENDINGS = ("s", "x", "z", "ch", "sh")

#: Minimum Zipf frequency for a generated alternant to count as a real English word. Affixal
#: rules over-generate (``knives`` -> ``knif``, ``babies`` -> ``babi``), and no rule set fixes that
#: -- a lexicon check is the standard remedy. 1.5 is about "attested in a large corpus at all",
#: chosen to admit rare-but-real plurals while rejecting rule debris. Falls open (admits
#: everything) if wordfreq is unavailable, so the relation degrades to pure rules rather than
#: silently emptying.
MIN_ZIPF = 1.5


@functools.lru_cache(maxsize=1)
def _irregular_map() -> dict[str, frozenset[str]]:
    out: dict[str, set[str]] = {}
    for group in IRREGULAR_SETS:
        for form in group:
            out.setdefault(form, set()).update(group - {form})
    return {k: frozenset(v) for k, v in out.items()}


def _looks_plural(w: str) -> bool:
    """A surface -s that is plausibly the number affix (``gifts`` yes, ``glass``/``bus`` no)."""
    return w.endswith("s") and not w.endswith(("ss", "us", "is"))


def _pluralize(w: str) -> set[str]:
    if w.endswith("y") and len(w) > 1 and w[-2] not in _VOWELS:
        return {w[:-1] + "ies"}                      # baby -> babies
    if w.endswith(_ES_ENDINGS):
        return {w + "es"}                            # box -> boxes, catch -> catches
    if w.endswith("fe"):
        return {w + "s", w[:-2] + "ves"}             # knife -> knives
    if w.endswith("f"):
        return {w + "s", w[:-1] + "ves"}             # roof -> roofs, leaf -> leaves
    if w.endswith("o"):
        return {w + "s", w + "es"}                   # photo -> photos, hero -> heroes
    return {w + "s"}                                 # gift -> gifts


def _singularize(w: str) -> set[str]:
    """One analysis, most specific rule first -- an -ies is not also an -es and an -s."""
    if w.endswith("ies") and len(w) > 4:
        return {w[:-3] + "y"}                        # babies -> baby
    if w.endswith("ves") and len(w) > 4:
        return {w[:-3] + "f", w[:-3] + "fe"}         # leaves -> leaf, knives -> knife
    if w.endswith("es") and len(w) > 3 and w[:-2].endswith(_ES_ENDINGS):
        return {w[:-2]}                              # boxes -> box, catches -> catch
    return {w[:-1]}                                  # gifts -> gift, notes -> note


def _regular_variants(w: str) -> set[str]:
    """Number/agreement alternants from the regular affixal rules.

    Direction-aware: a word is analysed as EITHER a stem to be inflected or an inflected form to
    be reduced, never both, so ``gifts`` does not also yield ``giftses``.
    """
    out = _singularize(w) if _looks_plural(w) else _pluralize(w)
    out.discard(w)
    return {v for v in out if len(v) > 1}


@functools.lru_cache(maxsize=100_000)
def morph_variants(word: str) -> frozenset[str]:
    """Every word that alternates with *word* for number or subject agreement.

    Case-folded in and out. Symmetric by construction: ``y in morph_variants(x)`` iff
    ``x in morph_variants(y)`` (asserted for the irregular sets by construction, and for the
    regular rules by ``test_morphology``). Returns lowercase bodies, not token ids -- the caller
    resolves those against whatever vocabulary it uses.
    """
    w = "".join(c for c in word if c.isalpha() or c == "'").lower()
    if not w:
        return frozenset()
    irregular = _irregular_map().get(w)
    if irregular is not None:
        return irregular          # blocking: an irregular paradigm suppresses the regular rules
    if w in INVARIANT:
        return frozenset()        # closed-class: no number to alternate
    out = {v for v in _regular_variants(w) if _is_word(v)}
    out.discard(w)
    return frozenset(out)


@functools.lru_cache(maxsize=1)
def _zipf():
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return None
    return zipf_frequency


def _is_word(w: str) -> bool:
    """Is *w* attested English, or is it affixal-rule debris?"""
    if w in _irregular_map():
        return True
    zipf = _zipf()
    return True if zipf is None else zipf(w, "en") >= MIN_ZIPF


def alternates(a: str, b: str) -> bool:
    """Do *a* and *b* stand in the inflectional-alternation relation?"""
    return b.lower() in morph_variants(a) or a.lower() in morph_variants(b)
