"""Gates for the inflectional-variant edit class and the comma in the indel insertion pool.

The relation itself (``genjax_port.morphology``) is pure python and tested exhaustively here.
The channel wiring is tested two ways: that ``morph=False`` leaves the filter bit-identical, and
that ``morph=True`` actually moves the emission cell it is supposed to move.
"""

import jax
import jax.numpy as jnp
import numpy as np

from genjax_port import morphology, pairhmm_smc, pythia_word_caprop as pwc, tokenizer
from genjax_port.morphology import alternates, morph_variants


# ---------------------------------------------------------------------------------------
# The relation
# ---------------------------------------------------------------------------------------

def test_suppletive_agreement_pairs():
    """The whole point: forms a character channel cannot see as related."""
    for a, b in [("is", "are"), ("was", "were"), ("has", "have"), ("does", "do"),
                 ("am", "is"), ("goes", "go")]:
        assert alternates(a, b), f"{a}/{b} should alternate"
        assert b in morph_variants(a) and a in morph_variants(b)


def test_regular_number_morphology():
    for a, b in [("gift", "gifts"), ("star", "stars"), ("baby", "babies"), ("box", "boxes"),
                 ("catch", "catches"), ("waiter", "waiters"), ("knife", "knives")]:
        assert alternates(a, b), f"{a}/{b} should alternate"


def test_irregular_plurals():
    for a, b in [("child", "children"), ("man", "men"), ("person", "people"),
                 ("mouse", "mice"), ("analysis", "analyses")]:
        assert alternates(a, b), f"{a}/{b} should alternate"


def test_irregular_blocks_the_regular_rule():
    """A suppletive paradigm suppresses affixation -- no ``childs``, no ``haves``, no ``ares``."""
    assert "childs" not in morph_variants("child")
    assert "haves" not in morph_variants("have")
    assert "ares" not in morph_variants("are")
    assert "wases" not in morph_variants("was")


def test_closed_class_words_do_not_inflect():
    """Prepositions, determiners and the like are invariant for number."""
    for w in ["for", "under", "the", "a", "that", "with", "and", "which", "can", "could"]:
        assert morph_variants(w) == frozenset(), f"{w} should have no alternants, got {morph_variants(w)}"


def test_no_double_affixation():
    """Direction-awareness: an already-plural form is not pluralized again."""
    for w in ["gifts", "stars", "babies", "boxes", "waiters"]:
        assert not any(v.endswith("ses") or v.endswith("eses") for v in morph_variants(w)), \
            f"{w} -> {morph_variants(w)}"


def test_rule_debris_is_rejected():
    """Affixal rules over-generate; the lexicon floor is what keeps the class honest."""
    for junk in ["knif", "kniv", "babi", "boxe", "ric"]:
        assert junk not in morph_variants({"knif": "knives", "kniv": "knives", "babi": "babies",
                                           "boxe": "boxes", "ric": "rices"}[junk])


def test_relation_is_not_a_lexeme_change():
    """It must not become a general 'related word' relation -- that is the LM's job."""
    for a, b in [("gift", "present"), ("is", "was"), ("kid", "child"), ("big", "large")]:
        assert not alternates(a, b), f"{a}/{b} must NOT alternate"


def test_alternates_is_symmetric():
    words = ["is", "are", "was", "were", "has", "have", "gift", "gifts", "child", "children",
             "baby", "babies", "for", "the", "star", "stars", "box", "boxes"]
    for a in words:
        for b in words:
            assert alternates(a, b) == alternates(b, a), f"asymmetric on {a}/{b}"


def test_variant_ids_resolve_to_single_word_initial_tokens():
    ids = pwc._morph_variant_ids("is")
    surfaces = {tokenizer.surface(i) for i in ids}
    assert " are" in surfaces, surfaces
    for i in ids:
        assert tokenizer.surface(i).startswith(" ")


# ---------------------------------------------------------------------------------------
# The comma in the indel insertion pool
# ---------------------------------------------------------------------------------------

def test_comma_is_in_the_insertion_pool():
    assert "," in pwc.FUNCWORDS
    surfaces = [tokenizer.surface(i) for i in pwc._funcword_ids()]
    assert "," in surfaces, surfaces


def test_every_pool_entry_is_a_single_token():
    """A multi-token entry would be silently dropped, shrinking the pool without a word."""
    assert len(pwc._funcword_ids()) == len(pwc.FUNCWORDS)


def test_comma_is_its_own_observed_unit():
    """Restoring a comma is an ordinary word insertion only because it segments as its own unit."""
    units = pwc._obs_word_units("Because the suspect changed, the file deserved further review.")
    assert "," in units, units


# ---------------------------------------------------------------------------------------
# The channel wiring
# ---------------------------------------------------------------------------------------

def _emit_row(observed, morph):
    """The (M, Vc) form table the filter builds, with the morphological patch applied or not."""
    model = pwc._pythia_model(".", morph=morph)
    obs_words = model.obs_words(observed)
    obs_char = jnp.stack([jnp.asarray(model.char_ids(w)[0], jnp.int32) for w in obs_words])
    emit = jax.vmap(jax.vmap(model.channel_form_align, in_axes=(None, 0, 0)),
                    in_axes=(0, None, None))(obs_char, model.vocab_char, model.vocab_clen)
    if morph:
        rows, cols = [], []
        for m, w in enumerate(obs_words):
            for tid in model.morph_variant_ids(w.strip().lower()):
                rows.append(m)
                cols.append(int(tid))
        idx = (jnp.asarray(rows, jnp.int32), jnp.asarray(cols, jnp.int32))
        emit = emit.at[idx].set(jnp.logaddexp(emit[idx], jnp.asarray(morphology.MORPH_LP, emit.dtype)))
    return obs_words, emit


def test_morph_channel_prices_a_suppletive_pair_as_one_edit():
    obs = "The gifts for the kid is hidden under the bed."
    words, off = _emit_row(obs, morph=False)
    _, on = _emit_row(obs, morph=True)
    m = words.index("is")
    are = int(tokenizer.encode(" are")[0])
    # Without the class the char DP charges is->are as several edits (it sums over alignments, so
    # it lands a little above a strict 3*K); with the class it is one edit flat.
    assert float(off[m, are]) < 2 * morphology.MORPH_LP, \
        f"without the class is->are should cost more than two edits, got {float(off[m, are])}"
    assert abs(float(on[m, are]) - morphology.MORPH_LP) < 0.05, \
        f"with the class it should cost about one edit, got {float(on[m, are])}"
    assert float(on[m, are]) - float(off[m, are]) > 7.0, \
        "the class should be worth several nats on a suppletive pair"


def test_morph_channel_leaves_the_copy_untouched():
    obs = "The gifts for the kid is hidden under the bed."
    words, off = _emit_row(obs, morph=False)
    _, on = _emit_row(obs, morph=True)
    for m, w in enumerate(words):
        if not w.strip().isalpha():
            continue
        tid = tokenizer.encode(" " + w.strip().lower())
        if len(tid) == 1:
            assert float(off[m, int(tid[0])]) == float(on[m, int(tid[0])]), \
                f"copy cell for {w!r} moved"


def test_morph_channel_touches_only_alternants():
    """The patch must be sparse: nothing outside the relation may change."""
    obs = "The gifts for the kid is hidden under the bed."
    words, off = _emit_row(obs, morph=False)
    _, on = _emit_row(obs, morph=True)
    changed = np.argwhere(np.asarray(off) != np.asarray(on))
    assert len(changed) > 0, "the class changed nothing at all"
    for m, v in changed:
        surf = tokenizer.surface(int(v)).strip().lower()
        assert alternates(words[m].strip().lower(), surf), \
            f"{words[m]!r} -> {surf!r} changed but is not an alternant"


def test_morph_off_is_bit_identical():
    """morph=False must reproduce the certified path exactly, including the candidate list."""
    obs = "The gifts for the kid is hidden under the bed."
    model_off = pwc._pythia_model(".", morph=False)
    assert model_off.morph_variant_ids is None
    for w in model_off.obs_words(obs):
        plain = pwc._candidate_words(w, None, 2, 12)
        via_model = list(model_off.candidate_words(w, None, 2, 12))
        assert plain == via_model, f"candidate list for {w!r} differs with morph off"


def test_morph_adds_candidates_only_when_on():
    off = [s for _sp, s in pwc._candidate_words("is", None, 2, 12, morph=False)]
    on = [s for _sp, s in pwc._candidate_words("is", None, 2, 12, morph=True)]
    assert "are" not in off and "are" in on, (off, on)
    assert on[0] == off[0] == "is", "the COPY must stay first"


def test_run_accepts_morph_and_defaults_on():
    import inspect
    assert inspect.signature(pwc.run).parameters["morph"].default is True
    assert inspect.signature(pairhmm_smc.run).parameters["morph_lp"].default is None
