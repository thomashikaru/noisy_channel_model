"""Token-level noise model for the Genjax noisy-channel port.

Given an observed BPE token, we enumerate the small set of *intended* tokens that could
have produced it, à la ``get_form_sub_ps`` / ``choose_action`` in the original Gen.jl code:

- ``copy``: intended == observed (the LM must explain the observed token directly).
- ``sub`` : intended is a form-neighbor of the observed token -- an edit-distance-1 variant
  of its surface string that is itself a single vocab token.

Because the observed token is always *data* (we score it, never sample it) and the BPE
vocab has ~50k tokens, we never build a dense ``[V, V]`` table or compute edit distance
in-graph. Instead we generate the (tiny) candidate set per observed token on the host.

Likelihood conventions (``p(observed | intended, action)``):
- copy: ``1`` if observed == intended else ``0``.
- sub : ``SUB_PARAM**dist / Z(intended)`` where ``Z`` sums over the intended token's own
  form-neighbors. With edit-1 only (dist=1) this reduces to a uniform ``1 / n_neighbors``.
- insert: flat ``1 / V`` (a spurious token from nowhere; strongly penalized, as it should be).
"""

import functools
import math

from .tokenizer import surface, str_to_id

SUB_PARAM = 0.1  # form-substitution sharpness (mode of Beta(2,11) in the original config)
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _split_leading_space(s):
    """GPT-NeoX word-initial tokens carry a leading space; edit only the letters."""
    if s.startswith(" "):
        return " ", s[1:]
    return "", s


def _edits1(word):
    """Norvig-style edit-distance-1 neighborhood of a (space-stripped) string."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in _LETTERS]
    inserts = [a + c + b for a, b in splits for c in _LETTERS]
    return set(deletes + transposes + replaces + inserts)


@functools.lru_cache(maxsize=4096)
def _single_token_edit1_neighbors(token_id):
    """Token ids whose surface is an edit-1 variant of ``token_id``'s surface.

    Restricted to strings that are themselves a single vocab token (so the substitution
    stays a clean one-token -> one-token event). Returns a tuple of ids.
    """
    prefix, body = _split_leading_space(surface(token_id))
    if not body:
        return ()
    s2id = str_to_id()
    out = []
    for variant in _edits1(body):
        cand = prefix + variant
        cid = s2id.get(cand)
        if cid is not None and cid != token_id:
            out.append(cid)
    # Sort for determinism: _edits1 returns a set whose iteration order is hash-seed
    # dependent, which would otherwise make candidate ordering (and thus sampling) vary
    # across processes even with a fixed PRNG seed.
    return tuple(sorted(set(out)))


@functools.lru_cache(maxsize=4096)
def _neighbor_count(token_id):
    """Z-normalization proxy: number of single-token edit-1 form-neighbors."""
    return len(_single_token_edit1_neighbors(token_id))


def sub_candidates(obs_id):
    """Form-substitution explanations of an observed token.

    Returns a list of ``(intended_id, sub_loglik)`` where ``sub_loglik`` is
    ``log p(observed=obs_id | intended=intended_id, sub)``. A neighbor ``x`` of the
    observed token has the observed token among *its* neighbors, so the likelihood is
    ``1 / n_neighbors(x)`` (uniform over x's form-neighbors).
    """
    out = []
    for x in _single_token_edit1_neighbors(obs_id):
        nx = _neighbor_count(x)
        if nx > 0:
            out.append((x, -math.log(nx)))
    return out


def insertion_loglik(vocab_size):
    """Log-likelihood of any observed token under an insertion (flat over the vocab)."""
    return -math.log(vocab_size)
