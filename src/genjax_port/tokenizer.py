"""Tokenizer wrapper for the Genjax noisy-channel port.

Uses the Pythia / GPT-NeoX BPE tokenizer (pure Python, load-time only -- no torch in the
inference loop). Provides id<->string conversion and single-token surface forms, which the
token-level noise model uses to build edit-distance substitution candidates.
"""

import functools

from .lm_penzai import MODEL_NAME, EOS_ID

_tok = None


def _tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tok


def encode(text):
    """Encode a string to a list of token ids (no special tokens added)."""
    return _tokenizer().encode(text, add_special_tokens=False)


def decode(ids):
    """Decode an iterable of token ids back to a string."""
    return _tokenizer().decode([int(i) for i in ids])


def surface(token_id):
    """Surface string of a single token (e.g. ' boy' -> a leading space marks word start)."""
    return _tokenizer().decode([int(token_id)])


@functools.lru_cache(maxsize=1)
def vocab_strings():
    """Tuple of surface strings indexed by token id (for candidate lookup)."""
    tok = _tokenizer()
    n = len(tok)
    return tuple(tok.decode([i]) for i in range(n))


@functools.lru_cache(maxsize=1)
def str_to_id():
    """Map from single-token surface string -> token id (first id wins on collision)."""
    mapping = {}
    for i, s in enumerate(vocab_strings()):
        mapping.setdefault(s, i)
    return mapping
