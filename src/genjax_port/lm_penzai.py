"""In-graph language model for the Genjax noisy-channel port.

Loads a Pythia (GPT-NeoX) model from HuggingFace and converts it to penzai once, then
exposes a jittable next-token-logits function. penzai has no GPT-2 loader, but
``gpt_neox_from_huggingface_model`` supports the Pythia suite; after conversion the forward
pass is pure penzai/JAX (no torch in the loop), so it jits, vmaps, and runs on GPU.

The LM lives behind ``next_token_logits`` so a different LM (e.g. literal GPT-2 via flax)
could be swapped in without touching the model/proposal/filter code.

Buffer convention used throughout the port:
- The intended sentence is a fixed-shape int buffer ``[max_intended]``.
- Position 0 is seeded with ``EOS_ID`` (GPT-NeoX <|endoftext|>) as the start-of-sequence
  context, so a buffer with ``i_len`` filled positions has its "next token" distribution at
  logits position ``i_len - 1``. Padded positions ``>= i_len`` hold ``EOS_ID``; causal
  attention means they never influence earlier positions.
"""

import functools
import os

import jax
import jax.numpy as jnp

# Default LM. Override with NC_LM, e.g. NC_LM=EleutherAI/pythia-70m for the smaller/faster
# model. All Pythia sizes share the GPT-NeoX arch + tokenizer (vocab 50304, tokenizer len
# 50277), so swapping sizes needs no other code changes. A sharper LM gives a steeper
# plausibility gradient so edits track real corruptions (LM quality is the dominant lever).
MODEL_NAME = os.environ.get("NC_LM", "EleutherAI/pythia-410m")
EOS_ID = 0  # GPT-NeoX <|endoftext|>, used as BOS seed and padding

_model = None  # cached penzai TransformerLM
_pz = None     # cached penzai.pz module


def load_model():
    """Load + convert the LM once; returns the cached penzai TransformerLM."""
    global _model, _pz
    if _model is not None:
        return _model
    import torch
    from transformers import GPTNeoXForCausalLM
    from penzai import pz
    from penzai.models.transformer.variants import gpt_neox

    hf = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    hf.eval()
    hf.config._name_or_path = ""  # penzai's strict config check rejects this metadata attr
    _model = gpt_neox.gpt_neox_from_huggingface_model(
        hf, upcast_activations_to_float32=True
    )
    _pz = pz
    return _model


def vocab_size():
    load_model()
    # embedder vocabulary axis size; read lazily from a 1-token forward
    logits = _raw_logits(jnp.array([[EOS_ID]], dtype=jnp.int32))
    return int(logits.shape[-1])


def _raw_logits(token_buf):
    """token_buf: int array [batch, seq] -> logits [batch, seq, vocab] (jnp)."""
    model = load_model()
    pz = _pz
    tokens_nx = pz.nx.wrap(token_buf).tag("batch", "seq")
    out = model(tokens_nx)  # NamedArray [batch, seq, vocabulary]
    return out.unwrap("batch", "seq", "vocabulary")


@functools.partial(jax.jit, static_argnums=())
def _next_token_logits_jit(token_bufs, i_lens):
    all_logits = _raw_logits(token_bufs)  # [P, seq, vocab]
    P = token_bufs.shape[0]
    pos = jnp.clip(i_lens - 1, 0, token_bufs.shape[1] - 1)
    return all_logits[jnp.arange(P), pos]  # [P, vocab]


def next_token_logits(token_bufs, i_lens):
    """Batched next-token logits, one GPT forward across all particles.

    Args:
        token_bufs: int array ``[P, max_intended]`` (position 0 = EOS_ID seed).
        i_lens: int array ``[P]``, number of filled positions per particle.

    Returns:
        logits ``[P, vocab]`` -- the next-token distribution for each particle,
        read at position ``i_len - 1``.

    ``load_model()`` is forced here, EAGERLY, before the jitted forward: otherwise the first LM call
    builds the penzai model inside the jit trace, leaking its arrays as tracers (an
    ``UnexpectedTracerError`` once a second jit -- e.g. the rejuvenation move -- sees them). Callers
    that ran ``load_model()`` at startup never hit it; cold callers (run.py) did.
    """
    load_model()
    return _next_token_logits_jit(token_bufs, i_lens)


def next_token_logprobs(token_bufs, i_lens):
    """Log-softmax of :func:`next_token_logits` (normalized log-probabilities)."""
    return jax.nn.log_softmax(next_token_logits(token_bufs, i_lens), axis=-1)
