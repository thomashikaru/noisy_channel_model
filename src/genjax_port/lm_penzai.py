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

# Default LM. pythia-70m is the default: it already shows reasonable correctness/behavior and
# is ~6x cheaper than 410m, so it is the baseline LM for all runs. Override with NC_LM, e.g.
# NC_LM=EleutherAI/pythia-410m for the larger/sharper model. All Pythia sizes share the GPT-NeoX
# arch + tokenizer (vocab 50304, tokenizer len 50277), so swapping sizes needs no other code
# changes. A sharper LM gives a steeper plausibility gradient so edits track real corruptions
# (LM quality is the dominant lever).
MODEL_NAME = os.environ.get("NC_LM", "EleutherAI/pythia-70m")
EOS_ID = 0  # GPT-NeoX <|endoftext|>, used as BOS seed and padding
CACHE_PAD = -1  # KV cache padding (NOT EOS_ID)

_model = None  # cached penzai TransformerLM
_pz = None     # cached penzai.pz module
_kv_stateless = None
_kv_vars0 = None
_kv_cache_len = None


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


@functools.partial(jax.jit, static_argnums=())
def _seq_token_logprobs_jit(token_bufs):
    all_lp = jax.nn.log_softmax(_raw_logits(token_bufs), axis=-1)             # [N, seq, vocab]
    nxt = token_bufs[:, 1:]                                                   # [N, seq-1] next tokens
    g = jnp.take_along_axis(all_lp[:, :-1, :], nxt[:, :, None], axis=2)[:, :, 0]  # [N, seq-1]
    return jnp.concatenate([jnp.zeros((token_bufs.shape[0], 1), g.dtype), g], axis=1)  # [N, seq]


def seq_token_logprobs(token_bufs):
    """Teacher-forcing next-token logprobs for EVERY position in ONE forward: ``[N, seq]`` where
    ``[:, j] = log P(token_bufs[:, j] | token_bufs[:, :j])`` (and ``[:, 0] = 0``, no prefix). The rewind/
    rescore scorers (the bd move's ``_lm_logprior``) need the logprob at many positions of the same buffer;
    reading them from a single pass replaces one full :func:`next_token_logits` forward PER position (which
    recomputes the identical forward and discards all but one row). ``load_model()`` is forced eagerly for
    the same tracer-leak reason as :func:`next_token_logits`."""
    load_model()
    return _seq_token_logprobs_jit(token_bufs)


def _batch_tail_logprobs_uncached(ctx_bufs, ctx_lens, tails, tail_lens):
    """Uncached chain-rule tail scorer (fallback)."""
    b, k, w = tails.shape
    flat_ctx = jnp.repeat(ctx_bufs, k, axis=0)
    flat_ilen = jnp.repeat(ctx_lens, k, axis=0)
    flat_tails = tails.reshape(b * k, w)
    flat_lens = tail_lens.reshape(b * k)
    scores = jnp.zeros(b * k, dtype=jnp.float32)
    rows = jnp.arange(b * k)

    def one_step(i, carry):
        sc, buf, ilen = carry
        active = i < flat_lens
        lp = next_token_logprobs(buf, ilen)
        tok = flat_tails[rows, i]
        sc = sc + jnp.where(active, lp[rows, tok], 0.0)
        buf = buf.at[rows, ilen].set(tok)
        ilen = ilen + active.astype(jnp.int32)
        return sc, buf, ilen

    scores, _, _ = jax.lax.fori_loop(0, w, one_step, (scores, flat_ctx, flat_ilen))
    return scores.reshape(b, k)


def batch_tail_logprobs(ctx_bufs, ctx_lens, tails, tail_lens, use_kv=None):
    """Chain-rule log P(tail | ctx) batched over ``[B, K]`` candidates, where ``tail`` is the
    continuation AFTER ``ctx[:ctx_len]``: ``sum_i log P(tail[i] | ctx, tail[:i])``. The KV path
    prefills the prefix once per row and shares it across the K candidates."""
    w = tails.shape[-1]
    if use_kv is None:
        use_kv = os.environ.get("NC_USE_KV", "0") == "1"
    if use_kv:
        load_model()
        # cache must hold the prefix + the fed tail: size = ctx width + max_tail (pad ctx wider).
        cache_len = int(ctx_bufs.shape[1]) + w
        _kv_setup(cache_len)
        pad = jnp.full((ctx_bufs.shape[0], cache_len - ctx_bufs.shape[1]), CACHE_PAD, ctx_bufs.dtype)
        return _batch_tail_logprobs_kv(jnp.concatenate([ctx_bufs, pad], axis=1),
                                       ctx_lens, tails, tail_lens, w)
    return _batch_tail_logprobs_uncached(ctx_bufs, ctx_lens, tails, tail_lens)


def _kv_setup(cache_len):
    """Build (or reuse) the single-sequence KV-caching LM for rewind tail scoring."""
    global _kv_stateless, _kv_vars0, _kv_cache_len
    if _kv_stateless is not None and _kv_cache_len >= cache_len:
        return
    from penzai.models.transformer import sampling_mode
    load_model()
    caching = sampling_mode.KVCachingTransformerLM.from_uncached(
        _model, cache_len=cache_len, batch_axes={}, pad_id=CACHE_PAD)
    _kv_stateless, _kv_vars0 = _pz.unbind_variables(caching, freeze=True)
    _kv_cache_len = cache_len


def _pad_buf(buf, ilen, cache_len):
    """Real tokens for ``idx < ilen``; ``CACHE_PAD`` elsewhere (JIT-safe)."""
    idx = jnp.arange(cache_len)
    return jnp.where(idx < ilen, buf[idx], jnp.int32(CACHE_PAD))


@functools.partial(jax.jit, static_argnums=(4,))
def _batch_tail_logprobs_kv(ctx_bufs, ctx_lens, tails, tail_lens, max_tail):
    """KV scorer: chain-rule log P(tail | ctx[:ctx_len]) for ``[B, K]`` candidates, tail = the
    continuation AFTER ctx. PREFILL the prefix ONCE per row and SHARE its K/V across the K
    candidates (one prefill, K short feeds): read ``P(tail[0]|ctx)`` from the prefill logits at
    ``ctx_len-1``, then feed each candidate's tail (starting at position ctx_len) and read
    ``P(tail[j+1]|ctx,tail[:j+1])`` from feed logit ``j``. Matches the uncached scorer / a hand
    chain-rule (planning/kv_cache_spikes/tail_scorer_verify.py)."""
    cache_len = ctx_bufs.shape[1]
    idx = jnp.arange(max_tail)

    def score_row(buf, ilen, row_tails, row_lens):
        kv_buf = _pad_buf(buf, ilen, cache_len)
        pf = _pz.bind_variables(_kv_stateless, _kv_vars0, unfreeze_as_copy=True)
        pf_logits = pf(_pz.nx.wrap(kv_buf).tag("seq")).untag("seq", "vocabulary").unwrap()  # [cl,V]
        pf_lp = jax.nn.log_softmax(pf_logits)
        _, prefix_vars = _pz.unbind_variables(pf, freeze=True)          # frozen prefix K/V (shared)
        first_lp = pf_lp[jnp.maximum(ilen - 1, 0)]                       # [V] = log P(.|ctx)

        def one_cand(tail, tlen):
            first = jnp.where(tlen > 0, first_lp[tail[0]], 0.0)          # log P(tail[0]|ctx)
            tail_feed = jnp.where(idx < tlen, tail, jnp.int32(CACHE_PAD))
            b = _pz.bind_variables(_kv_stateless, prefix_vars, unfreeze_as_copy=True)
            b.cache_end_index.value = ilen                              # feed tail at position ctx_len
            logits = b(_pz.nx.wrap(tail_feed).tag("seq")).untag("seq", "vocabulary").unwrap()
            lp = jax.nn.log_softmax(logits)                            # [max_tail, V]
            nxt = jnp.where(idx + 1 < tlen, tail[jnp.clip(idx + 1, 0, max_tail - 1)], 0)
            rest = jnp.sum(jnp.where(idx + 1 < tlen, lp[idx, nxt], 0.0))  # log P(tail[j+1]|ctx,tail[:j+1])
            return first + rest

        return jax.vmap(one_cand)(row_tails, row_lens)

    return jax.vmap(score_row)(ctx_bufs, ctx_lens, tails, tail_lens)
