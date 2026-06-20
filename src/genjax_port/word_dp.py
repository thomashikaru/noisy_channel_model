"""Word-level pair-HMM forward-DP recurrences + ESS — the live DP primitives.

Extracted from the toy ``poc_word_indel`` (now ``toy_bigram``) because these are pure ``jax`` (no
toy-vocab / LM dependency) and are shared by the unified RB-SMC filter (``pairhmm_smc``) and its
rejuvenation sweep (``pairhmm_rejuv``). They belong in the core, not in the toy bigram fixture.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _word_row_update(log_alpha, emit_col, wdel, wins):
    """One word-level pair-HMM row: extend the alignment by one intended word.

    log_alpha[k] : log P(prefix so far, k observed words consumed). emit_col[k-1] = channel score
    of aligning THIS intended word to observed word k. Returns the updated length-(M+1) vector.
    Same three-way recurrence as the char DP in poc_pairhmm_channel, one level up.

    ``wins`` is the spurious-word (insertion) log-cost. It may be a SCALAR (uniform over the vocab --
    the toy / certified default) OR an ``(M,)`` vector ``wins[j]`` = the cost of explaining observed
    word ``j`` as a spurious insertion. The insertion arc into consumed-count ``k`` consumes observed
    word ``k`` (1-indexed), so it pays ``wins[k-1]``. A frequency-aware channel passes the unigram cost
    here so a rare observed word is expensive to "explain away" as an insertion (broadcast of a scalar
    is bit-identical to the old uniform path)."""
    M = emit_col.shape[0]
    diag = log_alpha[0:M] + emit_col          # align this word to observed word k  (substitute/match)
    up = log_alpha[1:M + 1] + wdel            # this word is MISSING (no observation)  (delete)
    beta_rest = logsumexp(jnp.stack([diag, up]), axis=0)
    beta0 = log_alpha[0] + wdel
    beta = jnp.concatenate([beta0[None], beta_rest])
    wins_vec = jnp.broadcast_to(jnp.asarray(wins, beta.dtype), (M,))  # per-inserted-observed-word cost

    def ins_step(left, bw):                    # spurious observed word (insert): left-to-right sweep
        b, w = bw
        cell = logsumexp(jnp.stack([b, left + w]))
        return cell, cell

    _, rest = jax.lax.scan(ins_step, beta[0], (beta[1:], wins_vec))
    return jnp.concatenate([beta[0][None], rest])


def _wins_only_row(log_alpha, wins):
    """Consume one spurious observed word (INSERT); emit no intended word this step.

    ``new_alpha[k+1] = log_alpha[k] + wins[k]``; ``new_alpha[0] = -inf`` (must advance consumption).
    ``wins`` is a scalar (uniform) or an ``(M,)`` per-observed-word vector (see ``_word_row_update``)."""
    M = log_alpha.shape[0] - 1
    wins_vec = jnp.broadcast_to(jnp.asarray(wins, log_alpha.dtype), (M,))
    return jnp.concatenate([jnp.array([-jnp.inf], dtype=log_alpha.dtype), log_alpha[:-1] + wins_vec])


def _ess(log_w):
    w = jax.nn.softmax(log_w)
    return 1.0 / jnp.sum(w * w)
