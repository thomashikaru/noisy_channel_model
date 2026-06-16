"""M1 driver test: the substitution SMC's per-word evidence == the native Switch model's
branch importances, plus a behavioral run over the substitution suite.

The cross-check (``word_log_evidence`` vs ``make_word_model`` branch ``importance``) is the link
that justifies the lean filtering sweep: it computes the same joint the native ``@gen`` model
scores, just directly (one forward + gather) instead of through ``Switch`` branches. It is
LM-independent, so it holds for any NC_LM.

Run the behavioral suite (slow; use 410m to match the golden reference) with::

    NC_LM=EleutherAI/pythia-410m PYTHONPATH=. python -m src.genjax_port.tests.test_smc_substitution
"""

from collections import Counter

import jax
import jax.numpy as jnp

from src.genjax_port import lm_penzai as L
from src.genjax_port import noise_word as NW
from src.genjax_port.config import ACTION_ALPHAS
from src.genjax_port.model import COPY, SUB
from src.genjax_port.genjax_model import make_word_model, word_constraints
from src.genjax_port.smc_substitution import word_log_evidence, run_smc_substitution
from src.genjax_port.tokenizer import encode


def test_evidence_matches_word_model_branch_importances():
    """word_log_evidence[0, c] == make_word_model branch-c importance (deterministic prior)."""
    ctx = encode("the boy did an")
    span_ids = encode(" experimemt")
    n = len(span_ids)
    M = len(ctx) + n + 4
    buf = jnp.full((1, M), L.EOS_ID, jnp.int32).at[0, jnp.arange(1, 1 + len(ctx))].set(jnp.asarray(ctx))
    ilen = jnp.asarray([1 + len(ctx)], jnp.int32)
    subs = NW.word_sub_candidates("experimemt", max_dist=2)
    ap = jnp.log(jnp.asarray(ACTION_ALPHAS, jnp.float32) / sum(ACTION_ALPHAS))
    log_action_prior = ap[None, :]  # [1, 3], deterministic to match the word-model test

    log_ev = word_log_evidence(buf, ilen, log_action_prior, span_ids, subs,
                               L.next_token_logprobs)[0]  # [1 + n_sub]

    model = make_word_model(n, len(subs))
    copy_args = (buf[0], ilen[0], ap[COPY])
    sub_args = [(buf[0], ilen[0], ap[SUB] + NW.word_sub_loglik(d)) for _, d in subs]

    _, w_copy = model.importance(jax.random.key(0), word_constraints(0, span_ids, None),
                                 (0, copy_args, *sub_args))
    assert abs(float(log_ev[0]) - float(w_copy)) < 1e-2, (float(log_ev[0]), float(w_copy))
    for k, (x, _) in enumerate(subs):
        _, w_k = model.importance(jax.random.key(k + 1), word_constraints(k + 1, span_ids, x),
                                  (k + 1, copy_args, *sub_args))
        assert abs(float(log_ev[1 + k]) - float(w_k)) < 1e-2, (k, float(log_ev[1 + k]), float(w_k))


def test_bucket_padding_is_inert():
    """Fixing max_intended to a bucket leaves the discrete posterior identical to per-sentence
    sizing. EOS-padded positions beyond i_len don't affect causal-attention logits at i_len-1, so
    every downstream random draw under the same key is the same and the sampled sentences match
    exactly. (log_marginal agrees only up to ~1e-1 float noise: XLA tiles the padded [P, bucket]
    forward differently than [P, M], so the logits differ at ~1e-3 -- benign, never flips a draw.)
    """
    obs = jnp.asarray(encode("did you recieve the message"))
    a, la, _ = run_smc_substitution(jax.random.key(0), obs, num_particles=16, max_dist=2)
    b, lb, _ = run_smc_substitution(jax.random.key(0), obs, num_particles=16, max_dist=2,
                                    max_intended=24)
    assert a == b, (a[:3], b[:3])               # discrete posterior: exact match
    assert abs(la - lb) < 0.5, (la, lb)         # log-marginal: equal up to XLA float noise


def test_run_smc_batch_validates_bucket():
    """required_buffer_size is respected: a too-small bucket raises before running."""
    from src.genjax_port.smc_substitution import run_smc_batch, required_buffer_size
    obs = [jnp.asarray(encode(s)) for s in ("the boy ran", "did you recieve the message")]
    assert required_buffer_size(obs[1]) > 3
    raised = False
    try:
        run_smc_batch(jax.random.key(0), obs, bucket=3, num_particles=8)
    except ValueError:
        raised = True
    assert raised, "expected ValueError for an undersized bucket"


def test_dedup_matches_plain_and_fires():
    """Dedup is numerically exact and actually collapses rows.

    Same RNG => the deduped run reproduces the plain-forward run's discrete posterior exactly and
    its log-marginal up to XLA float noise; ``DedupStats.saved_frac > 0`` confirms identical
    intended-prefix rows really were collapsed to one LM call (the perf win this exists for).
    """
    from src.genjax_port.cache_dedup import DedupStats
    obs = jnp.asarray(encode("did you recieve the message"))
    plain, lm_plain, _ = run_smc_substitution(jax.random.key(0), obs, num_particles=32,
                                              max_dist=2, dedup=False)
    stats = DedupStats()
    ddup, lm_ddup, _ = run_smc_substitution(jax.random.key(0), obs, num_particles=32,
                                            max_dist=2, dedup=True, dedup_stats=stats)
    assert ddup == plain, (ddup[:3], plain[:3])     # discrete posterior: exact match
    assert abs(lm_plain - lm_ddup) < 0.5, (lm_plain, lm_ddup)   # log-marginal: XLA float noise
    assert stats.saved_frac > 0.0, stats            # dedup actually fired


def test_clean_text_stays_literal_smoke():
    """Smoke test: clean text is dominated by the literal reading (any Pythia LM)."""
    obs = jnp.asarray(encode("the boy did an experiment today"))
    sents, _, _ = run_smc_substitution(jax.random.key(0), obs, num_particles=16, max_dist=2)
    top, count = Counter(sents).most_common(1)[0]
    assert top == "the boy did an experiment today", (top, count)
    assert count >= 12, count  # >= ~75% of 16 particles stay literal


def _behavioral_suite():
    """Substitution (+ M2 deletion) suite vs the golden idealized behaviors (P=64, 410m)."""
    # (observed, ideal, max_deletions, allow_insertion)
    suite = [
        ("the boy did an experimemt today", "experimemt -> experiment (dominant)", 0, False),
        ("did you recieve the message", "recieve -> receive (weak at 410m)", 0, False),
        ("the boy did an experiment today", "clean: stay literal", 0, False),
        ("he wants go home", "deletion: reconstruct omitted 'to' (~0.5, low ESS)", 1, False),
        ("the boy handed handed the pencil to the girl",
         "insertion: remove the doubled 'handed' (~0.5)", 0, True),
    ]
    # Fixed buffer bucket so every sentence shares the compiled [P, bucket] forward (the latency
    # bucketing optimization): only the first sentence pays the ~8s/410m compile, the rest run warm.
    # All suite sentences need <= 13 (required_buffer_size); 16 is a tight bucket.
    bucket = 16
    for observed, ideal, max_del, allow_ins in suite:
        obs = jnp.asarray(encode(observed))
        sents, logm, ess = run_smc_substitution(jax.random.key(0), obs, num_particles=64,
                                                 max_intended=bucket, max_dist=2,
                                                 max_deletions=max_del,
                                                 allow_insertion=allow_ins, progress=False)
        tags = [f"max_deletions={max_del}"] if max_del else []
        tags += ["insertion"] if allow_ins else []
        tag = f" [{', '.join(tags)}]" if tags else ""
        print(f"\nobserved : {observed}{tag}\nideal    : {ideal}\nlogP~={logm:.1f} minESS={ess:.1f}/64")
        total = len(sents)
        for s, c in Counter(sents).most_common(5):
            print(f"  {c/total:6.1%}  ({c:>3d})  {s}")


if __name__ == "__main__":
    L.load_model()
    test_evidence_matches_word_model_branch_importances()
    print("OK  evidence matches word-model branch importances")
    test_clean_text_stays_literal_smoke()
    print("OK  clean text stays literal (smoke)")
    _behavioral_suite()
