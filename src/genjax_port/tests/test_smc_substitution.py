"""M1 driver test: the substitution SMC's per-word evidence == the native Switch model's
branch importances, plus a behavioral run over the substitution suite.

The cross-check (``word_log_evidence`` vs ``make_word_model`` branch ``importance``) is the link
that justifies the lean forward filter: it computes the same joint the native ``@gen`` model
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
from src.genjax_port.particle_filter import ACTION_ALPHAS
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


def test_clean_text_stays_literal_smoke():
    """Smoke test: clean text is dominated by the literal reading (any Pythia LM)."""
    obs = jnp.asarray(encode("the boy did an experiment today"))
    sents, _, _ = run_smc_substitution(jax.random.key(0), obs, num_particles=16, max_dist=2)
    top, count = Counter(sents).most_common(1)[0]
    assert top == "the boy did an experiment today", (top, count)
    assert count >= 12, count  # >= ~75% of 16 particles stay literal


def _behavioral_suite():
    """Substitution (+ M2 deletion) suite vs the golden idealized behaviors (P=64, 410m)."""
    suite = [
        ("the boy did an experimemt today", "experimemt -> experiment (dominant)", 0),
        ("did you recieve the message", "recieve -> receive (weak at 410m)", 0),
        ("the boy did an experiment today", "clean: stay literal", 0),
        ("he wants go home", "deletion: reconstruct omitted 'to' (~0.5, low ESS)", 1),
    ]
    for observed, ideal, max_del in suite:
        obs = jnp.asarray(encode(observed))
        sents, logm, ess = run_smc_substitution(jax.random.key(0), obs, num_particles=64,
                                                 max_dist=2, max_deletions=max_del, progress=False)
        tag = f" [max_deletions={max_del}]" if max_del else ""
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
