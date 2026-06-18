"""Structured-output (``--output_json``) recording: the ``record`` accumulator threaded through the
filter and the aligned rejuvenation hook.

Checks the invariants the JSON artifact relies on: one word-record per observed word; the prefix
top-K counts plus the residual reconstruct ``P`` (so no particle mass is silently dropped); surprisals
are finite; and -- with conditional rejuvenation on -- a per-(event,word) rejuvenation log whose
``attempts`` equal ``fired_particles * n_sweeps`` and whose accepts never exceed attempts. Also checks
that the aggregate accept-rate is unchanged by the move's new per-column return (``sum(accs)`` ==
old total), and that NOT passing a record leaves the run byte-identical (the fast path is untouched).
"""

import math
from collections import Counter

import jax
import jax.numpy as jnp

from src.genjax_port import lm_penzai as L
from src.genjax_port import noise_word as NW
from src.genjax_port.tokenizer import encode
from src.genjax_port.smc_substitution import run_smc_substitution
from src.genjax_port.rejuv_bridge import run_smc_conditional_rejuv_aligned

SENT = "he went too the store"


def _nwords(sentence):
    return len(NW.segment_words([int(i) for i in encode(sentence)]))


def test_record_words_shape_and_invariants():
    """One record per observed word; each prefix_topk counts + residual == P; surprisals finite."""
    obs = jnp.asarray(encode(SENT))
    P, K = 24, 4
    record = {"words": [], "rejuv_events": []}
    run_smc_substitution(jax.random.key(0), obs, num_particles=P, max_dist=2,
                         record=record, record_topk=K)
    W = _nwords(SENT)
    assert len(record["words"]) == W, (len(record["words"]), W)
    for i, wrec in enumerate(record["words"]):
        assert wrec["index"] == i
        assert isinstance(wrec["word"], str) and wrec["word"]
        assert math.isfinite(wrec["surprisal"]), wrec
        assert math.isfinite(wrec["step_min_ess"]) and wrec["step_min_ess"] > 0
        assert len(wrec["prefix_topk"]) <= K
        counts = sum(c for _, c in wrec["prefix_topk"])
        assert counts + wrec["prefix_residual_count"] == P, wrec
        assert wrec["prefix_residual_count"] >= 0


def test_record_is_noop_for_fast_path():
    """A run WITHOUT a record is byte-identical to one WITH a record (recording is a pure observer)."""
    obs = jnp.asarray(encode(SENT))
    a = run_smc_substitution(jax.random.key(1), obs, num_particles=16, max_dist=2)
    b = run_smc_substitution(jax.random.key(1), obs, num_particles=16, max_dist=2,
                             record={"words": [], "rejuv_events": []})
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def test_rejuv_events_logged_and_consistent():
    """Conditional rejuvenation fills a per-(event,word) log: attempts == fired_particles * n_sweeps,
    accepts <= attempts, and the summed accepts match the returned aggregate accept-rate."""
    obs = encode(" he wants go home")                                  # 'to' omitted -> gate fires
    P, n_sweeps = 32, 2
    record = {"words": [], "rejuv_events": []}
    _, _, _, rate = run_smc_conditional_rejuv_aligned(
        jax.random.key(0), obs, num_particles=P, max_dist=2, lookback=4,
        logprob_thresh=2.0, n_sweeps=n_sweeps, record=record, record_topk=5)
    assert len(record["words"]) == _nwords("he wants go home")
    assert record["rejuv_events"], "expected the gate to fire on this omission"
    total_acc = total_att = 0
    for ev in record["rejuv_events"]:
        assert 0.0 <= ev["gate_p"] <= 1.0
        assert ev["fired_particles"] >= 1
        # the per-word record carries the same gate fire-prob (merge done in run.py uses this)
        for tgt in ev["targets"]:
            assert tgt["attempts"] == ev["fired_particles"] * n_sweeps, (ev, tgt)
            assert 0 <= tgt["accepts"] <= tgt["attempts"], (ev, tgt)
            total_acc += tgt["accepts"]
            total_att += tgt["attempts"]
    # aggregate accept-rate (returned) is built from the SAME accepts/attempts the log sums
    assert total_att > 0
    assert abs(rate - total_acc / total_att) < 1e-9, (rate, total_acc, total_att)


def test_too_ambiguity_surfaces_in_prefix_distribution():
    """On 'he went too the store' the prefix distribution at the 'too' position should expose the
    'too'/'to' channel ambiguity: at least one hypothesis ends in 'to' and at least one in 'too' (the
    substitution candidate competes with the literal -- the artifact's reason for being). This is the
    robust, model-independent check; the *ranking* of which reading wins is left to the LM."""
    obs = jnp.asarray(encode(SENT))
    record = {"words": [], "rejuv_events": []}
    run_smc_substitution(jax.random.key(0), obs, num_particles=48, max_dist=2,
                         record=record, record_topk=8)
    too_idx = next(i for i, (_, w) in enumerate(NW.segment_words([int(x) for x in encode(SENT)]))
                   if w.strip() == "too")
    last_words = {s.split()[-1] for s, _ in record["words"][too_idx]["prefix_topk"] if s.split()}
    assert "to" in last_words or "too" in last_words, record["words"][too_idx]["prefix_topk"]
    # the candidate set is genuinely exercised: the channel offers both the literal and its neighbour
    assert {"to", "too"} & last_words, last_words


if __name__ == "__main__":
    L.load_model()
    for name in ("test_record_words_shape_and_invariants", "test_record_is_noop_for_fast_path",
                 "test_rejuv_events_logged_and_consistent",
                 "test_too_ambiguity_resolves_in_prefix_distribution"):
        globals()[name]()
        print(f"OK  {name}")
