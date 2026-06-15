"""M5 bridge (v1) tests: post-sweep rejuvenation on the filtering-sweep's particles.

The keystone test ties the bridge to the sweep: the chain trace the bridge materializes scores the
same per-word evidence the filtering sweep does, so the MH move provably targets the sweep's
posterior. The others check the zero-sweep identity and that the full round-trip (sweep -> chain
trace -> rejuvenate -> write back) preserves valid single-token reconstructions.

    NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python -m src.genjax_port.tests.test_rejuv_bridge
"""

from collections import Counter

import jax
import jax.numpy as jnp

from src.genjax_port import lm_penzai as L
from src.genjax_port import noise_word as NW
from src.genjax_port.particle_filter import ACTION_ALPHAS
from src.genjax_port.tokenizer import encode, decode
from src.genjax_port.smc_substitution import run_smc_substitution, word_log_evidence
from src.genjax_port.rejuvenation import make_chain_model
from src.genjax_port.rejuv_bridge import (
    _single_token_words, _word_candidate_tables, rejuvenate_particles, run_smc_rejuv,
    run_smc_conditional_rejuv,
)
from genjax import ChoiceMap as C

# A sentence whose every observed word is a single BPE token (asserted in the tests).
SENT = "he wants to go home"


def test_chain_importance_matches_sweep_evidence():
    """The materialized chain trace's joint == the sweep's per-word evidence at the literal reading.

    This is the correctness keystone: with the bridge's candidate tables the chain model's
    importance weight (sum over words of lm(x|ctx) + channel) equals the sweep's COPY-column
    word_log_evidence summed over words, so the rejuvenation move targets the sweep's posterior.
    """
    obs_ids = encode(SENT)
    words, obs = _single_token_words(obs_ids)
    W = len(words)
    ap = jnp.log(jnp.asarray(ACTION_ALPHAS, jnp.float32) / sum(ACTION_ALPHAS))
    lap = ap[None, :]                                       # deterministic 1-particle prior
    cand_xs, cand_ls = _word_candidate_tables(words, lap, max_dist=2)

    M = 1 + W + 6
    buf0 = jnp.full(M, L.EOS_ID, jnp.int32)
    ilen0 = jnp.array(1, jnp.int32)
    model = make_chain_model(W)
    chm = C.d({**{f"x{t}": jnp.int32(obs[t]) for t in range(W)},
               **{f"o{t}": jnp.int32(obs[t]) for t in range(W)}})
    _, w = model.importance(jax.random.key(0), chm, (buf0, ilen0, cand_xs, cand_ls[0]))

    # Sweep evidence at the literal (copy) reading, threading the observed tokens into the buffer.
    buf = jnp.full((1, M), L.EOS_ID, jnp.int32)
    il = jnp.array([1], jnp.int32)
    total = 0.0
    for t, (span_ids, word_str) in enumerate(words):
        subs = NW.word_sub_candidates(word_str, max_dist=2)
        ev = word_log_evidence(buf, il, lap, span_ids, subs, L.next_token_logprobs)
        total += float(ev[0, 0])                           # COPY column
        buf = buf.at[0, il[0]].set(obs[t])
        il = il + 1
    assert abs(float(w) - total) < 1e-2, (float(w), total)


def test_rejuv_zero_sweeps_is_identity():
    """n_sweeps=0 leaves every particle's reconstruction (and the accept rate) untouched."""
    obs = jnp.asarray(encode(SENT))
    P = 8
    _, _, _, (buf, _, lap) = run_smc_substitution(jax.random.key(0), obs, num_particles=P,
                                                  max_dist=2, return_state=True)
    new_buf, rate = rejuvenate_particles(jax.random.key(1), buf, lap, obs, max_dist=2, n_sweeps=0)
    assert rate == 0.0
    W = len(_single_token_words(obs)[0])
    before = [decode(buf[p, 1:1 + W]).strip() for p in range(P)]
    after = [decode(new_buf[p, 1:1 + W]).strip() for p in range(P)]
    assert before == after, (before[:3], after[:3])


def test_run_smc_rejuv_roundtrip():
    """Full sweep -> rejuvenate round-trip yields valid single-token sentences, accept in [0,1],
    and does not lose mass on the clean literal reading (rejuvenation targets the same posterior)."""
    obs = jnp.asarray(encode(SENT))
    P = 16
    base, _, _ = run_smc_substitution(jax.random.key(0), obs, num_particles=P, max_dist=2)
    sents, _, _, rate = run_smc_rejuv(jax.random.key(0), obs, num_particles=P, max_dist=2,
                                      n_sweeps=2)
    assert 0.0 <= rate <= 1.0, rate
    assert len(sents) == P and all(isinstance(s, str) and s for s in sents)
    lit = " ".join(SENT.split())
    base_lit = Counter(base)[lit]
    rejuv_lit = Counter(sents)[lit]
    # The literal is the dominant reading of clean text; rejuvenation must not collapse it.
    assert rejuv_lit >= base_lit - 2, (base_lit, rejuv_lit)


def test_conditional_gate_off_no_moves():
    """A huge surprisal threshold => the per-particle gate never fires => no MH attempts (rate 0),
    and clean text still resolves to the literal reading."""
    obs = jnp.asarray(encode(SENT))
    sents, _, _, rate = run_smc_conditional_rejuv(jax.random.key(0), obs, num_particles=16,
                                                  max_dist=2, lookback=4, logprob_thresh=1e6,
                                                  n_sweeps=2)
    assert rate == 0.0, rate
    lit = " ".join(SENT.split())
    assert Counter(sents)[lit] >= 12, Counter(sents).most_common(2)


def test_conditional_rejuv_runs_and_fires():
    """A very low threshold => gate fires => interleaved moves are attempted (rate in [0,1]); output
    stays valid single-token sentences and does not collapse the clean literal reading."""
    obs = jnp.asarray(encode(SENT))
    P = 16
    base, _, _ = run_smc_substitution(jax.random.key(0), obs, num_particles=P, max_dist=2)
    sents, _, _, rate = run_smc_conditional_rejuv(jax.random.key(0), obs, num_particles=P,
                                                  max_dist=2, lookback=3, logprob_thresh=-1e6,
                                                  logprob_spread=1.0, n_sweeps=1)
    assert 0.0 <= rate <= 1.0, rate
    assert len(sents) == P and all(isinstance(s, str) and s for s in sents)
    lit = " ".join(SENT.split())
    assert Counter(sents)[lit] >= Counter(base)[lit] - 3, (Counter(base)[lit], Counter(sents)[lit])


def test_multitoken_word_rejected():
    """v1 scope guard: a multi-token observed word raises a clear ValueError."""
    obs = jnp.asarray(encode("the boy did an experimemt today"))  # 'experimemt' is multi-token
    raised = False
    try:
        _single_token_words(obs)
    except ValueError:
        raised = True
    assert raised, "expected ValueError for a multi-token observed word"


if __name__ == "__main__":
    L.load_model()
    for name in ("test_chain_importance_matches_sweep_evidence", "test_rejuv_zero_sweeps_is_identity",
                 "test_run_smc_rejuv_roundtrip", "test_conditional_gate_off_no_moves",
                 "test_conditional_rejuv_runs_and_fires", "test_multitoken_word_rejected"):
        globals()[name]()
        print(f"OK  {name}")
