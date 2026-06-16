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
from src.genjax_port.tokenizer import encode, decode, surface
from src.genjax_port.smc_substitution import run_smc_substitution, word_log_evidence
from src.genjax_port.rejuvenation import make_chain_model
from src.genjax_port.rejuv_bridge import (
    _single_token_words, _word_candidate_tables, rejuvenate_particles, run_smc_rejuv,
    run_smc_conditional_rejuv, run_smc_add_delete,
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


def test_multitoken_falls_back_to_plain_filter():
    """A multi-token word is outside v1 rejuv scope but must NOT break: rejuvenation is skipped
    (accept_rate 0) and the plain substitution filter runs -- never less capable than the filter."""
    import warnings
    obs = jnp.asarray(encode("the boy did an experimemt today"))  # 'experimemt' is multi-token
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sents, _, _, rate = run_smc_conditional_rejuv(jax.random.key(0), obs, num_particles=8,
                                                      max_dist=2, logprob_thresh=-1e6)
    assert rate == 0.0, rate
    assert len(sents) == 8 and all(isinstance(s, str) and s for s in sents)


def test_add_delete_recovers_omitted_word():
    """Post-sweep R2 (add/delete) recovers a word the substitution-only sweep cannot insert.

    'he wants go home' has 'to' omitted. Run the sweep with max_dist=0 (no substitution candidates,
    so the forward pass commits the literal tokens and cannot mangle them on the weak 70m LM); the
    post-sweep add/delete pass should insert 'to' before 'go' in the majority of particles."""
    obs = encode(" he wants go home")
    sents, _, _, rate = run_smc_add_delete(jax.random.key(0), obs, num_particles=32,
                                           max_dist=0, n_sweeps=3)
    counts = Counter(sents)
    assert counts["he wants to go home"] > 16, counts.most_common(5)   # clear majority recovered
    assert rate > 0.0


def test_dyn_step_matches_static():
    """The position-independent dynamic step (one compile for all words) == the static add/delete
    move, bit-for-bit on the add/delete part. This is what lets the post-sweep reuse one compiled
    graph for every word instead of recompiling per position (~150s -> one ~35s compile)."""
    from src.genjax_port.rejuvenation_r2 import gap_chain_inputs, add_delete_step
    from src.genjax_port.rejuv_bridge import _dyn_step_fn, _materialize_fn, _gap_choices
    from src.genjax_port.particle_filter import P_DELETE_PRIOR
    obs = encode(" he wants go home")
    W, P, k, pdel = 4, 6, 2, float(P_DELETE_PRIOR)
    _, obsl, buf0, ilen0, cxs, cls = gap_chain_inputs(obs)
    obsa = jnp.asarray(obsl, jnp.int32)
    xr = jnp.tile(obsa, (P, 1))
    trs = _materialize_fn(W, pdel)(jax.random.split(jax.random.key(1), P), xr,
                                   buf0, ilen0, cxs, cls, obsa)
    keys = jax.random.split(jax.random.key(7), P)
    dtrs, dacc = _dyn_step_fn(W, 6, pdel, False)(jnp.int32(k), keys, trs,
                                                 buf0, ilen0, cxs, cls, obsa)
    dd, _, _ = _gap_choices(dtrs, W)
    for p in range(P):
        tp = jax.tree_util.tree_map(lambda a: a[p], trs)
        t2, _, a = add_delete_step(keys[p], tp, k, buf0, ilen0, cxs, cls, obsa)
        sd = jnp.array([t2.get_choices()[f"del{t}"] for t in range(W)])
        assert bool((dd[p] == sd).all()) and bool((dacc[p] > 0) == bool(a)), p


def test_manual_subflip_detailed_balance():
    """The manual (buffer-based) sub-flip move samples the exact posterior of the revisited word's
    token (MH detailed balance == Thm 2). The suffix-vote term of the weight is separately exercised
    by test_aligned_conditional_composes_with_forward_deletions and shares the R1 move's math."""
    import math
    from jax.scipy.special import logsumexp
    from src.genjax_port.lm_genjax import lm_logp
    from src.genjax_port.rejuv_bridge import manual_subflip_move, _word_candidate_tables
    from src.genjax_port.particle_filter import ACTION_ALPHAS
    obs = list(encode(" he too"))
    words, _ = _single_token_words(jnp.asarray(obs))
    P, M = 48, 8
    lap = jnp.log(jnp.broadcast_to(jnp.array(ACTION_ALPHAS, jnp.float32) / sum(ACTION_ALPHAS), (P, 3)))
    cand_xs, cand_ls = _word_candidate_tables(words, lap, max_dist=2)
    cx, cl = cand_xs[1], cand_ls[:, 1]                                  # revisit word 1 (the last word)
    # exact posterior over word-1 candidates: lm(x1 | EOS, x0) + channel
    base = jnp.full(M, L.EOS_ID, jnp.int32).at[1].set(obs[0])
    lm1 = lm_logp(base, jnp.int32(2))
    cands = [(int(cx[i]), float(cl[0, i])) for i in range(cx.shape[0]) if float(cl[0, i]) > -1e20]
    logp = {c: float(lm1[c]) + clc for c, clc in cands}
    Z = float(logsumexp(jnp.array(list(logp.values()))))
    exact = {c: math.exp(v - Z) for c, v in logp.items()}

    buf = jnp.full((P, M), L.EOS_ID, jnp.int32).at[:, 1].set(obs[0]).at[:, 2].set(obs[1])
    i_len = jnp.full((P,), 3, jnp.int32)
    pos = jnp.full((P,), 2, jnp.int32)
    gate = jnp.ones((P,), bool)
    key, counts = jax.random.key(0), Counter()
    for _ in range(200):
        key, sk = jax.random.split(key)
        buf, _ = manual_subflip_move(sk, buf, i_len, pos, cx, cl, gate)
        for p in range(P):
            counts[int(buf[p, 2])] += 1
    n = sum(counts.values())
    err = max(abs(exact[c] - counts.get(c, 0) / n) for c in exact)
    assert err < 0.05, (err, {surface(c): (round(exact[c], 3), round(counts.get(c, 0) / n, 3))
                              for c in exact if exact[c] > 0.02})


def test_aligned_conditional_composes_with_forward_deletions():
    """Forward filter with deletions ON (so per-particle alignment shifts) + interleaved aligned
    sub-rejuvenation: recovers the omitted 'to', stays coherent (the rejuvenation locates each word's
    token via the alignment instead of a broken 1:1 slot), and the move fires."""
    from src.genjax_port.rejuv_bridge import run_smc_conditional_rejuv_aligned
    obs = encode(" he wants go home")                                  # 'to' omitted before 'go'
    sents, _, _, rate = run_smc_conditional_rejuv_aligned(
        jax.random.key(0), obs, num_particles=48, max_dist=2, lookback=4,
        logprob_thresh=2.0, n_sweeps=2)
    counts = Counter(sents)
    assert counts["he wants to go home"] >= 16, counts.most_common(5)   # omission recovered, coherent
    assert rate > 0.0
    assert all(isinstance(s, str) and s for s in sents)


def test_add_delete_multitoken_falls_back():
    """A multi-token word falls back to the native filter (deletions on), accept_rate 0, never errors."""
    import warnings
    obs = encode("the boy did an experimemt today")                    # 'experimemt' is multi-token
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sents, _, _, rate = run_smc_add_delete(jax.random.key(0), obs, num_particles=8, max_dist=2)
    assert rate == 0.0 and len(sents) == 8 and all(isinstance(s, str) and s for s in sents)


if __name__ == "__main__":
    L.load_model()
    for name in ("test_chain_importance_matches_sweep_evidence", "test_rejuv_zero_sweeps_is_identity",
                 "test_run_smc_rejuv_roundtrip", "test_conditional_gate_off_no_moves",
                 "test_conditional_rejuv_runs_and_fires", "test_multitoken_falls_back_to_plain_filter",
                 "test_add_delete_recovers_omitted_word", "test_dyn_step_matches_static",
                 "test_manual_subflip_detailed_balance",
                 "test_aligned_conditional_composes_with_forward_deletions",
                 "test_add_delete_multitoken_falls_back"):
        globals()[name]()
        print(f"OK  {name}")
