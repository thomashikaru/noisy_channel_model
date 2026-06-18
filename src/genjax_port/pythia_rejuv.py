"""R1 validation harness: apply the capacity-parametric Gibbs/SMCP3 rejuvenation sweep
(:mod:`pairhmm_rejuv`) to the Pythia pair-HMM filter, and check it cures the ``cat/mat``
impoverishment collapse diagnosed in ``planning/kv_cache_spikes/``.

This is NOT the production interleave (that is R2 -- a windowed sweep inside the SMC loop behind a
flag). Here we run the certified filter to completion, RESAMPLE the cloud to equal weights, then run
the post-resample Gibbs sweep and decode -- the cleanest place to show the sweep recovers diversity
that resampling collapsed. The candidate pool per intended slot ``i`` is ``_candidate_words`` (COPY +
SymSpell) of the observed word ``i`` (1:1 alignment -- the dominant, substitution-aligned particles;
the per-particle COPY in ``gibbs_sweep`` keeps misaligned particles unchanged).

Supersedes the archived sampled-alignment ``pythia_rejuv`` stub (rejuv_bridge / manual_subflip_move),
which the REJUV_KV_REDESIGN paradigm replaces.

Run:  NC_LM=EleutherAI/pythia-70m PYTHONPATH=src python -m genjax_port.pythia_rejuv [P] [seeds] [sweeps]
"""

import time

import jax
import jax.numpy as jnp

from genjax_port import lm_penzai, pairhmm_rejuv as rejuv
from genjax_port import pythia_word_caprop as PW
from genjax_port.noise import insertion_loglik


def recover(observed, key, P=128, n_sweeps=2, max_dist=2, Ke=8, band=2, wdel=None, wins=None,
            positions="content"):
    """Run the filter, resample to equal weights, sweep, and return (before, after, logZ).

    ``positions="content"`` sweeps the M observed-aligned slots (skips the trailing slack slots)."""
    model = PW._pythia_model(PW.PRIME)
    ntok = model.emit_vocab
    WDEL = PW.WDEL_DEFAULT if wdel is None else wdel
    WINS = insertion_loglik(ntok) if wins is None else wins

    st, lw, logZ, sl = PW.run(observed, key, P=P, max_dist=max_dist, Ke=Ke, band=band,
                              wdel=WDEL, wins=WINS)
    ctx_buf, ctx_len = st[0], st[1]   # state tuple grew (Phase D); take the buffer + token length

    # resample to an equally-weighted cloud (the sweep is a Gibbs move on equal weights). "before" is
    # this resampled cloud -- post-collapse, so before/after differ only by the sweep (honest contrast).
    key, sub = jax.random.split(key)
    anc = jax.random.categorical(sub, lw, shape=(P,))
    ctx_buf, ctx_len = ctx_buf[anc], ctx_len[anc]
    before = rejuv.decode_counts(ctx_buf, ctx_len, model, sl)

    ctx = rejuv.make_rejuv_ctx(observed, model, WDEL, WINS, band=band)
    pool_tok, pool_len = rejuv.build_pool(observed, model, max_dist, Ke, ctx.Wmax)
    pos = range(ctx.M) if positions == "content" else None
    for _ in range(n_sweeps):
        key, sub = jax.random.split(key)
        ctx_buf, _, _ = rejuv.gibbs_sweep(sub, ctx_buf, ctx_len, ctx, pool_tok, pool_len, positions=pos)
    after = rejuv.decode_counts(ctx_buf, ctx_len, model, sl)
    return before, after, logZ


def _top(d, k=1):
    return sorted(d.items(), key=lambda kv: -kv[1])[:k]


def bench(observed, seeds=range(3), P=128, rejuv_lookback=3, max_dist=2):
    """R3 cost/quality measurement: run the filter with rejuv off vs in-loop gibbs (KV suffix tail
    scorer) on the same sentence/seeds, and report wall-clock, the **full-forward balloon** (the KV
    sweep does 1 shared prefill per word per particle -- a full forward -- plus cheap single-token
    tail steps; the comparable cost is `(filter_forwards + sweep_prefills)/filter_forwards`), the
    cheap tail-step count, the cloud degeneracy (unique/P), and the MAP. Each gibbs run pays the
    sweep-step JIT compile once (an R3-remaining item), so wall-clock overstates steady-state cost."""
    import statistics
    lm_penzai.load_model()
    print(f"obs={observed!r}  P={P}  lookback={rejuv_lookback}", flush=True)
    print(f"{'seed':>4} {'t_off':>7} {'t_gibbs':>7} {'fwd x':>6} {'uniq/P':>7}  MAP off -> gibbs", flush=True)
    balloons, uniqs, tails = [], [], []
    for s in seeds:
        t0 = time.time()
        st, lw, _, sl = PW.run(observed, jax.random.PRNGKey(s), P=P, max_dist=max_dist, rejuv="off")
        off_map = PW.decode(st, lw, skip=sl, top=1)[0]
        t_off = time.time() - t0

        stats = {}
        t0 = time.time()
        st, lw, _, sl = PW.run(observed, jax.random.PRNGKey(s), P=P, max_dist=max_dist,
                               rejuv="gibbs", rejuv_lookback=rejuv_lookback, rejuv_stats=stats)
        g_map = PW.decode(st, lw, skip=sl, top=1)[0]
        t_g = time.time() - t0

        filt = stats["filter_lm_calls"]
        balloon = (filt + stats["sweep_prefills"]) / max(filt, 1)
        uf = stats["uniq_frac"]
        balloons.append(balloon)
        tails.append(stats["sweep_tail_steps"])
        if uf:
            uniqs.extend(uf)
        print(f"{s:>4} {t_off:7.1f} {t_g:7.1f} {balloon:6.2f} "
              f"{statistics.median(uf) if uf else float('nan'):7.2f}  "
              f"{off_map[0]!r} ({off_map[1]:.2f}) -> {g_map[0]!r} ({g_map[1]:.2f})", flush=True)
    print(f"\nmean full-forward balloon x{statistics.mean(balloons):.2f} "
          f"(KV: filter forwards + shared prefills; was x151 whole-sentence in R2)  "
          f"+ {statistics.mean(tails):.0f} cheap tail-steps/run; "
          f"median unique/P {statistics.median(uniqs) if uniqs else float('nan'):.2f}", flush=True)


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "bench":      # in-loop cost/quality measurement
        P = int(sys.argv[2]) if len(sys.argv) > 2 else 128
        nseeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        bench("The cat sat on mat.", seeds=range(nseeds), P=P)
        return
    lm_penzai.load_model()
    obs = "The cat sat on mat."
    truth = "The cat sat on mat."
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    seeds = range(int(sys.argv[2])) if len(sys.argv) > 2 else range(6)
    n_sweeps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    print(f"obs={obs!r}  truth={truth!r}  P={P}  n_sweeps={n_sweeps}", flush=True)
    t0 = time.time()
    n_ok_before = n_ok_after = 0
    for s in seeds:
        before, after, logZ = recover(obs, jax.random.PRNGKey(s), P=P, n_sweeps=n_sweeps)
        b, bp = _top(before)[0]
        a, ap = _top(after)[0]
        ok_b, ok_a = (b == truth), (a == truth)
        n_ok_before += ok_b
        n_ok_after += ok_a
        print(f"[{time.time()-t0:6.1f}s] seed {s}: "
              f"before {'OK ' if ok_b else 'BAD'} p={bp:.2f} {b!r}  ->  "
              f"after {'OK ' if ok_a else 'BAD'} p={ap:.2f} {a!r}", flush=True)
    n = len(list(seeds))
    print(f"\nMAP==truth: before {n_ok_before}/{n}  ->  after {n_ok_after}/{n}", flush=True)


if __name__ == "__main__":
    main()
