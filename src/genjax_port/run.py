"""Run the token-level Genjax noisy-channel particle filter on a sentence.

Usage (from repo root, in the `ncgenjax` conda env):

    python -m src.genjax_port.run
    python -m src.genjax_port.run --sentence "the boy handed the pencil too the girl" \\
        --particles 64 --seed 0

Given an observed (possibly noisy) sentence -- now *arbitrary* text via the Pythia BPE
tokenizer -- it prints the posterior over inferred *intended* sentences, ranked by particle
frequency.
"""

import argparse
from collections import Counter

import jax

from .particle_filter_unified import run_particle_filter_unified
from .tokenizer import encode

DEFAULT_SENTENCE = "the boy handed the pencil to the girl"


def summarize(sentences, top_k=5):
    counts = Counter(sentences)
    total = len(sentences)
    return [(s, c, c / total) for s, c in counts.most_common(top_k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument(
        "--max_dist",
        type=int,
        default=2,
        help="max character edit distance for word-substitution candidates "
        "(SymSpell). Distance is not hard-capped at modeling time -- "
        "SUB_PARAM**d down-weights far candidates; this only bounds "
        "candidate retrieval.",
    )
    parser.add_argument(
        "--filter",
        choices=["unified", "native"],
        default="unified",
        help="'unified' = the hand-rolled reference filter (default); "
        "'native' = the genjax-native word-scan SMC (smc_substitution).",
    )
    parser.add_argument(
        "--max_deletions",
        type=int,
        default=1,
        help="[native] max omitted-word reconstructions per gap (0 disables "
        "deletion). The unified filter uses MAX_DELETIONS.",
    )
    parser.add_argument(
        "--no_insertion",
        action="store_true",
        help="[native] disable the INSERT action (spurious-word removal).",
    )
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="[native] disable LM-forward dedup (on by default, as in the unified "
        "filter). Dedup is numerically exact; disabling it is for A/B timing.",
    )
    parser.add_argument(
        "--rejuvenate",
        action="store_true",
        help="[native] run post-sweep rejuvenation (full-context MH substitution-flip "
        "reanalysis) on the particles. v1: single-token words, substitution-only "
        "(forces deletion/insertion off).",
    )
    parser.add_argument(
        "--rejuv_sweeps",
        type=int,
        default=1,
        help="[native] number of MH sweeps per rejuvenation event (--rejuvenate / --conditional_rejuv).",
    )
    parser.add_argument(
        "--conditional_rejuv",
        action="store_true",
        help="[native] interleaved surprisal-gated rejuvenation during the sweep (vectorized over "
        "particles). v1: single-token words, substitution-only. Overrides --rejuvenate.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=2,
        help="[native] lookback window (words) for --conditional_rejuv reanalysis.",
    )
    parser.add_argument(
        "--logprob_thresh",
        type=float,
        default=0.0,
        help="[native] gate center on (contextual - unigram) surprisal for --conditional_rejuv "
        "(higher => rejuvenate less; 0 fires when a word is more surprising in context than its "
        "base rate).",
    )
    parser.add_argument(
        "--logprob_spread",
        type=float,
        default=1.0,
        help="[native] surprisal gate steepness for --conditional_rejuv.",
    )
    parser.add_argument(
        "--add_delete",
        action="store_true",
        help="[native] post-sweep add/delete (R2) reanalysis: a substitution-only sweep, then a "
        "trans-dimensional MH pass that inserts omitted / removes spurious words using full-sentence "
        "context. v1: single-token words; multi-token sentences fall back to the native filter.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        metavar="PATH",
        help="[native] write a structured JSON artifact of the run (per-word surprisal / unigram "
        "surprisal / gate fire-prob / step ESS, the top-K inferred-prefix distribution per word, the "
        "final posterior, and -- with --conditional_rejuv -- a per-(event,word) rejuvenation log) to "
        "PATH. Off by default (zero change to the fast path). Supported for --filter native, with or "
        "without --conditional_rejuv.",
    )
    parser.add_argument(
        "--json_topk",
        type=int,
        default=5,
        help="[native] K for the per-word inferred-prefix distribution and the final posterior in "
        "--output_json.",
    )
    args = parser.parse_args()

    # Structured-output accumulator (filled in by the two supported native paths; see --output_json).
    record = None
    if args.output_json:
        if args.filter == "native" and not (args.add_delete or args.rejuvenate):
            record = {"words": [], "rejuv_events": []}
        else:
            print(
                "warning: --output_json is only supported for --filter native (optionally with "
                "--conditional_rejuv); ignoring it for this run."
            )

    obs = encode(args.sentence)
    key = jax.random.key(args.seed)
    accept_rate = None
    if args.filter == "native" and args.conditional_rejuv:
        # genjax-native sweep (copy/substitution/deletion/insertion) with interleaved, surprisal-gated
        # SUBSTITUTION rejuvenation, vectorized over particles. The forward filter does add/delete; the
        # rejuvenation revises substitutions, locating each word's token via the per-particle alignment
        # so forward deletions/insertions don't misalign it. Works on any sentence.
        from .rejuv_bridge import run_smc_conditional_rejuv_aligned

        sentences, log_marginal, min_ess, accept_rate = run_smc_conditional_rejuv_aligned(
            key,
            jax.numpy.array(obs),
            num_particles=args.particles,
            max_dist=args.max_dist,
            lookback=args.lookback,
            logprob_thresh=args.logprob_thresh,
            logprob_spread=args.logprob_spread,
            n_sweeps=args.rejuv_sweeps,
            max_deletions=args.max_deletions,
            allow_insertion=not args.no_insertion,
            dedup=not args.no_dedup,
            record=record,
            record_topk=args.json_topk,
            progress=True,
        )
    elif args.filter == "native" and args.add_delete:
        # genjax-native sweep + post-sweep add/delete (R2) reanalysis: a trans-dimensional MH pass that
        # inserts omitted / removes spurious words with full-sentence context. With --rejuvenate also
        # set, the same pass ALSO runs the R1 substitution-flip per word, so one post-sweep reanalysis
        # revises both substitutions and add/deletes (the maximal post-sweep move). v1: single-token
        # words; multi-token sentences fall back to the native filter.
        from .rejuv_bridge import run_smc_add_delete

        sentences, log_marginal, min_ess, accept_rate = run_smc_add_delete(
            key,
            jax.numpy.array(obs),
            num_particles=args.particles,
            max_dist=args.max_dist,
            n_sweeps=args.rejuv_sweeps,
            sub_flip=args.rejuvenate,
            dedup=not args.no_dedup,
            progress=True,
        )
    elif args.filter == "native" and args.rejuvenate:
        # genjax-native sweep + post-sweep substitution-flip rejuvenation (full-context reanalysis).
        # v1: single-token words, substitution-only; raises on multi-token words.
        from .rejuv_bridge import run_smc_rejuv

        sentences, log_marginal, min_ess, accept_rate = run_smc_rejuv(
            key,
            jax.numpy.array(obs),
            num_particles=args.particles,
            max_dist=args.max_dist,
            n_sweeps=args.rejuv_sweeps,
            dedup=not args.no_dedup,
            progress=True,
        )
    elif args.filter == "native":
        # genjax-native word-scan SMC: copy / substitution / deletion / insertion. Per-sentence
        # buffer sizing here (one sentence => one compile); use run_smc_batch for corpora.
        from .smc_substitution import run_smc_substitution

        sentences, log_marginal, min_ess = run_smc_substitution(
            key,
            jax.numpy.array(obs),
            num_particles=args.particles,
            max_dist=args.max_dist,
            max_deletions=args.max_deletions,
            allow_insertion=not args.no_insertion,
            dedup=not args.no_dedup,
            record=record,
            record_topk=args.json_topk,
            progress=True,
        )
    else:
        # Unified word-scan filter: copy / substitution (incl. BPE-token-count typos) / insertion /
        # deletion in one model, with dedup LM forwards by default.
        sentences, log_marginal, min_ess = run_particle_filter_unified(
            key,
            jax.numpy.array(obs),
            num_particles=args.particles,
            max_dist=args.max_dist,
            progress=True,
        )

    norm_observed = " ".join(args.sentence.split())
    print(f"observed : {args.sentence}")
    rejuv_note = f"   rejuv accept = {accept_rate:.1%}" if accept_rate is not None else ""
    print(
        f"particles: {args.particles}   log P(observed) ~= {log_marginal:.3f}"
        f"   min ESS = {min_ess:.1f}/{args.particles}{rejuv_note}\n"
    )
    print("inferred intended sentences (posterior over alternatives):")
    for sent, count, frac in summarize(sentences, top_k=args.top_k):
        marker = "  <- matches observed" if sent == norm_observed else ""
        print(f"  {frac:5.1%}  ({count:>3d})  {sent}{marker}")

    if record is not None:
        import json

        from . import lm_penzai as L
        from .unigram import unigram_surprisal

        # Merge the deterministic per-word unigram surprisal (the gate's baseline) and, when the
        # rejuvenation hook ran, each event's gate fire-probability into the per-word records.
        for wrec in record["words"]:
            wrec["unigram_surprisal"] = unigram_surprisal(wrec["word"])
        for ev in record["rejuv_events"]:
            record["words"][ev["t"]]["gate_p"] = ev["gate_p"]
        out = {
            "observed": args.sentence,
            "config": {
                "lm": L.MODEL_NAME, "particles": args.particles, "max_dist": args.max_dist,
                "max_deletions": args.max_deletions, "allow_insertion": not args.no_insertion,
                "conditional_rejuv": args.conditional_rejuv, "lookback": args.lookback,
                "logprob_thresh": args.logprob_thresh, "logprob_spread": args.logprob_spread,
                "n_sweeps": args.rejuv_sweeps, "seed": args.seed, "json_topk": args.json_topk,
                # counts == posterior probs only because we resample every word (uniform weights).
                "counts_are_probs": True,
            },
            "log_marginal": log_marginal,
            "min_ess": min_ess,
            "accept_rate": accept_rate,
            "posterior": [[s, c, f] for s, c, f in summarize(sentences, top_k=args.json_topk)],
            "words": record["words"],
        }
        if record["rejuv_events"]:        # omit entirely for forward-only runs (no hook fired)
            out["rejuv_events"] = record["rejuv_events"]
        with open(args.output_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
