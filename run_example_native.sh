#!/usr/bin/env bash
# Run the UNIFIED pair-HMM RB-SMC noisy-channel model on one sentence and print the inferred
# intended alternatives + runtime. This is the production filter from planning/PAIRHMM_RBSMC_PLAN.md
# (genjax_port.pythia_word_caprop, a thin Pythia config over genjax_port.pairhmm_smc):
#
#   The ONLY sampled latent is the intended sentence, generated left-to-right from Pythia. The
#   edit-alignment (which observed words are substitutions / missing / spurious) is marginalized
#   exactly by a nested pair-HMM forward DP carried in fixed-shape per-particle state -- NOT sampled,
#   so there is no rejuvenation / trans-dimensional MCMC. The proposal is channel-aware (fully
#   adapted): each step scores a candidate set (SymSpell edit-neighbours of the observed words at the
#   alignment frontier + top-J LM words + EOS) by LM logprob + forward-mass increment, so weights are
#   near-zero-variance and small particle counts suffice. The band keeps the intended prefix
#   synchronized with observed consumption (and gives insertions/deletions their reach); there is NO
#   explicit INSERT action -- a spurious word is a channel event marginalized inside the DP.
#
# Correctness is certified on a toy bigram by exact enumeration (src/genjax_port/tests/
# test_pairhmm_exact.py); Pythia is the same filter with the LM swapped in.
#
# Uses pythia-70m for fast iteration (set NC_LM=EleutherAI/pythia-410m for the stronger LM). NB: 70m
# is weak enough that it substitutes/inserts short words fairly freely on some clean sentences.
#
# REJUVENATION (R3): REJUV=gibbs (the default here) runs a post-resample Gibbs/SMCP3 sweep that
# re-diversifies the particle cloud and cures impoverishment collapses (e.g. P=128 flipping a correct
# word to a wrong neighbour across seeds). It improves the flat-posterior inferences at ~a few x the
# runtime (a KV-cached suffix scorer keeps it bounded). Set REJUV=off for the certified forward-only
# filter (faster, what the exact-enumeration tests gate).
#
#   Usage:  ./run_example_native.sh ["sentence"] [particle_count] [max_edit_distance]
#   e.g.    ./run_example_native.sh "We ran an experimemt."
#           ./run_example_native.sh "the cat sat on mat." 256 3
#           REJUV=off ./run_example_native.sh "i want go home"        # certified forward-only filter
#           NC_LM=EleutherAI/pythia-70m ./run_example_native.sh "teh cat"   # faster, weaker LM

# ---- edit these (or pass as CLI args / env vars) ----
SENTENCE="${1:-We ran an experimemt.}"   # observed (noisy) sentence -- pass as the first CLI argument
PARTICLES="${2:-128}"            # number of SMC particles (P=128 is the validated budget; ~flat in P)
MAX_DIST="${3:-2}"               # max char edit distance for word-substitution candidates (SymSpell)
REJUV="${REJUV:-gibbs}"          # "gibbs" = post-resample rejuvenation sweep; "off" = forward-only
REJUV_LOOKBACK="${REJUV_LOOKBACK:-5}"   # rejuvenation window: recent words each sweep revisits
BAND=2                           # |consumed - emitted| tolerance; also the insertion/deletion reach
WDEL=-6                          # missing-word (over-editing) log-penalty in nats. The model "adds"
                                 # words to its reconstruction by positing MISSING words; this is how
                                 # costly that is. More negative => fewer inferred extra words (less
                                 # over-editing) but also less willing to restore genuinely-dropped
                                 # words. -9 curbs over-editing while still restoring e.g. a dropped
                                 # "to"; try -7 (looser) or -11 (stricter, may drop real restorations).
NC_LM="${NC_LM:-EleutherAI/pythia-410m}"    # set NC_LM=EleutherAI/pythia-410m for the stronger LM

# structured-output JSON for the interactive viewer (src/genjax_port/viz.py). Set OUTPUT_JSON="" to
# skip. It records the inferred intended sentence (per-word surprisal heat-map), the top-K intended-
# prefix distribution at each step, and the final posterior. View with:
#   PYTHONPATH=src python -m genjax_port.viz "$OUTPUT_JSON"
OUTPUT_JSON="${OUTPUT_JSON:-run_native.json}"
JSON_TOPK=8                       # hypotheses kept per step + in the final posterior
# --------------------

cd "$(dirname "$0")"
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh
conda activate ncgenjax
export TOKENIZERS_PARALLELISM=false

JSON_ARGS=()
[ -n "$OUTPUT_JSON" ] && JSON_ARGS=(--output_json "$OUTPUT_JSON" --json_topk "$JSON_TOPK")

SECONDS=0
NC_LM="$NC_LM" PYTHONPATH=src python -m genjax_port.pythia_word_caprop \
  --sentence "$SENTENCE" --particles "$PARTICLES" --max_dist "$MAX_DIST" --band "$BAND" \
  --wdel "$WDEL" --rejuv "$REJUV" --rejuv_lookback "$REJUV_LOOKBACK" "${JSON_ARGS[@]}" \
  2>&1 | grep -vEi "warning|tqdm|fork|tokenizers|avoid using|explicitly set|word/s"
echo "runtime: ${SECONDS}s"
[ -n "$OUTPUT_JSON" ] && echo "view: PYTHONPATH=src python -m genjax_port.viz $OUTPUT_JSON"
