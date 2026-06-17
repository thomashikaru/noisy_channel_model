#!/usr/bin/env bash
# Run the PAIR-HMM channel-aware noisy-channel SMC (the reframed model) on one sentence and print
# the inferred intended-sentence alternatives + runtime. This is src/genjax_port/pythia_word_caprop.py
# -- the PYTHIA CONFIG of the unified filter src/genjax_port/pairhmm_smc.py (the toy bigram used by
# tests/test_pairhmm_exact.py is the SAME filter with a different LM; see planning/PAIRHMM_RBSMC_PLAN.md).
#
# How it differs from run_example_native.sh: instead of SAMPLING the edit-alignment between intended
# and observed strings, this marginalizes it with a nested pair-HMM forward DP -- so only the intended
# sentence is sampled, left-to-right, from the LM, and each particle carries a fixed-shape word-level
# forward vector alpha[k] = P(intended prefix, k observed words consumed). The next intended word is
# drawn from the channel-aware (fully-adapted) proposal over a small candidate set scored by
# LM x channel-evidence, so the importance weights have near-zero variance. No rejuvenation.
#
# Key correctness piece: the forward DP is BANDED (|k - t| <= band, i.e. config.MAX_DELETIONS applied
# to the marginalized vector). Without it the intermediate SMC target collapses to the pure LM prior
# and the filter drifts to high-probability boilerplate regardless of particle count; the band forces
# each intended prefix to keep pace with consuming observed words. Channel + indel priors are the
# production values (normalized char channel; spurious word = 1/V; missing word = P_DELETE_PRIOR=0.005).
# Two further anti-drift pieces: (1) each observed word's OWN token is always in its candidate set (the
# COPY branch), so a correctly-spelled word can be emitted, not only reached via the LM; (2) a content-
# neutral ". " (period + SPACE) primes the LM out of its document-START distribution -- the space
# matters (it tokenizes differently from "."). For a hard sentence, prime harder with a full neutral
# carrier sentence by editing PRIME in pythia_word_caprop.py.
#
# NB pythia-70m is weak: it substitutes/keeps short words freely, and won't remove a doubled word it
# scores near 1/V. Set NC_LM=EleutherAI/pythia-410m for the sharper LM (~6x slower). P=128 corrects the
# tested examples ('teh cat sat on teh mat' -> 'the cat sat on the mat', 'i want go home' -> '... to ...').
#
#   Usage:  ./run_example_hmm.sh [particles] [band] [max_edit_distance]
#   e.g.    ./run_example_hmm.sh 128
#           ./run_example_hmm.sh 256 2 3

# ---- edit these ----
SENTENCE="teh cat sat on teh mat"
PARTICLES="${1:-128}"            # number of SMC particles (P=128 corrects the tested examples)
BAND="${2:-2}"                   # bounded-indel band |k-t|<=band (=MAX_DELETIONS on the forward vector)
MAX_DIST="${3:-2}"               # max char edit distance for word-substitution candidates (SymSpell)
TOP=5                            # number of inferred alternatives to print
SEED=0
NC_LM="${NC_LM:-EleutherAI/pythia-70m}"   # set NC_LM=EleutherAI/pythia-410m for the stronger LM
# --------------------

cd "$(dirname "$0")"
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh
conda activate ncgenjax
export TOKENIZERS_PARALLELISM=false

NC_LM="$NC_LM" PYTHONPATH=src python -m genjax_port.pythia_word_caprop \
  --sentence "$SENTENCE" --particles "$PARTICLES" --band "$BAND" \
  --max_dist "$MAX_DIST" --top "$TOP" --seed "$SEED" \
  2>&1 | grep -vEi "warning|tqdm|fork|tokenizers|avoid using|explicitly set"
