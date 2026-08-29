# Plan: noisy-channel experimental harness

## High-level goal: use the genjax noisy-channel model to simulate behavior patterns from the noisy-channel literature. 

## List of noisy-channel patterns
1. Missing word restoration. Example: "The mother gave the candle the daughter." -> "The mother gave the candle to the daughter." (Gibson et al., 2013)
1. Extra word deletion. Example: "The mother gave the daughter to the candle." -> "The mother gave the daughter the candle." (Gibson et al., 2013)
1. Form-based substitution/paraphasia restoration. Example: "The medics tried to prevent an inflection." -> "The medics tried to prevent an infection." (Ryskin et al., 2021)
1. Ambiguous agreement error correction. Example: "The gifts for the kid was hidden under the table." (Qian & Levy, 2023)
1. Classic garden-path sentences via SAP benchmark. (Huang et al., 2024)
1. Noisy-channel garden-path sentences. Example: "The boy licked the big round ball into the net." -> "The boy kicked the big round ball into the net." (Clark et al., 2026)
1. Form-based substitution with "local coherence". Example: "The coach smiled at the player tossed the frisbee." -> "The coach smiled as the player tossed the frisbee." (Tabor et al., 2004) [Note: I currently don't have these materials, so this is optional and low priority relative to the others, but it would be great if you could track them down].

## Key requirements
1. Efficient, usable harness that processes each of the above datasets and runs the genjax noisy channel model on it with the default parameter settings. 
1. Data in the data/ directory, per study, are currently in their original formats. They first need to be standardized and harmonized into a common format that can be run by the harness. 
1. Relevant model outputs should be saved in an analysis-friendly, stable format: posterior distribution over intended sentences, conditional on noisy observation; per-word surprisals; per-word posterior probability that word is an error; rejuvenation acceptance rate for each word. 
1. Experiments should be run on SLURM ORCD cluster using the orcd skill. Connecting will require me to set up the SSH master manually, once per session. Try not to kill the master or accidentally disconnect so I don't have to babysit. 
1. Reproducibility and transparency are key -- someone should be able to replicate exactly what we did on their own cluster, and easily extend it to a new dataset. Document everything, track versions and random seeds, etc. 
1. Always start with a smoke test on a small amount of data before committing to a huge expensive run. 
1. The order of operations is to first run the model on the stimuli, blind to the human data. Once the model results are in, only then will we compare to human behavior. We are interested in comparing to human inferences and human reading times. You do not need to worry about touching human data until we explicitly green-light that phase.

## Hypotheses
1. Surprisal from noisy-channel model predicts human processing difficulty (i.e. reading times, reading regression rates) for noisy sentences better than standard surprisal.
1. Inferences from noisy-channel model (the prediction for the latent/inferred message given the noisy input) should qualitatively align with human behavior. It should align with the human preference for inferential readings of noisy sentences and the asymmetry between insertions and deletions in the Gibson et al. materials, the effect of supportive context in Chen et al., the error corrections of Ryskin et al., etc. 
1. Novel predictions for classic garden-path sentences. The model may infer the presence of missing words or punctuation in classic garden-path sentences, e.g. a missing "that" in "the horse raced past the barn fell", or missing punctuation, e.g. "when the girl attacked the lamb was calm", a missing comma after "attacked". This would present a novel explanation of how humans parse noisy-channel sentences, not necessarily as syntactic reanalysis always, but by inferring missing units. 