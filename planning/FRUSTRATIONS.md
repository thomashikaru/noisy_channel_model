Here are some frustrations with the entire current genjax_port project. 

We have been working on this together for the last 3-4 days continuously. The goal, as stated in the planning docs, is to translate the original Gen.jl noisy-channel model into GenJAX for greater performance and Python interoperability. As of now:

- The genjax model can't properly handle multi-token words. Rejuvenation moves are useless if the original word and/or replacement word consist of multiple BPE tokens. 
- Runtimes with rejuvenation are vastly slower than without rejuvenation. 
- We spent hours introducing fixes that supposedly would boost performance, such as: the dedup feature, the KV-prefix-cache, etc. A lot of these changes are somewhat haphazardly used (e.g. the dedup is used in the forward but not in rejuvenations), and the KV-prefix-cache drastically increases compile time.
- We don't end up using many features of GenJAX at all, such as the @gen decorator. We end up re-creating a lot of its functionality from scratch, including manually calculating weights and ratios. 