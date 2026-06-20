"""Diagnostic for the leading-opener artifact (the model prepends 'In this'/'The'/'I' to lowercase-
initial sentences). Hypothesis: the sentence-INITIAL word is lowercase in the observed input, but after
the '.' prime the LM expects a CAPITALIZED opener; the clean fix (capitalize the actual first word) is
not in the candidate set (SymSpell case-folds it away; the top-J LM bridge surfaces generic openers
ahead of the specific capitalized word), so the filter prepends/swaps a generic opener instead.

Two probes:
  (A) LM next-token distribution right after the seed '<|endoftext|>.': do capitalized openers dominate,
      and where do ' she'/' She' vs ' the'/' The' rank relative to J=8?
  (B) A/B the full word-action filter on lowercase- vs capitalized-initial versions of the same sentence:
      if capitalizing the first letter removes the leading opener, the capitalization root is confirmed.

Run: NC_LM=EleutherAI/pythia-70m PYTHONPATH=src conda run -n ncgenjax python -u -m genjax_port.diag_leading_opener
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from genjax_port import lm_penzai, tokenizer
from genjax_port import pythia_word_caprop as W

# Candidate seeds to sweep (the string after the leading <|endoftext|>). The current prime is ".";
# the others test the '\n' the LM wants after the prime, a bare boundary, and few-shot "transcribe a
# sentence" contexts (which should condition a capitalized sentence start instead of a generic opener).
PRIMES = [".", ".\n", "\n", "", "The cat sat on the mat.\n", "I saw a bird in the tree.\n"]

# First-word options to locate in each distribution: spaced vs unspaced x capitalized vs lowercase.
# (A '\n'-terminated prime makes the next word UNSPACED -- 'She' not ' She' -- so a space-only candidate
#  would mismatch; we report both so any new tokenization mismatch is visible.)
KEYS = [" the", " The", "the", "The", " she", " She", "she", "She", " he", " He", "he", "He"]


def _logprobs_after(seed, LCTX=32):
    buf = np.full((1, LCTX), lm_penzai.EOS_ID, np.int32)
    buf[0, :len(seed)] = seed
    return np.asarray(lm_penzai.next_token_logprobs(jnp.asarray(buf), jnp.asarray([len(seed)])))[0]


def probe_primes():
    EOS = lm_penzai.EOS_ID
    for prime in PRIMES:
        seed = [EOS] + (tokenizer.encode(prime) if prime else [])
        lp = _logprobs_after(seed)              # fixed LCTX -> compiled once, reused across primes
        order = np.argsort(-lp)
        print(f"\n=== prime {prime!r}   seed={tokenizer.decode(seed)!r}  (len {len(seed)}) ===")
        print("  top-6 next: " + "  ".join(
            f"{tokenizer.surface(int(i))!r}({lp[i]:.1f})" for i in order[:6]))
        for s in KEYS:
            ids = tokenizer.encode(s)
            tid = ids[0]
            rank = int(np.where(order == tid)[0][0])
            flag = " <top8" if rank < 8 else ""
            print(f"    {s!r:8s} surf={tokenizer.surface(tid)!r:10s} lp {lp[tid]:7.2f}  rank {rank:5d}{flag}")


def probe_filter():
    pairs = [
        ("she went to the store", "She went to the store"),
        ("he is good man", "He is good man"),
        ("they studied the causal link between the variables",
         "They studied the causal link between the variables"),
    ]
    print("\n\n=== A/B: lowercase- vs capitalized-initial, word-action gibbs alpha=(27,1,1,1) ===")
    for lo, hi in pairs:
        for sent in (lo, hi):
            st, lw, _, sl = W.run(sent, jax.random.PRNGKey(0), P=128, band=2,
                                  action_alpha=(27.0, 1.0, 1.0, 1.0), rejuv="gibbs", dedup=True)
            top = W.decode(st, lw, skip=sl, top=3)
            print(f"\n  {sent!r}")
            for s, p in top:
                print(f"     p={p:.2f}  {s!r}")


def main():
    lm_penzai.load_model()
    probe_primes()                                       # fast: ~one forward pass per prime
    if os.environ.get("NC_FILTER_AB"):                   # heavy: real SMC per sentence (opt-in only)
        probe_filter()


if __name__ == "__main__":
    main()
