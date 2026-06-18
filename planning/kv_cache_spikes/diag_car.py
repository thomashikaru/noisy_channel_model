"""Diagnose why 'The cat sat on mat.' -> 'The car sat on mat.' under pythia-70m."""
import sys
import jax, jax.numpy as jnp
from genjax_port import lm_penzai, tokenizer
from genjax_port import pythia_word_caprop as P

def log(*a): print(*a); sys.stdout.flush()

lm_penzai.load_model()
EOS = lm_penzai.EOS_ID
seed = [EOS] + tokenizer.encode(".")   # the PRIME

def lp_next(ctx_ids):
    buf = jnp.array([ctx_ids + [EOS]*1], jnp.int32)
    ln  = jnp.array([len(ctx_ids)], jnp.int32)
    return lm_penzai.next_token_logprobs(buf, ln)[0]

def tid(word):  # single word-initial token id
    ids = tokenizer.encode(" " + word)
    assert len(ids) == 1, (word, ids)
    return ids[0]

# ---- 1. LM prior: car vs cat after ". The" ----
ctx = seed + [tid("The")]
lp = lp_next(ctx)
for w in ["cat", "car", "man", "dog", "the"]:
    log(f"  P_LM({w!r:7} | '. The')  logp = {float(lp[tid(w)]):.2f}")

log("\n  -> LM bias car over cat:", f"{float(lp[tid('car')] - lp[tid('cat')]):.2f} nats")

# ---- 2. channel cost: copy(cat->cat) vs sub(cat->car) ----
def chan(obs, intended):
    oc, no = P._char_ids(obs); ic, ni = P._char_ids(intended)
    return float(P.channel_logpdf(jnp.array(oc, jnp.int32), jnp.array(ic, jnp.int32), ni))
log("\n  channel logpdf(obs 'cat' | intended 'cat') =", f"{chan('cat','cat'):.2f}  (copy)")
log("  channel logpdf(obs 'cat' | intended 'car') =", f"{chan('cat','car'):.2f}  (1 sub)")
log("  -> channel penalty for cat->car:",
    f"{chan('cat','cat') - chan('cat','car'):.2f} nats")

# ---- 3. the 'on mat' -> 'on the mat' decision ----
log("\n--- restoring dropped 'the' in 'on ___ mat' ---")
ctx = seed + [tid(w) for w in ["The","cat","sat","on"]]
lp = lp_next(ctx)
for w in ["the","mat","top","a"]:
    log(f"  P_LM({w!r:5} | '. The cat sat on') logp = {float(lp[tid(w)]):.2f}")
log("  WDEL (missing-word penalty, run_example) = -8.0")
