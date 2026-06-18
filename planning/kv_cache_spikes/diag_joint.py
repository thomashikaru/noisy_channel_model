"""Full-sentence joint score: log P_LM(intended) + best-alignment channel cost."""
import sys
import jax, jax.numpy as jnp
from genjax_port import lm_penzai, tokenizer
from genjax_port import pythia_word_caprop as P

def log(*a): print(*a); sys.stdout.flush()
lm_penzai.load_model()
EOS = lm_penzai.EOS_ID
seed = [EOS] + tokenizer.encode(".")
WDEL = -8.0

def tid(word):
    ids = tokenizer.encode(" " + word); assert len(ids)==1,(word,ids); return ids[0]

def lm_seq_logp(words):
    """sum log P_LM over the intended word sequence + final EOS, given the '.' seed."""
    ids = [tid(w) if w!="." else tokenizer.encode(".")[0] for w in words]
    total, ctx = 0.0, list(seed)
    parts=[]
    for t in ids + [EOS]:
        buf=jnp.array([ctx+[EOS]],jnp.int32); ln=jnp.array([len(ctx)],jnp.int32)
        lp=float(lm_penzai.next_token_logprobs(buf,ln)[0][t]); total+=lp; parts.append(lp); ctx=ctx+[t]
    return total, parts, ids

def chan(obs,intended):
    oc,_=P._char_ids(obs); ic,ni=P._char_ids(intended)
    return float(P.channel_logpdf(jnp.array(oc,jnp.int32),jnp.array(ic,jnp.int32),ni))

# observed words
obs = ["the","cat","sat","on","mat","."]

def report(name, intended_words, alignment):
    """alignment: list over intended words of ('copy',obs) / ('sub',obs) / ('del',None)."""
    lm_total, parts, _ = lm_seq_logp(intended_words)
    ch_total = 0.0; ch_desc=[]
    for w,(kind,o) in zip(intended_words, alignment):
        if kind=="del":
            ch_total += WDEL; ch_desc.append(f"{w}:DEL({WDEL})")
        else:
            c = chan(o,w); ch_total += c; ch_desc.append(f"{w}<-{o}:{c:.1f}")
    log(f"\n=== {name}: {' '.join(intended_words)}")
    log(f"   LM  total = {lm_total:7.2f}   " + " ".join(f"{w}:{p:.1f}" for w,p in zip(intended_words+['EOS'],parts)))
    log(f"   CHAN total= {ch_total:7.2f}   " + " ".join(ch_desc))
    log(f"   JOINT     = {lm_total+ch_total:7.2f}")
    return lm_total+ch_total

A = report("A car/no-restore", ["The","car","sat","on","mat","."],
           [("copy","the"),("sub","cat"),("copy","sat"),("copy","on"),("copy","mat"),("copy",".")])
B = report("B cat/restore-the", ["The","cat","sat","on","the","mat","."],
           [("copy","the"),("copy","cat"),("copy","sat"),("copy","on"),("del",None),("copy","mat"),("copy",".")])
C = report("C cat/no-restore",  ["The","cat","sat","on","mat","."],
           [("copy","the"),("copy","cat"),("copy","sat"),("copy","on"),("copy","mat"),("copy",".")])
D = report("D car/restore-the", ["The","car","sat","on","the","mat","."],
           [("copy","the"),("sub","cat"),("copy","sat"),("copy","on"),("del",None),("copy","mat"),("copy",".")])

log("\n--- ranking (higher = better) ---")
for n,v in sorted([("A car/no-restore",A),("B cat/restore-the",B),("C cat/no-restore",C),("D car/restore-the",D)],key=lambda t:-t[1]):
    log(f"   {v:8.2f}  {n}")
