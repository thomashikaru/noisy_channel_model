import torch, math
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m"); m = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m").eval()
for i in (0, 15, 187, 510, 380, 3101, 253, 28725, 6122, 209, 535):
    print(f"  id {i:5d} -> {tok.decode([i])!r}")
lit = [0, 15, 380, 3101, 3534, 253, 28725, 253, 6122, 15]
nl  = [0, 15, 187, 187, 510, 3101, 3534, 253, 28725, 253, 6122, 15]
def score(ids):
    x = torch.tensor([ids]); lp = torch.log_softmax(m(x).logits[0].double(), -1)
    per = [(tok.decode([ids[i]]), float(lp[i - 1, ids[i]])) for i in range(2, len(ids))]   # tokens after the 2-token prime
    return per, sum(v for _, v in per)
for name, ids in (("literal  ", lit), ("newline  ", nl)):
    per, tot = score(ids); print(f"{name} decode={tok.decode(ids[2:])!r}\n   total logP(after prime)={tot:.2f}   per-token: " + " ".join(f"{t!r}:{v:.2f}" for t, v in per))
pl, tl = score(lit); pn, tn = score(nl)
print(f"LM gain of the newline path = {tn - tl:+.2f} nats (channel must charge two deletions + the no-space 'The' copy)")
