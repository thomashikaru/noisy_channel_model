"""Model's TRUE target for a fixed intended sentence: LM_prior + marginal channel (forward DP
with band + WDEL deletions + WINS insertion sweep + terminal alpha[M]). This is exactly what the
SMC integrates -- if it ranks C above A, the SMC is mis-sampling (inference), not the model."""
import sys
import jax, jax.numpy as jnp
from jax.scipy.special import logsumexp
from genjax_port import lm_penzai, tokenizer
from genjax_port import pythia_word_caprop as PW
from genjax_port.poc_word_indel import _word_row_update
from genjax_port.noise import insertion_loglik

def log(*a): print(*a); sys.stdout.flush()
lm_penzai.load_model()
EOS = lm_penzai.EOS_ID
seed = [EOS] + tokenizer.encode(".")
WDEL = -8.0
BAND = 2
DOT = tokenizer.encode(".")[0]

def tid(w): return DOT if w=="." else tokenizer.encode(" "+w)[0]

observed = "The cat sat on mat."
obs_words = PW._obs_word_units(observed)             # ['the','cat','sat','on','mat','.']
M = len(obs_words)
WINS = insertion_loglik(len(tokenizer.vocab_strings()))
log("observed words:", obs_words, " M=",M, " WINS=%.2f"%WINS)

# channel emission column for a given intended word id against every observed word
vocab_char, vocab_clen = PW._vocab_char_table()
obs_char = jnp.stack([jnp.asarray(PW._char_ids(w)[0], jnp.int32) for w in obs_words])
def emit_col(intended_id):
    ic = vocab_char[intended_id]; ni = vocab_clen[intended_id]
    return jax.vmap(lambda oc: PW.channel_logpdf(oc, ic, ni))(obs_char)   # (M,)

ks = jnp.arange(M+1)
def band_mask(a, t): return jnp.where(jnp.abs(ks - t) <= BAND, a, -jnp.inf)

def marginal_channel(intended_ids):
    a = band_mask(jnp.where(ks==0, 0.0, ks*WINS), 0)
    for t, wid in enumerate(intended_ids, start=1):
        a = band_mask(_word_row_update(a, emit_col(wid), WDEL, WINS), t)
    return float(a[M])                                # terminal full consumption

def lm_prior(intended_ids):
    total, ctx = 0.0, list(seed)
    for t in intended_ids + [EOS]:
        buf=jnp.array([ctx+[EOS]],jnp.int32); ln=jnp.array([len(ctx)],jnp.int32)
        total += float(lm_penzai.next_token_logprobs(buf,ln)[0][t]); ctx=ctx+[t]
    return total

def score(name, words):
    ids = [tid(w) for w in words]
    lm = lm_prior(ids); ch = marginal_channel(ids)
    log(f"  {name:22} LM={lm:8.2f}  CHAN_marg={ch:8.2f}  JOINT={lm+ch:8.2f}   {' '.join(words)}")
    return lm+ch

log("\n--- model's true target (marginal over alignments) ---")
res = {
 "C cat/no-restore": score("C cat/no-restore", ["The","cat","sat","on","mat","."]),
 "A car/no-restore": score("A car/no-restore", ["The","car","sat","on","mat","."]),
 "B cat/restore-the": score("B cat/restore-the", ["The","cat","sat","on","the","mat","."]),
 "A' car/restore":   score("A' car/restore", ["The","car","sat","on","the","mat","."]),
}
log("\n  ranking (higher=better):")
for n,v in sorted(res.items(), key=lambda t:-t[1]): log(f"     {v:8.2f}  {n}")
