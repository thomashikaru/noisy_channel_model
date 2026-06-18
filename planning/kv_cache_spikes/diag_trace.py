"""Instrumented copy of pairhmm_smc.run loop: per-step, print ESS, resample events, and the
population of the just-emitted intended word (decoded), to see where 'cat' dies vs 'car'."""
import sys
from collections import Counter
import jax, jax.numpy as jnp
from jax.scipy.special import logsumexp
from genjax_port import lm_penzai, tokenizer, pairhmm_smc
from genjax_port import pythia_word_caprop as PW
from genjax_port.poc_word_indel import _ess

def log(*a): print(*a); sys.stdout.flush()
lm_penzai.load_model()

observed = "The cat sat on mat."
P = 128
key = jax.random.PRNGKey(0)
model = PW._pythia_model(PW.PRIME, None, False)
WDEL = -8.0
WINS = float(__import__("genjax_port.noise", fromlist=["insertion_loglik"]).insertion_loglik(model.emit_vocab))

# replicate run() internals
import genjax_port.pairhmm_smc as S
from genjax import ChoiceMap
band, slack, max_dist, Ke, J, cwin = 2, 3, 2, 8, 8, 1
seed_ids = list(model.seed_ids); seed_len=len(seed_ids)
obs_words = model.obs_words(observed); M=len(obs_words)
log("obs words:", obs_words)
obs_char = jnp.stack([jnp.asarray(model.char_ids(w)[0],jnp.int32) for w in obs_words])
emit_full = jax.vmap(jax.vmap(model.channel_logpdf,in_axes=(None,0,0)),in_axes=(0,None,None))(
    obs_char, model.vocab_char, model.vocab_clen)
emit_tab = S._emit_table(model, obs_words, max_dist, Ke)
offs = jnp.arange(-cwin,cwin+1); eos_id,Vc = model.eos_id, model.emit_vocab
LCTX = seed_len+M+slack+1
kernel, band_mask = S._make_kernel(seed_len, M, band, WDEL, WINS)
constraint = ChoiceMap.d({"ev": jnp.float32(0.0)})
ks=jnp.arange(M+1)
a0 = band_mask(jnp.where(ks==0,0.0,ks*WINS),0)
ctx0 = jnp.full((P,LCTX),eos_id,jnp.int32).at[:,:seed_len].set(jnp.array(seed_ids,jnp.int32))
state=(ctx0, jnp.full(P,seed_len,jnp.int32), jnp.broadcast_to(a0,(P,M+1)), jnp.zeros(P,bool))
log_w=jnp.zeros(P)

def assemble(ctx_len, log_alpha, lmlog):
    ne = ctx_len - seed_len
    return jax.vmap(lambda la,lm,n: S._caprop_scores(la,lm,emit_tab,emit_full,offs,J,M,band_mask,
        n+1,eos_id,Vc,WDEL,WINS,model.word_mask), in_axes=(0,0,0))(log_alpha,lmlog,ne)

@jax.jit
def extend(keys,cb,cl,la,dn,cand,emit_cols,scores):
    def one(k,cb,cl,la,dn,c,ec,sc):
        tr,w=kernel.importance(k,constraint,((cb,cl,la,dn),c,ec,sc))
        rv=tr.get_retval(); return rv[0],rv[1],rv[2],rv[3],w
    return jax.vmap(one)(keys,cb,cl,la,dn,cand,emit_cols,scores)

def lastword_pop(state):
    cb,cl,_,_=state
    out=Counter()
    for p in range(P):
        n=int(cl[p])
        w = tokenizer.decode([int(cb[p][n-1])]).strip() if n>seed_len else "<seed>"
        out[w]+=1
    return out

for step in range(M+slack):
    cb,cl,la,dn=state
    lmlog=model.lm_fn(cb,cl)
    key,sub=jax.random.split(key); keys=jax.random.split(sub,P)
    cand,emit_cols,scores=assemble(cl,la,lmlog)
    cb2,cl2,la2,dn2,incr=extend(keys,cb,cl,la,dn,cand,emit_cols,scores)
    state=(cb2,cl2,la2,dn2)
    log_w=log_w+incr
    pop=lastword_pop(state)
    ess=float(_ess(log_w))
    msg=f"step {step}: ESS={ess:6.1f}  lastword={dict(pop.most_common(6))}"
    if ess < 0.5*P:
        key,sub=jax.random.split(key)
        anc=jax.random.categorical(sub,log_w,shape=(P,))
        state=jax.tree_util.tree_map(lambda a:a[anc],state)
        log_w=jnp.zeros(P)
        msg+="  <-- RESAMPLED"
    log(msg)

# final decode
top = pairhmm_smc.decode(state, log_w, model, skip=seed_len, top=5)
log("\nfinal decode:")
for s,p in top: log(f"   p={p:.2f}  {s!r}")
