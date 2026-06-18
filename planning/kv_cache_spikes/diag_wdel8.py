import sys, jax
from genjax_port import lm_penzai
from genjax_port import pythia_word_caprop as PW
def log(*a): print(*a); sys.stdout.flush()
lm_penzai.load_model()
obs="The cat sat on mat."
log("=== WDEL=-8 (script config), P=128 seeds 0..3 ===")
for s in range(4):
    st,lw,lz,sl=PW.run(obs, jax.random.PRNGKey(s), P=128, wdel=-8.0)
    t=PW.decode(st,lw,skip=sl,top=1)[0]; log(f"  seed {s}: top={t[0]!r} p={t[1]:.2f}")
for Pn in (2000, 8000):
    st,lw,lz,sl=PW.run(obs, jax.random.PRNGKey(0), P=Pn, wdel=-8.0)
    log(f"=== WDEL=-8, P={Pn} ===")
    for s,p in PW.decode(st,lw,skip=sl,top=3): log(f"   p={p:.2f}  {s!r}")
