import sys, time, jax
from genjax_port import lm_penzai
from genjax_port import pythia_word_caprop as PW
def log(*a): print(*a); sys.stdout.flush()
t0=time.time(); lm_penzai.load_model(); log(f"loaded {lm_penzai.MODEL_NAME} in {time.time()-t0:.0f}s")
obs="The cat sat on mat."
for Pn in (512,):
    for wd in (-8.0, -9.0):
        st,lw,lz,sl=PW.run(obs, jax.random.PRNGKey(0), P=Pn, wdel=wd)
        log(f"=== {lm_penzai.MODEL_NAME} P={Pn} wdel={wd} logZ={lz:.2f} ===")
        for s,p in PW.decode(st,lw,skip=sl,top=4): log(f"   p={p:.2f}  {s!r}")
