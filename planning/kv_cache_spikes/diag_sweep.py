"""Collapse test: if 'The cat sat on mat.' is an inference (impoverishment) failure, the answer
should (a) vary across seeds at P=128 and (b) converge to the model MAP 'The cat sat on mat.' as P
grows. If it's a model/target issue, A ('car') persists for all P and seeds."""
import sys, time
import jax
from genjax_port import lm_penzai
from genjax_port import pythia_word_caprop as PW
def log(*a): print(*a); sys.stdout.flush()
lm_penzai.load_model()
obs = "The cat sat on mat."

log("=== P=128, seeds 0..5 ===")
for s in range(6):
    st,lw,lz,sl = PW.run(obs, jax.random.PRNGKey(s), P=128)
    top = PW.decode(st,lw,skip=sl,top=1)[0]
    log(f"  seed {s}: logZ={lz:.2f}  top={top[0]!r} (p={top[1]:.2f})")

for Pn in (512, 2000, 8000):
    t0=time.time()
    st,lw,lz,sl = PW.run(obs, jax.random.PRNGKey(0), P=Pn)
    top = PW.decode(st,lw,skip=sl,top=3)
    log(f"\n=== P={Pn} (seed 0, logZ={lz:.2f}, {time.time()-t0:.0f}s) ===")
    for s,p in top: log(f"   p={p:.2f}  {s!r}")
