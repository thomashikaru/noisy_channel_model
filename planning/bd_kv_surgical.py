"""Surgical probe: is the indel move's ~110s/run dominated by COMPILE (per-run retrace/lower of the big
lax.map graph -> Cause 3, fix = lru_cache the sweep) or steady-state EXEC (the LM forwards -> Cause 1,
fix = KV suffix-tail rewrite, only ~1.5-3x)?

Monkeypatch make_gibbs_indel_sweep so the (single) post-loop invocation calls the built sweep TWICE on
the identical inputs and times each. Call #1 = compile + exec (fresh jitted _logits). Call #2 reuses the
SAME sweep object's SAME jitted _logits -> JAX jit-cache hit -> exec only. So:
    compile = call1 - call2      exec = call2
Low P for speed; the compile/exec split is P-robust (compile ~ shape-only; exec scales with P).
"""
import os, time

import jax

from genjax_port import pythia_word_caprop as W
from genjax_port import lm_penzai
from genjax_port import pairhmm_rejuv as RJ
from genjax_port.pythia_word_caprop import ALIGN_ALPHA_DEFAULT

P = int(os.environ.get("NC_P", "64"))
A = "the cat sat on on the warm mat"

_orig = RJ.make_gibbs_indel_sweep


def _patched(*a, **k):
    inner = _orig(*a, **k)

    def sweep(*sa, **sk):
        t = time.time(); r1 = inner(*sa, **sk); jax.block_until_ready(r1); t1 = time.time() - t
        t = time.time(); r2 = inner(*sa, **sk); jax.block_until_ready(r2); t2 = time.time() - t
        print(f"\n[indel-sweep]  call1 (compile+exec) = {t1:7.1f}s   call2 (exec only) = {t2:7.1f}s"
              f"   => COMPILE = {t1 - t2:7.1f}s   EXEC = {t2:7.1f}s\n", flush=True)
        return r1

    return sweep


RJ.make_gibbs_indel_sweep = _patched


def _wellform(s):
    s = s.strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?":
        s = s + "."
    return s


def main():
    print(f"=== bd_kv_surgical: indel-sweep compile-vs-exec (P={P}) ===", flush=True)
    lm_penzai.load_model()
    t = time.time()
    st, lw, logZ, sl = W.run(_wellform(A), jax.random.PRNGKey(0), P=P, band=2,
                             action_alpha=ALIGN_ALPHA_DEFAULT, rejuv="gibbs+bd", dedup=True,
                             channel="align", bd_mode="gibbs", bd_attempts=1, bd_funcwords=True)
    jax.block_until_ready((lw, logZ))
    print(f"full gibbs+bd run (len {len(A.split())}, P={P}) = {time.time()-t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
