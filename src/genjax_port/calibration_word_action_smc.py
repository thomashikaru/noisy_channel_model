"""Close-the-loop (plan WORD_ACTION_CHANNEL_PLAN.md sec 5.5): run battery items through the WORD-ACTION
SMC and compare the per-item correction behaviour to the offline word-action preview (q_hier).

For each item the SMC posterior is reduced to the same readings the preview contests:
  * EDIT items:  q_smc = mass(correction reading E) / (mass(E) + mass(literal L))   -> compare to q_hier
  * KEEP items:  kept  = mass(literal L)  (the over-editing guard; high = good, the model keeps it)
  * junk        = 1 - mass(L) - mass(E): mass on neither clean reading -- spurious-insertion / over-edit
                  hypotheses the 2-reading offline preview structurally cannot see (the diagnostic that
                  surfaced the rejuv='off' failures: leading '1?'/'-' insertions, over-edits).

FINDINGS (2026-06-19, alpha=(3,1,1,1), 410m, P=256, rejuv='off'): does NOT pass. The prior-mean action
probs are edit-happy (p_ins=p_del=p_sub=0.17), and the full filter -- unlike the 2-reading preview --
explores spurious-insertion / over-edit hypotheses the cheap rates permit: leading junk insertions,
DEL/INS items left uncorrected, subject over-edits. The deployed filter needs real theta-INFERENCE (the
posterior, not a prior draw); but theta-rejuv concentrates theta on each particle's CURRENT parse, so a
clean-looking 'I want go home' pulls p_del low and makes the genuine deletion HARDER to restore (a
mode-collapse the Gibbs-from-current-parse move won't escape). This is the open problem (plan sec 6).

PERFORMANCE NOTE: no inference hot-path regression from the word-action code (ON vs OFF +~7%; theta-rejuv
+0.3s, LM-free). Slow batch runs were 410m + dedup-off + P=256; use dedup=True (and the 70m default) for
fast iteration. Pass dedup=True below.

Usage: NC_LM=EleutherAI/pythia-410m PYTHONPATH=src conda run -n ncgenjax python -u \
         -m genjax_port.calibration_word_action_smc P SEED [item_id ...]
"""
import csv
import os
import sys
import time

import jax

from genjax_port import pythia_word_caprop as W
from genjax_port import lm_penzai
from genjax_port.pythia_word_caprop import _norm, ACTION_ALPHA_DEFAULT, ALIGN_ALPHA_DEFAULT

# Channel hook (plan ALIGN_ACTION_CHANNEL_PLAN Phase 4): NC_CHANNEL=align runs the 3-way align channel
# (default word_action, unchanged). NC_ALIGN_SLOPE overrides K (the form per-edit cost); NC_ALPHA still
# overrides the concentration (length-3 'align,ins,del' for align, length-4 for word_action).
CHANNEL = os.environ.get("NC_CHANNEL", "word_action")
ALIGN_SLOPE = float(os.environ["NC_ALIGN_SLOPE"]) if os.environ.get("NC_ALIGN_SLOPE") else None
# Action-latent names for the theta print, per channel: align is the 3-way (align,ins,del); word_action
# is the 4-way (copy,sub,ins,del). (Hardcoding 'c,s,i,d' mislabeled the 3-vector printed for align.)
THETA_LABEL = "align,ins,del" if CHANNEL == "align" else "c,s,i,d"

# Default to the 70m gate for 70m sweeps (item membership / observed / intended are identical to the
# 410m variant -- only the gate columns differ). NC_CSV overrides (e.g. the 410m gate for a 410m run).
CSV = os.environ.get("NC_CSV", "planning/calibration_battery_v0_gated.csv")
META = {r["item_id"]: r for r in csv.DictReader(open(CSV))}
# offline q_hier reference (alpha=(3,1,1,1)) from calibration_word_action_preview, for a few named items.
QREF = {"SUBW-01a": 0.89, "SUBW-03a": 0.63, "SUBN-01a": 0.87, "SUBN-02a": 1.00,
        "DEL-to-05a": 0.96, "INS-to-04a": 0.90, "SUBW-01b": 0.00, "INS-to-01b": 0.00}
DEFAULT_ITEMS = ["SUBW-01a", "SUBW-01b", "SUBN-01a", "SUBN-02a", "DEL-to-05a", "INS-to-04a", "INS-to-01b"]


# Capitalize the sentence-INITIAL letter (default ON; NC_NOCAP=1 to disable). The battery was authored
# mixed-case and many families are lowercase-initial; after the '.' prime a lowercase first word sits at
# ~-14 logprob (rank ~16k), so the LM prepends/swaps a capitalized opener ('In this ...', 'The ...') --
# the leading-opener artifact. Real sentences (and the behavioral stimuli) are capitalized, so this fixes
# malformed test input, not the model. Idempotent on already-capitalized / 'I ...' sentences.
CAP = os.environ.get("NC_NOCAP", "0") not in ("1", "true", "yes")

# Indel rejuv knobs (only active when NC_REJUV=gibbs+bd). BD_MODE: "gibbs" (default, the effective indel
# move -- resample the single edit from its full conditional: amplifies a dropped-word restoration in one
# post-loop sweep, can't over-edit, no junk), "mh" (per-word Metropolis-Hastings accept/reject), or "smcp3"
# (legacy always-apply + weight-fold). NC_BD_BRIDGE_J adds the LM-bridge candidates the move needs to insert
# a word NOT in the observed sentence (restoration); 0 = observed surfaces only (dedup/duplicate removal).
BD_MODE = os.environ.get("NC_BD_MODE", "gibbs")
BD_ATTEMPTS = int(os.environ.get("NC_BD_ATTEMPTS", "1"))   # MH moves per bd event (more = better mixing, slower)
BD_P_STAY = float(os.environ.get("NC_BD_P_STAY", "0.0"))
BD_BRIDGE_J = int(os.environ.get("NC_BD_BRIDGE_J", "0"))
BD_POOL_CAP = int(os.environ["NC_BD_POOL_CAP"]) if os.environ.get("NC_BD_POOL_CAP") else None
BD_FUNCWORDS = os.environ.get("NC_BD_FUNCWORDS", "1") not in ("0", "false", "")  # fixed function-word insert pool


def _wellform(s):
    """Well-form the LM input: capitalize the initial letter AND ensure a terminal period (the user
    requirement -- real sentences / behavioral stimuli are capitalized and punctuated). Safe for L/E
    matching because ``_norm`` strips ``[^a-z0-9 ]``, so the period only well-forms the LM input; it
    does not change which decoded sentences count as the literal / correction reading."""
    s = s.strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]          # capitalize initial
    if s and s[-1] not in ".!?":
        s = s + "."                       # ensure terminal period
    return s


def evaluate(item_id, P, seed, alpha, rejuv, dedup):
    m = META[item_id]
    observed, intended = m["observed"], m["intended"]
    if CAP:
        observed, intended = _wellform(observed), _wellform(intended)
    trace = [] if rejuv != "off" else None    # capture the final posterior theta (rejuv_info.theta_mean)
    st, lw, logZ, sl = W.run(observed, jax.random.PRNGKey(seed), P=P, band=2,
                             action_alpha=alpha, rejuv=rejuv, dedup=dedup, trace=trace,
                             channel=CHANNEL, align_slope=ALIGN_SLOPE,
                             bd_bridge_j=BD_BRIDGE_J, bd_pool_cap=BD_POOL_CAP,
                             bd_p_stay=BD_P_STAY, bd_mode=BD_MODE, bd_attempts=BD_ATTEMPTS,
                             bd_funcwords=BD_FUNCWORDS)
    top = W.decode(st, lw, skip=sl, top=60)
    lit_n, cor_n = _norm(observed), _norm(intended)
    lit = sum(p for s, p in top if _norm(s) == lit_n)
    cor = sum(p for s, p in top if _norm(s) == cor_n)
    junk = max(0.0, 1.0 - lit - cor)
    is_edit = m["expected"] == "edit"
    metric = (cor / (cor + lit) if (cor + lit) > 0 else float("nan")) if is_edit else lit  # q_smc | kept
    theta = None
    if trace:                                  # last step that recorded a theta refresh
        for s in reversed(trace):
            if s.get("rejuv") and "theta_mean" in s["rejuv"]:
                theta = s["rejuv"]["theta_mean"]
                break
    return metric, lit, cor, junk, logZ, top[:3], theta


def main():
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    args = sys.argv[3:]
    if args == ["ALL"]:                               # whole gated battery (87 items)
        items = list(META.keys())
    else:
        items = args or DEFAULT_ITEMS
    rejuv = os.environ.get("NC_REJUV", "off")        # 'gibbs' = intended deployment (theta posterior)
    dedup = True
    verbose = os.environ.get("NC_VERBOSE", "1") not in ("", "0", "false")  # per-item top-3 (off for ALL)
    alpha = ALIGN_ALPHA_DEFAULT if CHANNEL == "align" else ACTION_ALPHA_DEFAULT
    if os.environ.get("NC_ALPHA"):                   # sweep concentrated/copy-favoured priors
        alpha = tuple(float(x) for x in os.environ["NC_ALPHA"].split(","))
    lm_penzai.load_model()
    bd = f"  bd_mode={BD_MODE} bd_p_stay={BD_P_STAY} bd_bridge_j={BD_BRIDGE_J} bd_pool_cap={BD_POOL_CAP}" \
        if rejuv == "gibbs+bd" else ""
    print(f"LM={lm_penzai.MODEL_NAME}  P={P}  seed={seed}  channel={CHANNEL}  alpha={alpha}  "
          f"align_slope={ALIGN_SLOPE}  rejuv={rejuv}  dedup={dedup}  cap_initial={CAP}{bd}  "
          f"items={len(items)}\n", flush=True)
    print(f"{'item':12s} {'exp':5s} {'metric':>6s} {'q_ref':>6s}  {'L':>4s} {'E':>4s} {'junk':>4s}  obs -> intended",
          flush=True)
    # Aggregate pass-rate tallies. EDIT pass = correction wins the L-vs-E contest (q_smc > 0.5); KEEP pass
    # = the literal reading holds (kept > 0.5). junk_hi flags items where spurious-insertion / over-edit
    # mass dominates (> 0.5) -- mostly the separate leading-opener prime artifact, tracked but not a pass/fail.
    agg = {"edit": [0, 0, 0.0], "keep": [0, 0, 0.0]}   # exp -> [n, n_pass, sum_metric(non-nan)]
    junk_hi = 0
    for iid in items:
        t = time.time()
        metric, lit, cor, junk, logZ, top3, theta = evaluate(iid, P, seed, alpha, rejuv, dedup)
        m = META[iid]
        exp = m["expected"]
        tag = "q_smc" if exp == "edit" else "kept"
        th = f"  theta({THETA_LABEL})={theta}" if theta else ""
        print(f"{iid:12s} {exp:5s} {metric:6.2f} {QREF.get(iid, float('nan')):6.2f}  "
              f"{lit:4.2f} {cor:4.2f} {junk:4.2f}  '{m['observed']}' -> '{m['intended']}'  "
              f"({tag}; {time.time()-t:.0f}s){th}", flush=True)
        if verbose:
            for s, p in top3:
                print(f"             p={p:.2f}  {s!r}", flush=True)
        a = agg[exp]
        a[0] += 1
        if metric == metric:                          # not nan
            a[2] += metric
            if metric > 0.5:
                a[1] += 1
        if junk > 0.5:
            junk_hi += 1
    print("\n=== SUMMARY ===", flush=True)
    for exp, (n, npass, smetric) in agg.items():
        if n:
            crit = "q_smc>0.5" if exp == "edit" else "kept>0.5"
            print(f"  {exp:5s} n={n:3d}  pass({crit})={npass:3d}/{n} ({100*npass/n:.0f}%)  "
                  f"mean_metric={smetric/n:.3f}", flush=True)
    print(f"  junk>0.5 (leading-opener / over-edit artifact): {junk_hi}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
