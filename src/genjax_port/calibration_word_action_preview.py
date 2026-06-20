"""WORD-ACTION offline preview: the sanity check for the redesigned channel (planning/
WORD_ACTION_CHANNEL_PLAN.md sec 4). Confirms the over-certification is GONE once the latent noise rate is the
word-level Dirichlet ACTION distribution (p_copy, p_sub, p_insert, p_delete) instead of a character copy rate.

For each 2-reading item (literal L = observed, vs the designed correction E), the channel score factors into
a word-level action cost + a conditional form cost:

  log u(reading) = log P_LM(intended)  +  log[ Dirichlet-multinomial of the reading's action counts ]
                                        +  form cost  (sub: edit distance; insert: -surprisal; copy/del: 0)

Action counts n = (n_copy, n_sub, n_ins, n_del) over the reading's intended words:
  L (every reading):      M copies                         -> n^L = (M, 0, 0, 0)
  E, substitution item:   M-1 copies + 1 sub               -> n^E = (M-1, 1, 0, 0)
  E, deletion item:       M copies + 1 delete (extra word) -> n^E = (M,   0, 0, 1)
  E, insertion item:      M-1 copies + 1 insert            -> n^E = (M-1, 0, 1, 0)
where M = observed word count and g = logP_LM(E) - logP_LM(L) is the cached 410m gain.

Because the action prior is Dirichlet, the marginal over the action probabilities is the closed-form
Dirichlet-multinomial -- NO Monte Carlo. We report:
  q_point = correction prob at the prior-MEAN action probs (the action shared copies cancel -> NO certification)
  q_hier  = correction prob with the action probs MARGINALIZED (Dirichlet-multinomial -> word-level certification)
The certification now lives in (alpha_copy + M - 1), i.e. it scales with the WORD count (~7), not the character
count (~38). So q_hier ~ q_point (mild gap), where the char-copy model collapsed (antidote 0.67 -> 0.19).

Run:  PYTHONPATH=src conda run -n ncgenjax python -u -m genjax_port.calibration_word_action_preview
"""
import math
import os

import numpy as np

from genjax_port.calibration_marginalize import build_items

# Dirichlet action prior (copy, sub, insert, delete). Original Gen.jl was [3,1,1] over (copy,sub,insert)
# (config.ACTION_ALPHAS); extended here with delete. UNCALIBRATED -- the magnitudes get settled vs the
# targets in the next step (task #3); this preview is about the certification *mechanism*, read off q_point->q_hier.
# Override for sweeping/calibration:  NC_ALPHA="10,1,1,1" python -m genjax_port.calibration_word_action_preview
ALPHA = np.array([float(x) for x in os.environ.get("NC_ALPHA", "3,1,1,1").split(",")])  # (copy,sub,insert,delete)
SUB_FORM_LP = math.log(1.0 / 26.0)              # form: 'which of 26 letters' per substituted/indel char
TRANSP_FORM_LP = 0.0                            # a transposition is determined -> no letter choice -> free form
IDX = dict(copy=0, sub=1, ins=2, dele=3)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def log_dirichlet_multinomial(counts, alpha):
    """log P(counts) = log[ B(alpha+counts)/B(alpha) ] -- the Dirichlet-multinomial marginal over theta."""
    a0, n = alpha.sum(), sum(counts)
    return (math.lgamma(a0) - math.lgamma(a0 + n)
            + sum(math.lgamma(alpha[k] + counts[k]) - math.lgamma(alpha[k]) for k in range(len(alpha))))


def reading_counts(kind, M):
    """(n^L, n^E) action-count vectors over (copy, sub, ins, del)."""
    nL = [M, 0, 0, 0]
    if kind == "sub":
        nE = [M - 1, 1, 0, 0]
    elif kind == "del":
        nE = [M, 0, 0, 1]              # E has the extra (restored) intended word, deleted by the channel
    else:                              # ins: the spurious observed word is an insertion under E
        nE = [M - 1, 0, 1, 0]
    return nL, nE


def form_cost_E(it):
    """log P(form | action) for the edited reading. Literal reading is all copies -> form 0."""
    if it["kind"] == "sub":
        return ((it["n_sub"] + it["n_indel"]) * SUB_FORM_LP + it["n_trans"] * TRANSP_FORM_LP)
    if it["kind"] == "ins":
        return -it["surp"]             # spurious-word content cost (unigram surprisal)
    return 0.0                         # delete: deterministic (word -> nothing)


def predict(items, meta):
    mean = ALPHA / ALPHA.sum()
    out = {}
    for it in items:
        M = len(meta[it["id"]]["observed"].split())
        nL, nE = reading_counts(it["kind"], M)
        dn = np.array(nE) - np.array(nL)
        form = form_cost_E(it)
        # q_point: action probs at the prior mean (shared copies cancel -> dn . log(mean)); NO certification.
        logit_point = it["g"] + float(dn @ np.log(mean)) + form
        # q_hier: action probs MARGINALIZED (Dirichlet-multinomial ratio); word-level certification.
        logit_hier = it["g"] + (log_dirichlet_multinomial(nE, ALPHA)
                                 - log_dirichlet_multinomial(nL, ALPHA)) + form
        out[it["id"]] = (sigmoid(logit_point), sigmoid(logit_hier), M)
    return out


def main():
    items, meta = build_items()
    pred = predict(items, meta)
    a = ALPHA
    print("=========== WORD-ACTION offline preview: point (no certification) vs hierarchical (marginalized) ===========")
    print(f"LM=410m  items={len(items)}   Dirichlet alpha (copy,sub,ins,del) = {list(a)}  (UNCALIBRATED)")
    print(f"prior-mean action probs: copy {a[0]/a.sum():.2f}  sub {a[1]/a.sum():.2f}  ins {a[2]/a.sum():.2f}  del {a[3]/a.sum():.2f}")
    print("q_point = action probs at prior mean (NO certification);  q_hier = action probs marginalized (word-level certification)")
    print("contrast: the CHARACTER-copy hierarchical model collapsed SUBW 0.84->0.68 (antidote 0.67->0.19).\n")

    def fam_key(it):
        m = meta[it["id"]]
        return m["family"] if m["expected"] == "edit" else m["family"] + " (keep)"

    fams = {}
    for it in items:
        fams.setdefault(fam_key(it), []).append(it["id"])
    print(f"-- by family --   {'q_point':>8s} {'q_hier':>7s} {'gap':>7s}   n")
    order = ["SUBN", "SUBW", "DEL_TO", "DEL_FOR", "DEL_FROM", "DEL_OF", "DEL_A", "DEL_THE",
             "INS_TO", "LADDER", "SUBW (keep)", "INS_TO (keep)"]
    for fam in order:
        if fam not in fams:
            continue
        qp = np.array([pred[i][0] for i in fams[fam]])
        qh = np.array([pred[i][1] for i in fams[fam]])
        print(f"   {fam:16s} {qp.mean():8.2f} {qh.mean():7.2f} {qh.mean()-qp.mean():+7.2f}   {len(qp)}")

    print("\n-- named cases (the cases the char-copy model broke) --")
    print(f"   {'item':14s} {'q_point':>7s} {'q_hier':>7s} {'M':>3s}  observed")
    for key in ["SUBW-01a", "SUBW-03a", "SUBN-01a", "SUBN-02a", "SUBN-03a",
                "DEL-to-05a", "DELTO-01a", "INS-to-04a", "LADDER-give-1",
                "SUBW-01b", "INS-to-01b"]:
        if key in pred:
            qp, qh, M = pred[key]
            tag = "EDIT" if meta[key]["expected"] == "edit" else "keep"
            print(f"   {key:14s} {qp:7.2f} {qh:7.2f} {M:3d}  [{tag}] '{meta[key]['observed']}'")

    ed = [pred[it["id"]] for it in items if meta[it["id"]]["expected"] == "edit"]
    kp = [pred[it["id"]] for it in items if meta[it["id"]]["expected"] == "keep"]
    print("\n-- targets (implausible edit>0.5 ; plausible keep i.e. edit<0.1) [alpha UNCALIBRATED] --")
    for label, idx in [("q_point", 0), ("q_hier ", 1)]:
        e_ok = sum(p[idx] > 0.5 for p in ed)
        k_ok = sum(p[idx] < 0.1 for p in kp)
        print(f"   {label}: implausible>0.5: {e_ok}/{len(ed)}   plausible<0.1: {k_ok}/{len(kp)}"
              f"   (edit mean {np.mean([p[idx] for p in ed]):.2f}, keep max {max(p[idx] for p in kp):.2f})")

    # the headline number: how big is the certification gap, vs the char-copy collapse?
    sub_items = [pred[it["id"]] for it in items if it["kind"] == "sub" and meta[it["id"]]["expected"] == "edit"]
    gap = np.mean([p[0] - p[1] for p in sub_items])
    print(f"\n   SUBSTITUTION certification gap (q_point - q_hier), mean over edits: {gap:+.3f}"
          f"   <- char-copy model's gap was ~ -0.16 (and -0.48 on antidote)")
    print("===========================================================================================================")


if __name__ == "__main__":
    main()
