"""PRIOR SEARCH for the word-action channel (planning/WORD_ACTION_CHANNEL_PLAN.md sec 4, last step;
HIERARCHICAL_CALIBRATION_PLAN.md sec 6.3 step 4). The sec-4 offline preview PASSED the certification
*mechanism* at the uncalibrated prior alpha=(3,1,1,1); this script settles the prior.

We sweep the Dirichlet action prior alpha = (copy, sub, ins, del) and the substitution form-sharpness
SUB_FORM_LP (the per-edited-char `which letter` cost, the demoted pair-HMM's only free knob), score
every 2-reading battery item by the closed-form Dirichlet-multinomial hierarchical posterior (q_hier,
exactly as in calibration_word_action_preview -- no MC, no SMC, no LM work; the g_i are cached), and
pick the **WIDEST** prior (smallest concentration alpha0 = sum alpha) that hits the battery targets
with margin:

  * implausible (edit) members:  q_hier > 0.5   (margin: count how many clear 0.55)
  * plausible   (keep) members:  q_hier < 0.1
  * asymmetry desideratum:       mean(DEL families) > mean(INS families)  (Gibson; del easier than ins)

WIDTH is the regularizer now (lambda is dropped): wider alpha = more word-level certification (the M-1
clean copies count for more relative to the prior) AND more capacity for the data to move theta in the
deployed filter (HIERARCHICAL_CALIBRATION_PLAN sec 6.1). So we want the widest prior still separating
the matched pairs. We keep COPY as the prior mode (faithful to Gen.jl's [3,1,1] copy-favoured action
prior) and the three error pseudo-counts symmetric (Gen.jl's single error_alpha), so the grid is
(a_copy, a_err, SUB_FORM_LP).

Run:  PYTHONPATH=src conda run -n ncgenjax python -u -m genjax_port.calibration_word_action_prior_search
"""
import math

import numpy as np

from genjax_port.calibration_marginalize import build_items
from genjax_port.calibration_word_action_preview import (
    log_dirichlet_multinomial, reading_counts, sigmoid)

TRANSP_FORM_LP = 0.0  # a transposition is determined -> no 'which letter' choice -> free form


def form_cost_E(it, sub_form_lp):
    if it["kind"] == "sub":
        return (it["n_sub"] + it["n_indel"]) * sub_form_lp + it["n_trans"] * TRANSP_FORM_LP
    if it["kind"] == "ins":
        return -it["surp"]
    return 0.0


def predict(items, meta, alpha, sub_form_lp):
    """q_hier per item (action probs MARGINALIZED -> word-level certification)."""
    out = {}
    for it in items:
        M = len(meta[it["id"]]["observed"].split())
        nL, nE = reading_counts(it["kind"], M)
        form = form_cost_E(it, sub_form_lp)
        logit = it["g"] + (log_dirichlet_multinomial(nE, alpha)
                           - log_dirichlet_multinomial(nL, alpha)) + form
        out[it["id"]] = sigmoid(logit)
    return out


def fam_means(items, meta, pred):
    fams = {}
    for it in items:
        m = meta[it["id"]]
        key = m["family"] if m["expected"] == "edit" else m["family"] + " (keep)"
        fams.setdefault(key, []).append(pred[it["id"]])
    return {k: float(np.mean(v)) for k, v in fams.items()}


def score(items, meta, alpha, sub_form_lp):
    pred = predict(items, meta, alpha, sub_form_lp)
    ed = [pred[it["id"]] for it in items if meta[it["id"]]["expected"] == "edit"]
    kp = [pred[it["id"]] for it in items if meta[it["id"]]["expected"] == "keep"]
    fm = fam_means(items, meta, pred)
    del_fams = [v for k, v in fm.items() if k.startswith("DEL_")]
    ins_fams = [v for k, v in fm.items() if k.startswith("INS_") and "keep" not in k]
    return dict(
        n_edit=len(ed), edit_gt05=sum(q > 0.5 for q in ed), edit_gt055=sum(q > 0.55 for q in ed),
        edit_mean=float(np.mean(ed)),
        n_keep=len(kp), keep_lt01=sum(q < 0.1 for q in kp), keep_max=float(max(kp)),
        del_mean=float(np.mean(del_fams)) if del_fams else float("nan"),
        ins_mean=float(np.mean(ins_fams)) if ins_fams else float("nan"),
        mean_copy=float(alpha[0] / alpha.sum()), pred=pred)


# A sensibility FLOOR (battery-only; the matched keeps rely on g, never penalize over-editing, so the
# battery alone always pushes the optimum toward an edit-happy channel). We require COPY to remain the
# prior mode -- mean p_copy >= COPY_FLOOR -- so the settled prior is still a sensible deployment channel
# (Gen.jl deliberately favoured copy: --normal_alpha=3 vs --error_alpha=1). Among priors honouring this,
# we take the WIDEST (smallest alpha0) that hits the targets.
COPY_FLOOR = 0.50
EDIT_TARGET = 29   # 29/33 is the battery ceiling: the residual 4 are documented weak-g LM cases.


def main():
    items, meta = build_items()
    a_copy_grid = [1.5, 2.0, 3.0, 4.0, 6.0]
    a_err_grid = [0.5, 0.75, 1.0, 1.5, 2.0]
    sub_form_grid = [(math.log(1 / 15.0), "1/15"), (math.log(1 / 26.0), "1/26"),
                     (math.log(1 / 40.0), "1/40")]

    print("================ WORD-ACTION PRIOR SEARCH (closed-form q_hier; LM=410m cached) ================")
    print(f"items={len(items)}   targets: implausible q>0.5 (>={EDIT_TARGET}/33; margin col >0.55), "
          f"plausible q<0.1, DEL>INS asymmetry")
    print(f"widest = smallest alpha0 (=copy+3*err); SENSIBILITY FLOOR: mean p_copy >= {COPY_FLOOR} "
          f"(copy stays the mode, Gen.jl-faithful).\n")
    hdr = (f"{'alpha (c,s,i,d)':20s} {'a0':>4s} {'pcpy':>4s} {'subform':>7s}  {'edit>.5':>7s} {'>.55':>5s} "
           f"{'emean':>6s}  {'keep<.1':>7s} {'kmax':>5s}  {'DEL':>5s} {'INS':>5s}  {'sel':>3s}")
    print(hdr)
    rows = []
    for sf, sf_name in sub_form_grid:
        for a_copy in a_copy_grid:
            for a_err in a_err_grid:
                alpha = np.array([a_copy, a_err, a_err, a_err])
                s = score(items, meta, alpha, sf)
                a0 = float(alpha.sum())
                hits = (s["edit_gt05"] >= EDIT_TARGET and s["keep_lt01"] == s["n_keep"]
                        and s["del_mean"] > s["ins_mean"])
                sensible = s["mean_copy"] >= COPY_FLOOR
                ok = hits and sensible
                rows.append((ok, a0, alpha, sf_name, s))
                flag = "OK" if ok else ("hi" if hits else "  ")  # hits-but-edit-happy = 'hi'
                print(f"({a_copy:.1f},{a_err:.2f},{a_err:.2f},{a_err:.2f})  {a0:4.1f} {s['mean_copy']:4.2f} "
                      f"{sf_name:>7s}  {s['edit_gt05']:3d}/{s['n_edit']:<3d} {s['edit_gt055']:5d} "
                      f"{s['edit_mean']:6.2f}  {s['keep_lt01']:3d}/{s['n_keep']:<3d} {s['keep_max']:5.2f}  "
                      f"{s['del_mean']:5.2f} {s['ins_mean']:5.2f}  {flag:>3s}")
        print()

    # widest passing prior with margin: among OK rows (target + sensibility floor), smallest a0, then
    # most edits clearing 0.55. ('hi' rows hit the targets but drop copy below the mode -- edit-happy.)
    passing = [r for r in rows if r[0]]
    print("=" * 95)
    if not passing:
        print("NO prior hits the targets AND the copy-mode floor -- inspect the table / revisit the floor.")
        return
    # tie-break: widest (a0), then most margin (>0.55), then SUB_FORM_LP closest to the PRINCIPLED
    # log(1/26) ('which of 26 letters'). The battery barely distinguishes the sharpness (most sub-edits
    # are distance-1 / transpositions), so it is fixed by first principles, not fit -- the tie-break just
    # makes the script's auto-pick reflect that rather than an arbitrary grid order.
    principled = math.log(1 / 26.0)
    passing.sort(key=lambda r: (r[1], -r[4]["edit_gt055"], abs(math.log(eval(r[3])) - principled)))
    ok, a0, alpha, sf_name, s = passing[0]
    print(f"SETTLED PRIOR -- widest hitting targets with the copy-mode sensibility floor:")
    print(f"   alpha (copy,sub,ins,del) = ({alpha[0]:.0f},{alpha[1]:.1f},{alpha[2]:.1f},{alpha[3]:.1f})"
          f"   alpha0 = {a0:.1f}   SUB_FORM_LP = log({sf_name}) = {math.log(eval(sf_name)):.3f}")
    print(f"   prior-mean action probs: copy {alpha[0]/a0:.2f}  sub {alpha[1]/a0:.2f}  "
          f"ins {alpha[2]/a0:.2f}  del {alpha[3]/a0:.2f}")
    print(f"   implausible edit>0.5: {s['edit_gt05']}/{s['n_edit']}  (>.55: {s['edit_gt055']}; mean {s['edit_mean']:.2f})")
    print(f"   plausible keep<0.1:   {s['keep_lt01']}/{s['n_keep']}  (max {s['keep_max']:.2f})")
    print(f"   asymmetry: DEL mean {s['del_mean']:.2f} > INS mean {s['ins_mean']:.2f}  "
          f"({'OK' if s['del_mean']>s['ins_mean'] else 'FAIL'})")

    # the residual sub misses (the expected weak-g cases) under the chosen prior
    pred = s["pred"]
    misses = sorted([(pred[it["id"]], it["id"]) for it in items
                     if meta[it["id"]]["expected"] == "edit" and pred[it["id"]] <= 0.5])
    print(f"\n   residual edit misses (q<=0.5; expected to be weak-g LM cases):")
    for q, iid in misses:
        print(f"      {iid:14s} q={q:.2f}  '{meta[iid]['observed']}'")
    print("=" * 95)


if __name__ == "__main__":
    main()
