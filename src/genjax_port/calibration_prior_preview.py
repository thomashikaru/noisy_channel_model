"""q_full PREVIEW: the cheap, faithful stand-in for Option 2 (hierarchical channel) -- run OFFLINE on the
same cached LM gains, no SMC and no new LM forwards (planning/HIERARCHICAL_CALIBRATION_PLAN.md sec 6.3).

The point: settling the priors on the *offline* model can mislead, because the deployed hierarchical model
averages the edit-probability under the theta-POSTERIOR, not the prior. For each 2-reading item (literal L =
observed, vs the designed edit E), per channel draw theta_s:

    gch_s = g + ch(theta_s)                         (the per-theta logit; g = cached slp_gain at 410m)
    sigma_s = sigmoid(gch_s)                         (the per-theta conditional edit-probability)
    log W_L,s = n_char*log copy + n_word*log(1-pdel) + n_word*log(1-rins)   (literal reading's BULK channel)
    log V_s = log W_L,s + softplus(gch_s)            (V = u(L)+u(E) = total evidence at theta_s)

  q_off  = mean_s[ sigma_s ]                         (average the conditional under the PRIOR)
  q_full = sum_s[ softmax(log V)_s * sigma_s ]       (average under the theta-POSTERIOR: data reweights theta)

The bulk weight W_L is dominated by copy^n_char: a long clean sentence is n_char votes for a clean channel,
so the theta-posterior concentrates on high copy -> the one edit is charged at the CERTIFIED-clean noise
level, not the prior mean. q_full < q_off exactly when the clean context over-certifies (the antidote case).
We also report the implied posterior copy (the certified noise level) so the mechanism is visible.

Run:  PYTHONPATH=src conda run -n ncgenjax python -u -m genjax_port.calibration_prior_preview
"""
import numpy as np

from genjax_port.calibration_marginalize import (
    COPY_A, COPY_B, PDEL_A, PDEL_B, PINS_A, PINS_B, S, SEED,
    build_items, channel_logit,
)


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def predict(items, meta, copy, pdel, rins):
    """Return per-item (q_off, q_full_global, q_full_local, post_copy_global). copy/pdel/rins are (S,) draws.

    q_full_global: ONE global copy latent -> the edit is certified by the whole sentence (copy^N_sentence).
    q_full_local:  PER-WORD copy -> the edit is certified only by the SUSPECT WORD's chars (copy^len_word),
                   because every other word's copy latent is shared by both readings and cancels. This is the
                   targeted structural fix for the substitution over-certification (sec 6.2 option b)."""
    logcopy, log1mp, log1mr = np.log(copy), np.log(1 - pdel), np.log(1 - rins)

    def q_full(n_char, n_word, gch):
        # V = u(L)+u(E); the literal reading's bulk = copy^n_char * (1-pdel)^n_word * (1-rins)^n_word.
        # (pdel/rins stay GLOBAL in the local model; only copy becomes per-word, varying n_char.)
        logV = n_char * logcopy + n_word * log1mp + n_word * log1mr + softplus(gch)
        w = np.exp(logV - logV.max())
        w /= w.sum()
        return float((w * sigmoid(gch)).sum()), float((w * copy).sum())

    out = {}
    for it in items:
        m = meta[it["id"]]
        obs = m["observed"]
        n_sentence = sum(1 for c in obs if not c.isspace())          # whole-sentence copy events
        n_word = len(obs.split())
        suspect = m["edit_from"] if it["kind"] == "sub" and m["edit_from"] else ""
        n_local = len(suspect) if suspect else n_sentence            # per-word: only the suspect word certifies
        gch = channel_logit(it, copy, pdel, rins)                    # g + ch(theta), shape (S,)
        qfg, pc = q_full(n_sentence, n_word, gch)
        qfl, _ = q_full(n_local, n_word, gch)
        out[it["id"]] = (float(sigmoid(gch).mean()), qfg, qfl, pc)
    return out


def main():
    items, meta = build_items()
    rng = np.random.default_rng(SEED)
    copy = rng.beta(COPY_A, COPY_B, S)
    pdel = rng.beta(PDEL_A, PDEL_B, S)
    rins = rng.beta(PINS_A, PINS_B, S)
    cm = COPY_A / (COPY_A + COPY_B)
    pred = predict(items, meta, copy, pdel, rins)

    print("======= q_full PREVIEW: offline prior-average  vs  hierarchical (GLOBAL copy)  vs  (PER-WORD copy) =======")
    print(f"LM=410m  items={len(items)}  S={S} draws   prior copy~Beta({COPY_A},{COPY_B}) mean {cm:.2f}")
    print("q_off    = average under the PRIOR (current offline model)")
    print("q_glob   = hierarchical posterior, ONE global copy latent  (edit certified by the whole sentence)")
    print("q_local  = hierarchical posterior, PER-WORD copy latent     (edit certified by the suspect word only)")
    print("post_copy = implied posterior copy under the GLOBAL model (the certified noise level)\n")

    def fam_key(it):
        m = meta[it["id"]]
        return m["family"] if m["expected"] == "edit" else m["family"] + " (keep)"

    fams = {}
    for it in items:
        fams.setdefault(fam_key(it), []).append(it["id"])
    print(f"-- by family --   {'q_off':>7s} {'q_glob':>7s} {'q_local':>8s} {'post_copy':>10s}")
    order = ["SUBN", "SUBW", "DEL_TO", "DEL_FOR", "DEL_FROM", "DEL_OF", "DEL_A", "DEL_THE",
             "INS_TO", "LADDER", "SUBW (keep)", "INS_TO (keep)"]
    for fam in order:
        if fam not in fams:
            continue
        qo = np.array([pred[i][0] for i in fams[fam]])
        qg = np.array([pred[i][1] for i in fams[fam]])
        ql = np.array([pred[i][2] for i in fams[fam]])
        pc = np.array([pred[i][3] for i in fams[fam]])
        print(f"   {fam:16s} {qo.mean():7.2f} {qg.mean():7.2f} {ql.mean():8.2f} {pc.mean():10.2f}")

    print("\n-- named cases (real-word-substitution casualties + anchors) --")
    print(f"   {'item':14s} {'q_off':>6s} {'q_glob':>7s} {'q_local':>8s}  observed")
    for key in ["SUBW-01a", "SUBW-03a", "SUBN-01a", "SUBN-02a", "SUBN-03a",
                "DEL-to-05a", "DELTO-01a", "INS-to-04a", "LADDER-give-1",
                "SUBW-01b", "INS-to-01b"]:
        if key in pred:
            qo, qg, ql, pc = pred[key]
            tag = "EDIT" if meta[key]["expected"] == "edit" else "keep"
            n_char = sum(1 for c in meta[key]["observed"] if not c.isspace())
            print(f"   {key:14s} {qo:6.2f} {qg:7.2f} {ql:8.2f}  [{tag} N={n_char:2d}] '{meta[key]['observed']}'")

    # health: do the targets hold under each model?
    ed = [(it["id"], pred[it["id"]]) for it in items if meta[it["id"]]["expected"] == "edit"]
    kp = [(it["id"], pred[it["id"]]) for it in items if meta[it["id"]]["expected"] == "keep"]
    print("\n-- targets (implausible edit>0.5 ; plausible keep i.e. edit-prob<0.1) --")
    for label, q_idx in [("q_off ", 0), ("q_glob", 1), ("q_local", 2)]:
        e_ok = sum(p[q_idx] > 0.5 for _, p in ed)
        k_ok = sum(p[q_idx] < 0.1 for _, p in kp)
        print(f"   {label}: implausible>0.5: {e_ok}/{len(ed)}   plausible<0.1: {k_ok}/{len(kp)}"
              f"   (edit mean {np.mean([p[q_idx] for _,p in ed]):.2f})")
    print("=========================================================================================================")


if __name__ == "__main__":
    main()
