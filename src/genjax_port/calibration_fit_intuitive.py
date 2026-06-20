"""Calibrate the channel dials to our INTUITIVE calibration set (not human data): find dial settings
that make the model prefer the corrections we believe it should make (antidote->anecdote, restore the
missing 'to', delete the spurious 'to', ...) while leaving controls alone. The reserved human data stays
untouched -- it is the later TEST of whether intuition-fit dials predict actual human inferences.

Decisions baked in (2026-06-18):
  * SOFT, MODERATE targets (~0.7) even for confident corrections -- uncertainty-aware, never a 0/1 gold
    label. This intentionally tempers lambda rather than letting the model assert edits with certainty.
  * INS_DUP (doubled-word) items EXCLUDED -- frequency-aware insertion structurally can't cheaply remove
    an exact rare-content duplicate (charges its full unigram surprisal), a known model gap to revisit.
  * INS_TO/INS_FOR (Gibson spurious-FUNCTION-word insertions, e.g. delete a spurious 'to') ARE INCLUDED:
    a function word is cheap to remove, so no structural problem, and they are the proper mirror of the
    missing-'to' deletions. So rho_ins IS calibrated. Free dials: {lambda, WDEL, log rho_ins, kappa_sub}.

2-reading reduction reused from `calibration_identifiability` (logit q_i = lambda*g_i + channel_i).
Run:  conda run -n ncgenjax python -u -m genjax_port.calibration_fit_intuitive
"""
import csv

import numpy as np

from genjax_port.calibration_identifiability import (
    GATED, PNAMES, PRIOR_PREC, design, load_items, nll, sigmoid,
)

N_PSEUDO = 40
EXCLUDE_FAMILIES = {"INS_DUP"}     # the structurally-stuck doublings (known model gap)

BOUNDS = np.array([
    [0.05, 1.00],                       # lambda
    [-12.0, -0.20],                     # WDEL
    [np.log(1e-3), np.log(0.30)],       # log rho_ins (rate 0.1%..30%)
    [-8.0, -0.20],                      # kappa_sub
])
START = np.array([0.7, -3.0, np.log(0.02), -3.0])

# Soft, moderate intuitive targets (edit-probabilities) -- never 0/1.
TGT_KEEP = 0.12
TGT_EDIT_CLEAR = 0.75      # typos, jarring omissions
TGT_EDIT_SUBTLE = 0.68     # malapropisms, missing/spurious function word
TGT_LADDER = {"implaus_high": 0.72, "implaus_mid": 0.52}


def target_for(meta):
    if meta["expected"] == "keep":
        return TGT_KEEP
    fam, plaus = meta["family"], meta["plausibility"]
    if fam == "LADDER":
        return TGT_LADDER.get(plaus, 0.5)
    if fam == "SUBN":
        return TGT_EDIT_CLEAR
    if fam.startswith("DEL_") and fam not in ("DEL_TO", "DEL_FOR", "DEL_FROM"):
        return TGT_EDIT_CLEAR                          # the added high-preference omissions
    return TGT_EDIT_SUBTLE                              # SUBW + datives (DEL/INS_TO/INS_FOR)


def bounded_fit(X, off, k, N, bounds, start, iters=300):
    lo, hi = bounds[:, 0], bounds[:, 1]
    theta = start.copy()
    for _ in range(iters):
        q = sigmoid(X @ theta + off)
        grad = X.T @ (k - N * q) - PRIOR_PREC * theta
        W = N * q * (1 - q)
        H = X.T @ (X * W[:, None]) + PRIOR_PREC * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        f0 = nll(theta, X, off, k, N)
        a = 1.0
        while a > 1e-7:
            cand = np.clip(theta + a * step, lo, hi)
            if nll(cand, X, off, k, N) <= f0 + 1e-9:
                break
            a *= 0.5
        new = np.clip(theta + a * step, lo, hi)
        if np.max(np.abs(new - theta)) < 1e-9:
            theta = new
            break
        theta = new
    cov = np.linalg.inv(X.T @ (X * (N * sigmoid(X @ theta + off)
                                    * (1 - sigmoid(X @ theta + off)))[:, None]) + PRIOR_PREC * np.eye(X.shape[1]))
    return theta, cov


def main():
    meta = {r["item_id"]: r for r in csv.DictReader(open(GATED))}
    items = [it for it in load_items(GATED) if meta[it["id"]]["family"] not in EXCLUDE_FAMILIES]
    X, off = design(items)
    t = np.array([target_for(meta[it["id"]]) for it in items])
    k = N_PSEUDO * t

    theta, cov = bounded_fit(X, off, k, float(N_PSEUDO), BOUNDS, START)
    sd = np.sqrt(np.diag(cov))
    q = sigmoid(X @ theta + off)
    at_bound = [PNAMES[j] for j in range(4)
                if abs(theta[j] - BOUNDS[j, 0]) < 1e-3 or abs(theta[j] - BOUNDS[j, 1]) < 1e-3]

    n_ins = sum(1 for it in items if it["kind"] == "ins")
    print("================ FIT TO INTUITIVE CALIBRATION SET (v3) ================")
    print(f"items fit: {len(items)}  (INS_DUP excluded; {n_ins} function-word insertion items in)")
    print("\n-- dial settings that best reproduce our intuitions --")
    for j, n in enumerate(PNAMES):
        extra = f"   (rho_ins = {np.exp(theta[j]):.3f})" if n == "log_rho_ins" else ""
        print(f"   {n:12s} = {theta[j]:+7.3f}  +/- {sd[j]:.2f}{extra}")
    print("   [compare current operating point: lambda=1.0, WDEL=-9.0, rho_ins=0.02]")
    if at_bound:
        print(f"   AT A PHYSICAL BOUND (intuition wants further): {at_bound}")

    hit = (q > 0.5) == (t > 0.5)
    print(f"\n-- reproduction: {hit.sum()}/{len(items)} items on the intended side of 0.5 --")
    by_fam = {}
    for it, h in zip(items, hit):
        by_fam.setdefault(meta[it["id"]]["family"], []).append(h)
    for fam in ["SUBW", "SUBN", "DEL_TO", "DEL_FOR", "DEL_FROM", "DEL_OF", "DEL_A", "DEL_THE",
                "INS_TO", "INS_FOR", "LADDER"]:
        if fam in by_fam:
            print(f"   {fam:9s} {sum(by_fam[fam])}/{len(by_fam[fam])}")

    print("\n-- named cases (incl. the new spurious-'to' insertions) --")
    for key in ["SUBW-01a", "DELTO-01a", "INS-to-01a", "INS-to-02a", "DEL-to-05a", "LADDER-give-1"]:
        for it, ti, qi in zip(items, t, q):
            if it["id"] == key:
                print(f"   {key:14s} target {ti:.2f}  model {qi:.2f}   '{meta[key]['observed']}'")

    misses = [(it["id"], meta[it["id"]]["expected"], ti, qi)
              for it, ti, qi, h in zip(items, t, q, hit) if not h]
    print(f"\n-- items NOT reproduced ({len(misses)}) --")
    for mid, exp, ti, qi in misses:
        print(f"   {mid:14s} [{exp:4s}] target {ti:.2f}  model {qi:.2f}   '{meta[mid]['observed']}'")
    if not misses:
        print("   (none)")
    print("======================================================================")


if __name__ == "__main__":
    main()
