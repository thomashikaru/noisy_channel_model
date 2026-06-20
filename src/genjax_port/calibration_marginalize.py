"""Hierarchical channel: the channel RATES are latent with wide priors (as in the original Gen.jl
model), and each item's edit-probability is MARGINALIZED over those priors -- NO lambda tempering. This
tests the user's idea that encoding parameter uncertainty in the generative model produces graded,
uncertainty-aware inferences without hand-picking dial values (planning/CALIBRATION_PLAN.md).

Per item, the 2-reading edit-probability at a sampled channel theta is logit = g_i + channel_i(theta):
  g_i           = the LM's preference for the correction (cached slp_gain, at 410m). lambda DROPPED (=1).
  channel_i     = kappa_sub * d_i        (sub: kappa_sub = log((1-copy)/26) - log(copy))
                = log(p_del)             (del: word-deletion log-rate)
                = log(rho_ins) - surp_i  (ins: word-insertion log-rate minus the word's content cost)
The MARGINAL prediction is q_i = E_theta[ sigmoid(logit) ], a Monte-Carlo average over prior draws -- the
prior WIDTH sets the confidence, not a tempered lambda. The LM scores are independent of theta, so this is
cheap (no new LM work). The deletion>insertion asymmetry emerges from the content cost surp_i, not the rates.

Wide v0 priors (centers near a clean-ish channel; widths encode genuine uncertainty; to be CALIBRATED to
human data later). Run:  conda run -n ncgenjax python -u -m genjax_port.calibration_marginalize
"""
import csv

import numpy as np

from genjax_port.calibration_identifiability import load_items, sigmoid

GATED_410M = "planning/calibration_battery_v0_gated_410m.csv"
EXCLUDE = {"INS_DUP"}
S = 20000
SEED = 0

# --- latent channel priors (Beta on interpretable rates; wide) ---
# char copy-rate (clean-ness): mean 0.8, spans ~0.5..0.96 -> entertains a noisier channel.
COPY_A, COPY_B = 8.0, 2.0
# word deletion / insertion rates: wide, mean ~0.08; the del>ins asymmetry comes from the content cost.
PDEL_A, PDEL_B = 1.5, 18.0
PINS_A, PINS_B = 1.5, 18.0


def edit_ops(a, b):
    """Optimal-string-alignment (restricted Damerau-Levenshtein) op counts to turn a into b:
    (n_substitution, n_indel, n_transposition). Transpositions are scored as their own op."""
    a, b = a.lower(), b.lower()
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    bk = [[None] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        dp[i][0] = i
        bk[i][0] = "del"
    for j in range(1, lb + 1):
        dp[0][j] = j
        bk[0][j] = "ins"
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            match = a[i - 1] == b[j - 1]
            best, op = dp[i - 1][j - 1] + (0 if match else 1), ("match" if match else "sub")
            if dp[i - 1][j] + 1 < best:
                best, op = dp[i - 1][j] + 1, "del"
            if dp[i][j - 1] + 1 < best:
                best, op = dp[i][j - 1] + 1, "ins"
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1] and dp[i - 2][j - 2] + 1 < best:
                best, op = dp[i - 2][j - 2] + 1, "trans"
            dp[i][j], bk[i][j] = best, op
    i, j, ns, ni, nt = la, lb, 0, 0, 0
    while i > 0 or j > 0:
        op = bk[i][j]
        if op == "match":
            i, j = i - 1, j - 1
        elif op == "sub":
            ns += 1; i, j = i - 1, j - 1
        elif op == "del":
            ni += 1; i -= 1
        elif op == "ins":
            ni += 1; j -= 1
        elif op == "trans":
            nt += 1; i, j = i - 2, j - 2
        else:
            break
    return ns, ni, nt


def channel_logit(item, copy, pdel, rins):
    g = item["g"]
    if item["kind"] == "sub":
        eps = 1.0 - copy
        k_letter = np.log(eps / 26.0) - np.log(copy)   # sub / indel: pays the 1/26 "which letter" penalty
        k_trans = np.log(eps) - np.log(copy)           # transposition: determined -> ~log(26) cheaper
        return g + k_letter * (item["n_sub"] + item["n_indel"]) + k_trans * item["n_trans"]
    if item["kind"] == "del":
        return g + np.log(pdel)
    return g + np.log(rins) - item["surp"]                  # ins


def build_items(path=GATED_410M):
    """Load the gated battery into 2-reading items with substitution op-counts decomposed (so the
    transposition discount applies). Returns (items, meta). Shared by the marginal model and the
    hierarchical-posterior preview."""
    meta = {r["item_id"]: r for r in csv.DictReader(open(path))}
    items = [it for it in load_items(path) if meta[it["id"]]["family"] not in EXCLUDE]

    # Decompose each substitution into op types (transpositions are cheaper). Keep items reuse their
    # pair's edit (the would-be wrong substitution); edit items use their own changed word.
    pair_words = {r["pair_id"]: (r["edit_from"], r["edit_to"]) for r in meta.values()
                  if r["expected"] == "edit" and r["edit_type"] == "sub" and r["edit_from"]}
    for it in items:
        if it["kind"] != "sub":
            continue
        m = meta[it["id"]]
        fw, tw = (m["edit_from"], m["edit_to"]) if m["expected"] == "edit" \
            else pair_words.get(m["pair_id"], ("", ""))
        ns, ni, nt = edit_ops(fw, tw) if fw else (it["d"], 0, 0)
        it["n_sub"], it["n_indel"], it["n_trans"] = ns, ni, nt
    return items, meta


def main():
    items, meta = build_items()

    rng = np.random.default_rng(SEED)
    copy = rng.beta(COPY_A, COPY_B, S)
    pdel = rng.beta(PDEL_A, PDEL_B, S)
    rins = rng.beta(PINS_A, PINS_B, S)

    # marginal prediction per item, and the point prediction at the prior MEAN (to show hedging).
    qm, qpoint = {}, {}
    cm, cp, rm = COPY_A / (COPY_A + COPY_B), PDEL_A / (PDEL_A + PDEL_B), PINS_A / (PINS_A + PINS_B)
    for it in items:
        qm[it["id"]] = float(np.mean(sigmoid(channel_logit(it, copy, pdel, rins))))
        qpoint[it["id"]] = float(sigmoid(channel_logit(it, np.array(cm), np.array(cp), np.array(rm))))

    print("================ HIERARCHICAL (MARGINALIZED) CHANNEL -- NO lambda ================")
    print(f"LM=410m  items={len(items)}  S={S} prior draws")
    print(f"priors: copy~Beta({COPY_A},{COPY_B}) mean {cm:.2f} | p_del~Beta({PDEL_A},{PDEL_B}) mean {cp:.3f}"
          f" | rho_ins~Beta({PINS_A},{PINS_B}) mean {rm:.3f}")

    def grp(pred):
        d = {}
        for it in items:
            m = meta[it["id"]]
            key = m["family"] if m["expected"] == "edit" else m["family"] + " (keep)"
            d.setdefault(key, []).append(pred[it["id"]])
        return d

    print("\n-- marginal edit-probability by family (mean; graded = strictly between 0 and 1) --")
    g = grp(qm)
    for fam in ["SUBN", "SUBW", "DEL_TO", "DEL_FOR", "DEL_FROM", "DEL_OF", "DEL_A", "DEL_THE",
                "INS_TO", "LADDER", "SUBW (keep)", "INS_TO (keep)"]:
        if fam in g:
            v = np.array(g[fam])
            print(f"   {fam:16s} mean q = {v.mean():.2f}   [{v.min():.2f}, {v.max():.2f}]   n={len(v)}")

    print("\n-- named cases: marginal vs point-at-prior-mean (marginal should hedge toward 0.5) --")
    for key in ["SUBN-01a", "SUBW-01a", "SUBW-03a", "DELTO-01a", "DEL-to-05a",
                "INS-to-02a", "INS-to-04a", "LADDER-give-1", "LADDER-give-2",
                "SUBW-01b", "DELTO-01b", "INS-to-01b"]:
        if key in qm:
            tag = "edit" if meta[key]["expected"] == "edit" else "KEEP"
            print(f"   {key:14s} [{tag}] marginal {qm[key]:.2f}   point {qpoint[key]:.2f}   "
                  f"'{meta[key]['observed']}'")

    print("\n-- typos by operation type (transpositions now discounted ~log26 vs substitutions) --")
    for key in ["SUBN-01a", "SUBN-04a", "SUBN-06a", "SUBW-05a",
                "SUBN-02a", "SUBN-03a", "SUBW-01a", "SUBW-03a"]:
        if key in qm:
            it = next(x for x in items if x["id"] == key)
            kind = f"sub={it['n_sub']} trans={it['n_trans']} indel={it['n_indel']}"
            print(f"   {key:10s} {kind:26s} marginal q = {qm[key]:.2f}   '{meta[key]['observed']}'")

    # health checks
    edits = [qm[it["id"]] for it in items if meta[it["id"]]["expected"] == "edit"]
    keeps = [qm[it["id"]] for it in items if meta[it["id"]]["expected"] == "keep"]
    print("\n-- health --")
    print(f"   edits lean to EDIT:   {sum(q > 0.5 for q in edits)}/{len(edits)}  (mean {np.mean(edits):.2f})")
    print(f"   keeps lean to KEEP:   {sum(q < 0.5 for q in keeps)}/{len(keeps)}  (mean {np.mean(keeps):.2f})")
    print(f"   over-editing guard:   max keep q = {max(keeps):.2f}  (want < ~0.5)")
    graded = sum(0.05 < q < 0.95 for q in edits + keeps)
    print(f"   graded (not 0/1):     {graded}/{len(edits)+len(keeps)} predictions strictly in (0.05,0.95)")
    print("=================================================================================")


if __name__ == "__main__":
    main()
