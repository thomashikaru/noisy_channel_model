"""Static probe for the targeted-rejuvenation gating signal (2026-09-02).

Question (planning/GIBBS_BD_OPTIMIZATION_DIRECTIONS.md, 09-01 addendum): if rejuvenation were
gated on RELATIVE surprisal -- a unit's contextual LM surprisal minus its unigram surprisal
(the Gen.jl conditional-rejuvenation signal, see genjax_port.unigram) -- how often would the
gate open at the known repair sites (recall), and how much of the per-item proposal grid would
survive (compression)? Answered statically, before building anything into the kernel: one
plain-LM forward per unique experiment input, no inference.

The signal is computed EXACTLY as the harness would see it: ``pwc.lm_word_surprisals(text,
prime=context or pwc.PRIME)`` (the same call the worker uses for the lookahead baseline, same
COPY spans, same prime convention) and ``unigram_surprisal(unit_str)`` (already used by the
frequency-aware insertion cost). Repair sites are recomputed from ``model_input`` vs the
repairs.csv ``intended_text`` with the same difflib call as ``converters.common.classify_edit``.
``critical_word_idx``/repairs are EVALUATION data here -- this probe measures the signal; the
signal itself never looks at them.

Usage (ncgenjax env, from the repo root):
    python -u planning/rejuv_gating_signal_probe.py compute [N_LIMIT]   # LM forwards -> signal_<ds>.csv
    python -u planning/rejuv_gating_signal_probe.py analyze             # -> ops_eval.csv + SUMMARY.md

``compute`` appends one row per unit per input as it goes (resume-safe: finished sentence_ids
are skipped on re-run) and prints flushed per-item progress -- redirect to a log, never pipe.
Outputs live in planning/rejuv_gating_probe/.
"""

import csv
import difflib
import json
import math
import os
import sys
import time

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STIM = os.path.join(REPO, "experiments", "stimuli")
OUT = os.path.join(REPO, "planning", "rejuv_gating_probe")

DATASETS = ["moses", "tabor2004", "huang2024", "gibson2013",
            "clark2026", "qian2023", "chen2023", "ryskin2021"]

EOS_UNIT = "<eos>"          # per-item end-of-sentence row (boundary M); toggled by the analysis


# ----------------------------------------------------------------------------------------------
# Phase A: the signal
# ----------------------------------------------------------------------------------------------

def _inputs(ds):
    path = os.path.join(STIM, f"{ds}.input.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute(limit=None):
    os.makedirs(OUT, exist_ok=True)
    from genjax_port import pythia_word_caprop as pwc
    from genjax_port.unigram import unigram_surprisal, CEIL_FREQ

    eos_uni = -math.log(CEIL_FREQ)
    total_done = 0
    for ds in DATASETS:
        items = _inputs(ds)
        if limit is not None:
            items = items[:limit]
        path = os.path.join(OUT, f"signal_{ds}.csv")
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                done = {int(r["sentence_id"]) for r in csv.DictReader(f)}
        fresh = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if fresh:
                w.writerow(["dataset", "sentence_id", "unit_idx", "unit",
                            "s_lm", "s_uni", "rel", "n_units", "has_context"])
            for it in items:
                sid = int(it["sentence_id"])
                if sid in done:
                    continue
                text, ctx = it["text"], (it.get("context") or "").strip()
                prime = ctx or pwc.PRIME
                t0 = time.time()
                base = pwc.lm_word_surprisals(text, prime=prime)
                units, s_lm = base["units"], base["surprisal_lm"]
                for j, (u, s) in enumerate(zip(units, s_lm)):
                    su = unigram_surprisal(u)
                    w.writerow([ds, sid, j, u, f"{float(s):.4f}", f"{su:.4f}",
                                f"{float(s) - su:.4f}", len(units), int(bool(ctx))])
                s_end = float(base["surprisal_end_lm"])
                w.writerow([ds, sid, len(units), EOS_UNIT, f"{s_end:.4f}",
                            f"{eos_uni:.4f}", f"{s_end - eos_uni:.4f}", len(units), int(bool(ctx))])
                f.flush()
                total_done += 1
                print(f"[{ds} {sid}] {len(units)}u ctx={len(ctx.split())}w "
                      f"{time.time() - t0:.2f}s (total {total_done})", flush=True)
    print("compute done", flush=True)


# ----------------------------------------------------------------------------------------------
# Phase B: repair sites vs the signal
# ----------------------------------------------------------------------------------------------

def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _word_unit_ranges(words, units):
    """Map whitespace word i -> inclusive unit range (lo, hi) by concatenating unit surfaces.
    Units are alphabetic words / punctuation runs (punct split off), whitespace words keep
    punctuation attached, so e.g. 'niece.' consumes units ['niece', '.']. Returns None on any
    mismatch (caller counts and skips)."""
    ranges, u = [], 0
    for wd in words:
        lo, acc = u, ""
        while u < len(units) and len(acc) < len(wd):
            acc += units[u]
            u += 1
        if acc != wd:
            return None
        ranges.append((lo, u - 1))
    if u != len(units):
        return None
    return ranges


def _ops(observed, intended):
    ow, iw = observed.split(), intended.split()
    ops = [op for op in difflib.SequenceMatcher(a=ow, b=iw, autojunk=False).get_opcodes()
           if op[0] != "equal"]
    return ow, iw, ops


def _op_distance(tag, i1, i2, ranges, m):
    """Signed unit-space distance function for one difflib op over observed-word indices.

    replace/delete over words i1..i2-1: units inside the words' range are distance 0.
    insert at word-gap i1: BOTH units touching the gap are distance 0 (a gate firing on either
    neighbour would open a proposal window containing the gap); further units count outward.
    Returns d(j) for j in 0..m (m = the <eos> row, adjacent to the end gap)."""
    if tag == "insert":
        b = ranges[i1][0] if i1 < len(ranges) else m   # unit just after the gap (m = eos row)
        return [(j - b) if j >= b else (j - (b - 1)) for j in range(m + 1)]
    lo, hi = ranges[i1][0], ranges[i2 - 1][1]
    return [0 if lo <= j <= hi else (j - hi if j > hi else j - lo) for j in range(m + 1)]


def analyze():
    signal = {}                       # (ds, sid) -> list of (unit, rel) incl. eos row
    for ds in DATASETS:
        path = os.path.join(OUT, f"signal_{ds}.csv")
        if not os.path.exists(path):
            print(f"WARNING: no signal for {ds}, run compute first", flush=True)
            continue
        for r in _read_csv(path):
            signal.setdefault((ds, int(r["sentence_id"])), []).append(
                (r["unit"], float(r["rel"])))

    ops_rows, skipped = [], []
    for ds in DATASETS:
        stim = {r["stim_uid"]: r for r in _read_csv(os.path.join(STIM, f"{ds}.stimuli.csv"))}
        rep_path = os.path.join(STIM, f"{ds}.repairs.csv")
        if not os.path.exists(rep_path):
            continue
        for rep in _read_csv(rep_path):
            if rep["edit_type"] in ("", "none"):
                continue
            srow = stim[rep["stim_uid"]]
            key = (ds, int(srow["sentence_id"]))
            if key not in signal:
                skipped.append((rep["stim_uid"], "no signal"))
                continue
            units = [u for u, _ in signal[key]]
            rels = [r for _, r in signal[key]]
            m = len(units) - 1                                   # last row is <eos>
            observed = srow["model_input"]
            ow, iw, ops = _ops(observed, rep["intended_text"])
            ranges = _word_unit_ranges(ow, units[:m])
            if ranges is None:
                skipped.append((rep["stim_uid"], "unit/word mismatch"))
                continue
            order = sorted(range(m + 1), key=lambda j: -rels[j])
            rank = [0] * (m + 1)                                 # rank 1 = highest rel in the item
            for pos, j in enumerate(order):
                rank[j] = pos + 1
            for k, (tag, i1, i2, j1, j2) in enumerate(ops):
                d = _op_distance(tag, i1, i2, ranges, m)
                def mx(pred):
                    v = [rels[j] for j in range(m + 1) if pred(d[j])]
                    return max(v) if v else float("-inf")
                def rk(pred):
                    v = [rank[j] for j in range(m + 1) if pred(d[j])]
                    return min(v) if v else m + 2
                item_max = max(range(m + 1), key=lambda j: rels[j])
                ops_rows.append({
                    "dataset": ds, "stim_uid": rep["stim_uid"],
                    "intended_uid": rep["intended_uid"], "contrast": srow["contrast"],
                    "op_idx": k, "n_ops": len(ops), "op_tag": tag,
                    "obs_words": " ".join(ow[i1:i2]), "int_words": " ".join(iw[j1:j2]),
                    "n_units": m,
                    "max_w0": mx(lambda x: x == 0),
                    "max_w1": mx(lambda x: abs(x) <= 1),
                    "max_w2": mx(lambda x: abs(x) <= 2),
                    "max_down3": mx(lambda x: 0 <= x <= 3),
                    "max_item": max(rels),
                    "d_of_item_max": d[item_max],
                    "rank_w0": rk(lambda x: x == 0),
                    "rank_w1": rk(lambda x: abs(x) <= 1),
                    "rank_w2": rk(lambda x: abs(x) <= 2),
                })
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ops_eval.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ops_rows[0].keys()))
        w.writeheader()
        w.writerows(ops_rows)

    _summary(signal, ops_rows, skipped)


def _dilate_frac(rels, tau, w):
    """Fraction of positions within w of a unit with rel >= tau (the surviving proposal grid)."""
    n = len(rels)
    spikes = [j for j in range(n) if rels[j] >= tau]
    if not spikes:
        return 0.0
    keep = set()
    for j in spikes:
        keep.update(range(max(0, j - w), min(n, j + w + 1)))
    return len(keep) / n


def _summary(signal, ops_rows, skipped):
    taus = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 8, 10]
    windows = [("w0", "max_w0", 0), ("w1", "max_w1", 1), ("w2", "max_w2", 2)]

    # repair -> covered iff ALL its ops covered (strict); stimulus -> ANY admissible repair covered
    by_repair, by_stim = {}, {}
    for r in ops_rows:
        rk = (r["stim_uid"], r["intended_uid"])
        by_repair.setdefault(rk, []).append(r)
        by_stim.setdefault(r["stim_uid"], set()).add(rk)

    lines = ["# Rejuv gating signal probe -- summary", "",
             f"Signal: {len(signal)} inputs; ops evaluated: {len(ops_rows)} "
             f"({len(by_repair)} repairs, {len(by_stim)} stimuli with an edit); "
             f"skipped: {len(skipped)}", ""]
    if skipped:
        lines.append("Skipped (first 10): " + "; ".join(f"{u} ({why})" for u, why in skipped[:10]))
        lines.append("")

    for wname, col, w in windows:
        lines.append(f"## Window {wname} (gate within +/-{w} units of the repair site)")
        lines.append("")
        lines.append("tau | op recall | repair recall (all ops) | stim recall (any repair) | "
                     "grid kept | items w/ any spike")
        lines.append("---|---|---|---|---|---")
        for tau in taus:
            op_rec = sum(1 for r in ops_rows if r[col] >= tau) / len(ops_rows)
            rep_rec = sum(1 for ops in by_repair.values()
                          if all(o[col] >= tau for o in ops)) / len(by_repair)
            stim_rec = sum(1 for rks in by_stim.values()
                           if any(all(o[col] >= tau for o in by_repair[rk]) for rk in rks)
                           ) / len(by_stim)
            fracs = [_dilate_frac([x for _, x in sig], tau, w) for sig in signal.values()]
            grid = sum(fracs) / len(fracs)
            hit = sum(1 for x in fracs if x > 0) / len(fracs)
            lines.append(f"{tau} | {op_rec:.3f} | {rep_rec:.3f} | {stim_rec:.3f} | "
                         f"{grid:.3f} | {hit:.3f}")
        lines.append("")

    # Top-k policy: always gate the k most anomalous positions per item (rank on rel, desc).
    ks = [1, 2, 3, 4, 6, 8]
    for wname, rcol, w in [("w0", "rank_w0", 0), ("w1", "rank_w1", 1), ("w2", "rank_w2", 2)]:
        lines.append(f"## Top-k policy, window {wname} (site within +/-{w} of a top-k unit)")
        lines.append("")
        lines.append("k | op recall | repair recall (all ops) | stim recall (any repair) | grid kept")
        lines.append("---|---|---|---|---")
        for k in ks:
            op_rec = sum(1 for r in ops_rows if r[rcol] <= k) / len(ops_rows)
            rep_rec = sum(1 for ops in by_repair.values()
                          if all(o[rcol] <= k for o in ops)) / len(by_repair)
            stim_rec = sum(1 for rks in by_stim.values()
                           if any(all(o[rcol] <= k for o in by_repair[rk]) for rk in rks)
                           ) / len(by_stim)
            fracs = []
            for sig in signal.values():
                rels_i = [x for _, x in sig]
                n = len(rels_i)
                top = sorted(range(n), key=lambda j: -rels_i[j])[:k]
                keep = set()
                for j in top:
                    keep.update(range(max(0, j - w), min(n, j + w + 1)))
                fracs.append(len(keep) / n)
            lines.append(f"{k} | {op_rec:.3f} | {rep_rec:.3f} | {stim_rec:.3f} | "
                         f"{sum(fracs) / len(fracs):.3f}")
        lines.append("")

    lines.append("## Per-dataset op recall at w1: threshold (tau = 0 / 2 / 4) and top-k (k = 2 / 4)")
    lines.append("")
    lines.append("dataset | n_ops | tau=0 | tau=2 | tau=4 | k=2 | k=4 | mean units/item")
    lines.append("---|---|---|---|---|---|---|---")
    for ds in DATASETS:
        sub = [r for r in ops_rows if r["dataset"] == ds]
        if not sub:
            continue
        recs = [sum(1 for r in sub if r["max_w1"] >= t) / len(sub) for t in (0, 2, 4)]
        krecs = [sum(1 for r in sub if r["rank_w1"] <= k) / len(sub) for k in (2, 4)]
        sizes = [len(sig) for key, sig in signal.items() if key[0] == ds]
        lines.append(f"{ds} | {len(sub)} | {recs[0]:.3f} | {recs[1]:.3f} | {recs[2]:.3f} | "
                     f"{krecs[0]:.3f} | {krecs[1]:.3f} | {sum(sizes) / len(sizes):.1f}")
    lines.append("")

    lines.append("## Op recall at w1 by op type (indel grid serves ins/del; sub-sweep serves replace)")
    lines.append("")
    lines.append("op_tag | n | tau=0 | tau=2 | k=2 | k=4")
    lines.append("---|---|---|---|---|---")
    for tag in ("insert", "delete", "replace"):
        sub = [r for r in ops_rows if r["op_tag"] == tag]
        if not sub:
            continue
        lines.append(f"{tag} | {len(sub)} | "
                     f"{sum(1 for r in sub if r['max_w1'] >= 0) / len(sub):.3f} | "
                     f"{sum(1 for r in sub if r['max_w1'] >= 2) / len(sub):.3f} | "
                     f"{sum(1 for r in sub if r['rank_w1'] <= 2) / len(sub):.3f} | "
                     f"{sum(1 for r in sub if r['rank_w1'] <= 4) / len(sub):.3f}")
    lines.append("")

    # Item-level gate: would a whole-item skip fire on the right items?
    needs_edit = set()
    for r in ops_rows:
        ds = r["dataset"]
        needs_edit.add((ds, r["stim_uid"]))
    edit_inputs = set()
    for ds in DATASETS:
        stim_path = os.path.join(STIM, f"{ds}.stimuli.csv")
        if not os.path.exists(stim_path):
            continue
        for srow in _read_csv(stim_path):
            if (ds, srow["stim_uid"]) in needs_edit:
                edit_inputs.add((ds, int(srow["sentence_id"])))
    lines.append("## Item-level gate: any unit with rel >= tau (whole-item skip candidate)")
    lines.append("")
    lines.append(f"Inputs needing an edit: {len(edit_inputs & set(signal))} / {len(signal)}")
    lines.append("")
    lines.append("tau | fires on edit-needing items | fires on clean items")
    lines.append("---|---|---")
    for tau in taus:
        e = [k for k in signal if k in edit_inputs]
        c = [k for k in signal if k not in edit_inputs]
        fe = sum(1 for k in e if max(x for _, x in signal[k]) >= tau) / len(e)
        fc = sum(1 for k in c if max(x for _, x in signal[k]) >= tau) / len(c)
        lines.append(f"{tau} | {fe:.3f} | {fc:.3f}")
    lines.append("")

    lines.append("## Where does the item's biggest spike sit relative to the repair site?")
    lines.append("")
    from collections import Counter
    c = Counter(min(max(int(r["d_of_item_max"]), -5), 5) for r in ops_rows)
    lines.append("d(item argmax, site): " + ", ".join(
        f"{d:+d}: {c.get(d, 0)}" for d in range(-5, 6)) + "  (clamped to +/-5)")
    lines.append("")

    out = "\n".join(lines)
    with open(os.path.join(OUT, "SUMMARY.md"), "w") as f:
        f.write(out + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "compute"
    if cmd == "compute":
        compute(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown command {cmd!r}")
