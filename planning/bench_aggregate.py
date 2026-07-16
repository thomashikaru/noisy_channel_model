#!/usr/bin/env python
"""Aggregate the posterior-stability benchmark into per-config / per-sentence tables.

Walks planning/bench_results/<stem>/<config>/results/, reads the evidence-merged item record
(item_NNNNN.json) and the per-seed records (item_NNNNN_sJ.json), and reports, per (config, sentence):
  - merged MAP + merged P(MAP)
  - across seeds: the modal MAP, MAP-agreement fraction (#seeds at modal MAP / #seeds), #distinct MAPs
  - logZ: per-seed list, mean, std, spread (max-min)   [the cross-seed stability readout]
  - mean per-seed runtime

Stdlib only (no jax). Prints human tables and writes planning/bench_results/aggregate.json.
"""
import glob
import json
import os
import statistics
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "bench_results")

# Short labels for the 3 benchmark sentences (by observed text). Each probes one channel operation.
SHORT = {
    "I want go home.": "want-go (INSERT 'to')",
    "The the patient recovered quickly.": "the-the (DELETE dup 'the')",
    "The boy licked the ball into the net.": "boy-licked (SUB licked->kicked)",
}
# The "target" correct inference for each (for a quick correctness flag; informational only).
# NB boy-licked is a SUBSTITUTION example: the intended noisy-channel inference is licked -> kicked.
TARGET = {
    "I want go home.": "I want to go home.",
    "The the patient recovered quickly.": "The patient recovered quickly.",
    "The boy licked the ball into the net.": "The boy kicked the ball into the net.",
}


def collect():
    rows = []
    for stem in sorted(glob.glob(os.path.join(ROOT, "*"))):
        if not os.path.isdir(stem) or os.path.basename(stem) == "logs":
            continue
        for cfgdir in sorted(glob.glob(os.path.join(stem, "*"))):
            resdir = os.path.join(cfgdir, "results")
            for merged_path in sorted(glob.glob(os.path.join(resdir, "item_*.json"))):
                b = os.path.basename(merged_path)
                if "_s" in b:           # skip per-seed files; we read them via the merged idx
                    continue
                with open(merged_path) as fh:
                    m = json.load(fh)
                if m.get("status") != "ok":
                    continue
                cfg = m["config"]       # resolved effective config is stored on every record
                idx = m["idx"]
                obs = m["observed"]
                seed_paths = sorted(glob.glob(os.path.join(resdir, f"item_{idx:05d}_s*.json")))
                seed_maps, seed_logZs, seed_rt = [], [], []
                for sp in seed_paths:
                    with open(sp) as fh:
                        s = json.load(fh)
                    if s.get("status") == "ok":
                        seed_maps.append(s["map"])
                        seed_logZs.append(s["logZ"])
                        seed_rt.append(s.get("runtime_s", float("nan")))
                if not seed_maps:        # n_seeds=1 case: use the merged record itself
                    seed_maps = [m["map"]]
                    seed_logZs = [m["logZ"]]
                    seed_rt = [m.get("runtime_s", float("nan"))]
                cnt = Counter(seed_maps)
                modal_map, modal_n = cnt.most_common(1)[0]
                rows.append({
                    "lm": cfg["lm"].split("/")[-1], "P": cfg["particles"], "rejuv": cfg["rejuv"],
                    "lookback": cfg["rejuv_lookback"], "channel": cfg["channel"],
                    "obs": obs, "short": SHORT.get(obs, obs[:24]),
                    "merged_map": m["map"], "merged_p": round(m["hypotheses"][0]["prob"], 3) if m["hypotheses"] else None,
                    "n_seeds": len(seed_maps),
                    "modal_map": modal_map, "agree": round(modal_n / len(seed_maps), 3),
                    "n_distinct_maps": len(cnt), "distinct_maps": dict(cnt),
                    "correct_modal": modal_map == TARGET.get(obs),
                    "logZ_mean": round(statistics.fmean(seed_logZs), 3),
                    "logZ_std": round(statistics.pstdev(seed_logZs) if len(seed_logZs) > 1 else 0.0, 3),
                    "logZ_spread": round(max(seed_logZs) - min(seed_logZs), 3),
                    "logZ_per_seed": [round(z, 2) for z in seed_logZs],
                    "rt_mean": round(statistics.fmean(seed_rt), 1) if seed_rt else None,
                })
    return rows


def fmt_tables(rows):
    out = []
    order = {"want-go (missing 'to')": 0, "the-the (dup 'the')": 1, "boy-licked (clean)": 2}
    rows = sorted(rows, key=lambda r: (r["lm"], r["rejuv"], r["lookback"], r["P"], order.get(r["short"], 9)))
    hdr = f"{'LM':10} {'P':>4} {'rejuv':9} {'lb':>3} | {'sentence':24} {'agree':>5} {'#m':>3} " \
          f"{'logZ_mean':>9} {'logZ_std':>8} {'spread':>7} {'rt_s':>6}  MAP"
    out.append(hdr)
    out.append("-" * len(hdr))
    last_key = None
    for r in rows:
        key = (r["lm"], r["rejuv"], r["lookback"], r["P"])
        if last_key is not None and key != last_key:
            out.append("")
        last_key = key
        flag = "" if r["correct_modal"] else "  <BAD-MODE>" if r["agree"] >= 0.5 else "  <UNSTABLE>"
        out.append(f"{r['lm']:10} {r['P']:>4} {r['rejuv']:9} {r['lookback']:>3} | "
                   f"{r['short']:24} {r['agree']:>5.2f} {r['n_distinct_maps']:>3} "
                   f"{r['logZ_mean']:>9.2f} {r['logZ_std']:>8.3f} {r['logZ_spread']:>7.2f} "
                   f"{str(r['rt_mean']):>6}  {r['merged_map']!r}{flag}")
    return "\n".join(out)


def fmt_pivot(rows):
    """Per-sentence P-trend pivot: rows=rejuv, cols=P, cell = agree (logZ_std). pythia-70m, lb=6 only
    (the core grid) so the particle/rejuv trend is read at a glance."""
    out = []
    Ps = sorted({r["P"] for r in rows if r["lm"] == "pythia-70m" and r["lookback"] == 6})
    rejuvs = ["off", "gibbs", "gibbs+bd"]
    for short in ["want-go (INSERT 'to')", "the-the (DELETE dup 'the')", "boy-licked (SUB licked->kicked)"]:
        out.append(f"\n### {short}   [cell = agree-rate (logZ_std); core grid pythia-70m lb6]")
        out.append(f"{'rejuv':9} | " + " ".join(f"P={p:<13}" for p in Ps))
        for rej in rejuvs:
            cells = []
            for p in Ps:
                m = [r for r in rows if r["lm"] == "pythia-70m" and r["lookback"] == 6
                     and r["P"] == p and r["rejuv"] == rej and r["short"] == short]
                cells.append(f"{m[0]['agree']:.2f} ({m[0]['logZ_std']:>4.1f})  " if m else f"{'--':>15}")
            out.append(f"{rej:9} | " + " ".join(f"{c:<15}" for c in cells))
    return "\n".join(out)


def fmt_cliffs(rows):
    """List the least-stable (config, sentence) cells: low MAP-agreement or high logZ spread."""
    bad = [r for r in rows if r["agree"] < 0.6 or r["logZ_spread"] > 8]
    bad = sorted(bad, key=lambda r: (r["agree"], -r["logZ_spread"]))
    out = ["\n### Least-stable cells (agree<0.6 or logZ_spread>8):"]
    for r in bad:
        out.append(f"  {r['lm']:10} P{r['P']:<3} {r['rejuv']:8} lb{r['lookback']:<2} {r['short']:24} "
                   f"agree={r['agree']:.2f} #maps={r['n_distinct_maps']} spread={r['logZ_spread']:.1f} "
                   f"MAP={r['merged_map']!r}")
    return "\n".join(out)


if __name__ == "__main__":
    rows = collect()
    print(fmt_tables(rows))
    print(fmt_pivot(rows))
    print(fmt_cliffs(rows))
    with open(os.path.join(ROOT, "aggregate.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n[{len(rows)} (config,sentence) rows] -> {os.path.join(ROOT, 'aggregate.json')}")
