#!/usr/bin/env python
"""Collect the worker's per-item JSON into tidy analysis tables (harness plan §4, Phase 3).

Usage:
    python experiments/collect.py [--results-root results_nc] [--out experiments/outputs]
                                  [--stimuli experiments/stimuli] [dataset ...]

For every dataset requested (default: every ``<ds>.input`` stem present under the results root),
and every config slug found under it, this joins ``results_nc/<ds>.input/<slug>/results/
item_*.json`` back to ``stimuli/<ds>.stimuli.csv`` on ``(dataset, sentence_id)`` and writes:

    <out>/<slug>/<ds>/sentences.csv.gz   one row per STIMULUS row (stimuli sharing a sentence_id
                                         each get the shared run's results)
    <out>/<slug>/<ds>/posterior.csv.gz   one row per (sentence_id, rank) hypothesis
    <out>/<slug>/<ds>/words.csv.gz       long format: one row per (sentence_id, unit) x
                                         (seed | "merged")
    <out>/status.md                      per (dataset, config): ok / error / missing counts

Stdlib + pandas only -- no jax, safe to run anywhere. Edit ops of each hypothesis vs the observed
sentence are recomputed with ``converters.common.classify_edit``, the same classifier the stimulus
build uses. Raw per-item JSON stays in ``results_nc/``; these tables are derived and untracked.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from converters.common import classify_edit  # noqa: E402  (the build's own edit classifier)


# --------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------

def _load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def load_results(cfg_dir):
    """{sentence_id: merged record} + {(sentence_id, seed_index): per-seed record}."""
    merged, seeds = {}, {}
    for path in sorted(glob.glob(os.path.join(cfg_dir, "results", "item_*.json"))):
        name = os.path.basename(path)
        if name.endswith(".viz.json"):
            continue
        m = re.fullmatch(r"item_(\d+)(?:_s(\d+))?\.json", name)
        if not m:
            continue
        rec = _load_json(path)
        if rec is None:
            continue
        idx = int(m.group(1))
        if m.group(2) is None:
            merged[idx] = rec
        else:
            seeds[(idx, int(m.group(2)))] = rec
    return merged, seeds


def _edit_cols(observed, hypothesis, prefix):
    if not hypothesis:
        return {f"{prefix}_edit_type": None, f"{prefix}_edit_ops": None}
    e = classify_edit(observed, hypothesis)
    return {f"{prefix}_edit_type": e.type, f"{prefix}_edit_ops": e.ops}


# --------------------------------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------------------------------

_STIM_CARRY = ["dataset", "subset", "item_id", "condition", "stim_uid", "model_input", "context",
               "sentence_id", "plausibility", "is_grammatical", "contrast", "intended_uids",
               "n_intended", "critical_word_idx"]


def sentences_table(stim, merged):
    rows = []
    for _, sr in stim.iterrows():
        sid = int(sr["sentence_id"])
        rec = merged.get(sid)
        row = {k: sr[k] for k in _STIM_CARRY}
        if rec is None:
            row.update(status="missing")
        else:
            st = rec.get("status")
            lzs = rec.get("logZ_stats") or {}
            hyps = rec.get("hypotheses") or []
            row.update(status=st, map=rec.get("map"),
                       map_prob=(hyps[0]["prob"] if hyps else None),
                       p_literal=rec.get("p_literal"), logZ=rec.get("logZ"),
                       logZ_std=lzs.get("std"), logZ_spread=lzs.get("spread"),
                       n_seeds=rec.get("n_seeds"), runtime_s=rec.get("runtime_s"),
                       git_sha=((rec.get("git") or {}).get("sha") or rec.get("git_commit")),
                       words_status=(rec.get("words") or {}).get("status"))
            row.update(_edit_cols(rec.get("observed", ""), rec.get("map"), "map"))
        rows.append(row)
    return pd.DataFrame(rows)


def posterior_table(stim, merged):
    rows = []
    for sid in sorted({int(x) for x in stim["sentence_id"]}):
        rec = merged.get(sid)
        if not rec or rec.get("status") != "ok":
            continue
        obs = rec.get("observed", "")
        for rank, h in enumerate(rec.get("hypotheses") or []):
            row = {"sentence_id": sid, "rank": rank, "hypothesis": h["sentence"],
                   "prob": h["prob"]}
            row.update(_edit_cols(obs, h["sentence"], "hyp"))
            rows.append(row)
    return pd.DataFrame(rows)


_UNIT_FIELDS = ["surprisal_nc", "surprisal_lm", "p_copy", "p_sub", "p_ins", "p_err",
                "p_err_positional", "del_before"]


def _words_rows(sid, seed_label, rec):
    wb = rec.get("words")
    if not isinstance(wb, dict) or wb.get("status") != "ok":
        return []
    rows = []
    for u in wb.get("units") or []:
        row = {"sentence_id": sid, "seed": seed_label,
               "unit_idx": u["unit_idx"], "unit_text": u["text"],
               "stim_word_idx": u["stim_word_idx"], "is_punct": u["is_punct"],
               "n_tokens": u["n_tokens"]}
        for k in _UNIT_FIELDS:
            row[k] = u.get(k)
        rj = u.get("rejuv") or {}
        row["rejuv_n_events"] = rj.get("n_events")
        row["rejuv_change_rate"] = rj.get("change_rate")
        row["rejuv_stay_prob"] = rj.get("stay_prob")
        ind = u.get("indel") or {}
        row["indel_p_ins_gap_before"] = ind.get("p_ins_gap_before")
        row["indel_p_del"] = ind.get("p_del")
        row["indel_n_chosen_ins_before"] = ind.get("n_chosen_ins_before")
        row["indel_n_chosen_del"] = ind.get("n_chosen_del")
        row["surprisal_end_nc"] = wb.get("surprisal_end_nc")
        row["surprisal_end_lm"] = wb.get("surprisal_end_lm")
        row["del_after_last"] = wb.get("del_after_last")
        rows.append(row)
    return rows


def words_table(stim, merged, seeds):
    rows = []
    for sid in sorted({int(x) for x in stim["sentence_id"]}):
        if sid in merged:
            rows.extend(_words_rows(sid, "merged", merged[sid]))
        for (i, j) in sorted(k for k in seeds if k[0] == sid):
            rows.extend(_words_rows(sid, str(j), seeds[(i, j)]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------

def collect_one(ds, cfg_dir, stimuli_dir, out_root):
    slug = os.path.basename(cfg_dir)
    stim = pd.read_csv(os.path.join(stimuli_dir, f"{ds}.stimuli.csv"), keep_default_na=False)
    merged, seeds = load_results(cfg_dir)
    out_dir = os.path.join(out_root, slug, ds)
    os.makedirs(out_dir, exist_ok=True)
    sent = sentences_table(stim, merged)
    sent.to_csv(os.path.join(out_dir, "sentences.csv.gz"), index=False)
    posterior_table(stim, merged).to_csv(os.path.join(out_dir, "posterior.csv.gz"), index=False)
    words_table(stim, merged, seeds).to_csv(os.path.join(out_dir, "words.csv.gz"), index=False)
    n_sid = len({int(x) for x in stim["sentence_id"]})
    counts = {"dataset": ds, "config": slug, "stim_rows": len(stim), "inputs": n_sid,
              "ok": int((sent["status"] == "ok").sum()),
              "error": int((sent["status"] == "error").sum()),
              "missing": int((sent["status"] == "missing").sum()),
              "words_ok": int((sent.get("words_status") == "ok").sum()) if "words_status" in sent else 0}
    print(f"  {ds} @ {slug}: {counts['ok']} ok / {counts['error']} error / "
          f"{counts['missing']} missing (of {counts['stim_rows']} stimulus rows)")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*",
                    help="datasets to collect (default: every <ds>.input under the results root)")
    ap.add_argument("--results-root", default="results_nc")
    ap.add_argument("--stimuli", default=os.path.join(os.path.dirname(__file__), "stimuli"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "outputs"))
    a = ap.parse_args(argv)

    if a.datasets:
        stems = [f"{ds}.input" for ds in a.datasets]
    else:
        stems = sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.results_root, "*.input"))
                       if os.path.isdir(d))
    if not stems:
        sys.exit(f"no <dataset>.input directories under {a.results_root!r} "
                 "(run the worker first, or name datasets explicitly)")
    all_counts = []
    for stem in stems:
        ds = stem[:-len(".input")]
        cfg_dirs = sorted(d for d in glob.glob(os.path.join(a.results_root, stem, "*"))
                          if os.path.isdir(os.path.join(d, "results")))
        if not cfg_dirs:
            print(f"  {ds}: no config directories with results/ -- skipped")
            continue
        for cfg_dir in cfg_dirs:
            all_counts.append(collect_one(ds, cfg_dir, a.stimuli, a.out))

    if all_counts:                                       # status.md: the run ledger
        os.makedirs(a.out, exist_ok=True)
        df = pd.DataFrame(all_counts)
        lines = ["# Collection status", "",
                 "| dataset | config | stim rows | inputs | ok | error | missing | words ok |",
                 "|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            lines.append(f"| {r['dataset']} | {r['config']} | {r['stim_rows']} | {r['inputs']} | "
                         f"{r['ok']} | {r['error']} | {r['missing']} | {r['words_ok']} |")
        with open(os.path.join(a.out, "status.md"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {os.path.join(a.out, 'status.md')}")


if __name__ == "__main__":
    main()
