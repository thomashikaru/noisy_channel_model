"""Rejuvenation cost/activity split probe (2026-09-02, targeted-rejuv step 2).

Question: under the main.env configuration (align, P=64, band 2, lb 6, LOOKAHEAD=1,
LA_PROPOSAL=1), how does gibbs+bd's extra cost split between (a) the per-resample-event
substitution sweep and (b) the ONE post-loop Gibbs indel move -- and how much of the
per-event sub-sweep work actually changes anything?  NB the deployed ``bd_mode="gibbs"``
indel move already fires exactly ONCE post-loop (pairhmm_smc.py ~line 1029); only the sub
sweep fires per event, so "run the indel grid less often" is not an available lever.

Method: the smoke set (8 items), one seed, three arms differing ONLY in ``rejuv``:
off / gibbs (sub sweep only) / gibbs+bd (sub sweep + one indel move).  Each run goes through
the harness worker's own ``_run_one`` (same RNG convention, same words block), so
T_gibbs - T_off ~= the sub-sweep cost and T_bd - T_gibbs ~= the indel move cost, and the
words block reports per-unit sub-sweep events/changes and indel choices.

Usage (ncgenjax env, repo root):
    python -u planning/rejuv_event_cost_probe.py run      # appends results.jsonl (resume-safe)
    python -u planning/rejuv_event_cost_probe.py report   # tables from results.jsonl
"""

import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(REPO, "src"), os.path.join(REPO, "slurm")):
    if p not in sys.path:
        sys.path.insert(0, p)

SMOKE = os.path.join(REPO, "experiments", "stimuli", "smoke.input.jsonl")
OUT = os.path.join(REPO, "planning", "rejuv_event_cost_probe.results.jsonl")
CONFIGS = ["off", "gibbs", "gibbs+bd"]


def _items():
    with open(SMOKE) as f:
        return [json.loads(line) for line in f if line.strip()]


def _args():
    """Worker-arg Namespace matching main.env + the worker's argparse defaults (the parser lives
    inline in run_nc_batch.main(), so it is restated here; _run_one reads only these fields)."""
    import argparse
    return argparse.Namespace(
        channel="align", particles=64, band=2, max_dist=2, rejuv="off",
        rejuv_lookback=6, seed=0, n_seeds=1, lm_temp=1.0, ins_rate=0.02,
        uniform_ins=False, wdel=None, wins=None, align_slope=None, action_alpha=None,
        dedup=True, bd_p_stay=0.0, bd_mode="gibbs", bd_attempts=1,
        no_bd_funcwords=False, lookahead=True, lookahead_proposal=True,
        top=20, no_viz=True, viz_topk=8)


def run():
    import jax
    import run_nc_batch as W
    from genjax_port import pythia_word_caprop as pwc

    a = _args()
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {(r["config"], r["idx"]) for r in map(json.loads, f) if r.get("ok")}
    items = _items()
    with open(OUT, "a") as out:
        for it in items:
            idx, text, context = int(it["sentence_id"]), it["text"], it.get("context", "")
            key = jax.random.fold_in(jax.random.PRNGKey(a.seed), idx)
            for cfg in CONFIGS:
                if (cfg, idx) in done:
                    continue
                a.rejuv = cfg
                t0 = time.time()
                res, _ = W._run_one(pwc, a, text, context, key, a.channel, None, False)
                wall = time.time() - t0
                rec = {"config": cfg, "idx": idx, "text": text, "ctx_words": len(context.split()),
                       "ok": res["status"] == "ok", "wall_s": round(wall, 1),
                       "runtime_s": res.get("runtime_s"), "logZ": res.get("logZ"),
                       "map": res.get("map"), "p_literal": res.get("p_literal"),
                       "words": res.get("words") if res["status"] == "ok" else None,
                       "error": res.get("error")}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"[{idx} {cfg}] {wall:.0f}s ok={rec['ok']} logZ={rec['logZ']}", flush=True)
    print("run done", flush=True)


def report():
    recs = [json.loads(line) for line in open(OUT)]
    by = {(r["config"], r["idx"]): r for r in recs if r["ok"]}
    idxs = sorted({i for _, i in by})
    print("item | M | T_off | T_gibbs | T_bd | sub_cost | indel_cost | "
          "sub nE(sum) | sub changed | indel p_noop | ins | del")
    tot = dict(off=0.0, gibbs=0.0, bd=0.0)
    for i in idxs:
        ro, rg, rb = by.get(("off", i)), by.get(("gibbs", i)), by.get(("gibbs+bd", i))
        if not (ro and rg and rb):
            continue
        w = rb["words"] or {}
        units = w.get("units") or []
        n_ev = sum((u.get("rejuv") or {}).get("n_events", 0) for u in units)
        ch = sum((u.get("rejuv") or {}).get("n_changed", 0) for u in units)
        ind = w.get("indel") or {}
        m = len(units)
        to, tg, tb = ro["runtime_s"], rg["runtime_s"], rb["runtime_s"]
        tot["off"] += to; tot["gibbs"] += tg; tot["bd"] += tb
        pn = ind.get("p_noop")
        print(f"{i} | {m} | {to:.0f} | {tg:.0f} | {tb:.0f} | {tg - to:+.0f} | {tb - tg:+.0f} | "
              f"{n_ev} | {ch} | {pn if pn is None else round(pn, 3)} | "
              f"{ind.get('n_chosen_ins')} | {ind.get('n_chosen_del')}")
    n = len(idxs)
    print(f"\nTOTAL ({n} items): off {tot['off']:.0f}s, gibbs {tot['gibbs']:.0f}s, "
          f"gibbs+bd {tot['bd']:.0f}s")
    if tot["off"]:
        print(f"sub-sweep cost = {tot['gibbs'] - tot['off']:.0f}s "
              f"({(tot['gibbs'] - tot['off']) / (tot['bd'] - tot['off']) * 100:.0f}% of rejuv extra), "
              f"indel move = {tot['bd'] - tot['gibbs']:.0f}s "
              f"({(tot['bd'] - tot['gibbs']) / (tot['bd'] - tot['off']) * 100:.0f}%); "
              f"bd/off ratio {tot['bd'] / tot['off']:.1f}x")
    # logZ sanity: gibbs+bd vs off per item
    print("\nitem | logZ off | gibbs | gibbs+bd | map_bd==map_off")
    for i in idxs:
        ro, rg, rb = by.get(("off", i)), by.get(("gibbs", i)), by.get(("gibbs+bd", i))
        if not (ro and rg and rb):
            continue
        print(f"{i} | {ro['logZ']:.2f} | {rg['logZ']:.2f} | {rb['logZ']:.2f} | "
              f"{ro['map'] == rb['map']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "report": report}[cmd]()
