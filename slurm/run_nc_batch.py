#!/usr/bin/env python
"""Per-shard batch runner for the pair-HMM noisy-channel model (SLURM-friendly).

This is the worker that every SLURM array task runs. One array task = one *shard* of the input
file (up to ``--shard-size`` sentences; with ``--sort-by-length`` shards group same-length sentences
rather than being contiguous in file order). The model (Pythia + the JIT-compiled SMC step) is
loaded ONCE per shard and reused across that shard's sentences, so the model load is always
amortized and -- when a shard is length-homogeneous -- the JAX trace/lower compile is too.

It mirrors ``genjax_port.pythia_word_caprop.cli`` for the actual inference call, but instead of a
single ``--sentence`` it processes a shard and writes, per sentence:

  * ``results/item_NNNNN.json``      -- compact record: observed, top-k inferred + probs, logZ,
                                        runtime, the FULL resolved config, git sha, SLURM ids.
  * ``results/item_NNNNN.viz.json``  -- the directly-viz-loadable structured-output trace
                                        (same artifact as ``--output_json``; view with
                                        ``python -m genjax_port.viz``). Skipped with ``--no-viz``.

Design choices that satisfy the harness requirements:

  * **Resume / no-rerun:** an item is "done" iff its compact json exists AND its stored ``observed``
    matches the current sentence AND ``status == "ok"``. So re-submitting resumes (skips done work),
    and editing a line in the input recomputes only that line (the text no longer matches). Use
    ``--overwrite`` to force recompute.
  * **Graceful failure:** each sentence is wrapped in try/except. A crash writes a ``status:"error"``
    record (with traceback) and the shard CONTINUES to the next sentence -- one bad sentence never
    kills the rest of the shard. Error items are RETRIED on the next run (they are not "done") unless
    ``--skip-errors``.
  * **Preemption-safe writes:** every file is written to a ``.tmp`` then ``os.replace``-d (atomic), and
    the viz file is written BEFORE the compact file, so a half-finished item never looks "done".
  * **Config -> directory:** ``--print-output-dir`` / ``--plan`` resolve a config-encoded output
    directory so different configs land in different directories (easy to parse/compare later).

The ``--print-output-dir``, ``--plan`` and ``--manifest`` modes are deliberately stdlib-only (NO jax /
penzai import) so the submit script can call them cheaply on a login node without a GPU.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

# Make the in-repo package importable no matter the cwd or PYTHONPATH (slurm/ -> repo root -> src).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# --------------------------------------------------------------------------------------------------
# Lightweight helpers (stdlib only -- safe to call without a GPU / heavy imports)
# --------------------------------------------------------------------------------------------------

def _lm_name():
    """The LM is selected by the NC_LM env var (read at import time by genjax_port.lm_penzai)."""
    return os.environ.get("NC_LM", "EleutherAI/pythia-70m")


def _san(v):
    """Filesystem-safe token for a config value: ',' -> '-', keep alnum / '.' / '-', drop the rest."""
    s = str(v).replace(",", "-").replace("+", "")
    return "".join(c for c in s if c.isalnum() or c in ".-") or "x"


def config_slug(a):
    """A deterministic, human-readable directory name encoding the knobs that change the posterior.

    Core knobs (lm, channel, rejuv, particles, band, max_dist, lookback, seed) are ALWAYS present;
    optional knobs are appended only when overridden from the harness defaults -- so a vanilla run
    gets a short name and any varied knob still produces a distinct directory."""
    lm = _lm_name().split("/")[-1]
    parts = [f"lm-{_san(lm)}", f"ch-{_san(a.channel)}", f"rej-{_san(a.rejuv)}",
             f"P{a.particles}", f"b{a.band}", f"d{a.max_dist}", f"lb{a.rejuv_lookback}", f"s{a.seed}"]
    if abs(a.lm_temp - 1.0) > 1e-9:        parts.append(f"lt{_san(a.lm_temp)}")
    if abs(a.ins_rate - 0.02) > 1e-12:     parts.append(f"ins{_san(a.ins_rate)}")
    if a.uniform_ins:                      parts.append("unifins")
    if a.wdel is not None:                 parts.append(f"wdel{_san(a.wdel)}")
    if a.wins is not None:                 parts.append(f"wins{_san(a.wins)}")
    if a.align_slope is not None:          parts.append(f"K{_san(a.align_slope)}")
    if a.action_alpha is not None:         parts.append(f"a{_san(a.action_alpha)}")
    if a.bd_p_stay != 0.0:                 parts.append(f"pstay{_san(a.bd_p_stay)}")
    if a.bd_mode != "gibbs":               parts.append(f"bd-{_san(a.bd_mode)}")
    if a.bd_attempts != 1:                 parts.append(f"bdatt{a.bd_attempts}")
    if a.no_bd_funcwords:                  parts.append("nofw")
    if not a.dedup:                        parts.append("nodedup")
    return "__".join(parts)


def output_dir(a):
    stem = _san(os.path.splitext(os.path.basename(a.input))[0])
    return os.path.abspath(os.path.join(a.results_root, stem, config_slug(a)))


def results_dir(a):
    return os.path.join(output_dir(a), "results")


def logs_dir(a):
    return os.path.join(output_dir(a), "logs")


def item_path(a, idx):
    return os.path.join(results_dir(a), f"item_{idx:05d}.json")


def viz_path(a, idx):
    return os.path.join(results_dir(a), f"item_{idx:05d}.viz.json")


def read_sentences(path):
    """One sentence per line. Blank lines and lines starting with '#' are skipped. The index of a
    sentence is its position among the *kept* lines (append new sentences at the END to maximize
    resume reuse -- inserting in the middle shifts indices and recomputes the shifted tail)."""
    out = []
    with open(path) as fh:
        for line in fh:
            t = line.strip()
            if t and not t.startswith("#"):
                out.append(t)
    return out


def _item_status(path, text):
    """'done' | 'error' | 'stale' | 'missing' for an existing (or absent) compact record."""
    if not os.path.exists(path):
        return "missing"
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except Exception:
        return "stale"                       # corrupt / partially-written -> recompute
    if rec.get("observed") != text:
        return "stale"                       # the input line changed -> recompute
    st = rec.get("status")
    if st == "ok":
        return "done"
    if st == "error":
        return "error"
    return "stale"


def _needs_work(path, text, overwrite, skip_errors):
    if overwrite:
        return True
    s = _item_status(path, text)
    if s == "done":
        return False
    if s == "error" and skip_errors:
        return False
    return True                              # missing / stale / (error and retrying)


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", _REPO, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _now_iso():
    # datetime is fine here (this is a plain script, not a workflow); UTC, no tz dependency.
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def _config_dict(a):
    """The full resolved config, embedded in every record and in the manifest (self-describing)."""
    return {
        "lm": _lm_name(), "channel": a.channel, "rejuv": a.rejuv,
        "particles": a.particles, "band": a.band, "max_dist": a.max_dist,
        "rejuv_lookback": a.rejuv_lookback, "seed": a.seed,
        "lm_temp": a.lm_temp, "ins_rate": a.ins_rate, "uniform_ins": a.uniform_ins,
        "wdel": a.wdel, "wins": a.wins, "align_slope": a.align_slope,
        "action_alpha": a.action_alpha, "dedup": a.dedup,
        "bd_p_stay": a.bd_p_stay, "bd_mode": a.bd_mode, "bd_attempts": a.bd_attempts,
        "bd_funcwords": not a.no_bd_funcwords, "top": a.top,
    }


def _length_key(s):
    """Cheap, stdlib-only proxy for the model's word-unit count ``M = len(obs_words)`` -- the dominant
    XLA-compile shape axis (see pairhmm_smc ``_make_kernel(seed_len, M, band, T_max, LCTX, Wmax)``).
    Counts word runs AND standalone punctuation, mirroring how the channel segments observed words.
    Exact M needs the tokenizer; this proxy clusters same-shape sentences closely enough to group
    them, and keeps ``--plan`` import-free (no transformers on the login node)."""
    return len(re.findall(r"\w+|[^\w\s]", s))


def _shard_plan(sentences, sort_by_length, min_size, max_size):
    """Deterministic assignment of sentence indices to shards: returns a list of index-lists where
    shard ``i`` -> the ORIGINAL indices it processes. Output files stay named by original index, so
    this changes shard *membership* only, never item identity -- resume is unaffected, and you can
    change the sharding knobs between runs without invalidating finished items.

    ``sort_by_length`` groups same-length sentences so each shard's process pays the JAX trace/lower
    compile ~once and same-shape items reuse the in-process jit cache (the persistent on-disk cache
    does NOT help here -- the cost is tracing/lowering, not XLA backend compile). ``min_size`` /
    ``max_size`` bound shard size: a shard is closed at a length boundary only once it has reached
    ``min_size`` (so we don't spawn tiny shards), and never exceeds ``max_size`` (so no shard runs too
    long). An undersized tail is merged back into the previous shard."""
    n = len(sentences)
    max_size = max(1, max_size)
    min_size = max(1, min(min_size, max_size))
    if n == 0:
        return []
    if not sort_by_length:                                   # original behaviour: contiguous blocks
        return [list(range(i, min(i + max_size, n))) for i in range(0, n, max_size)]
    # Group indices by length proxy (ascending length; original order within a length), then split
    # each length group into <=max_size chunks -- so a length's sentences stay together.
    by_len = {}
    for i in sorted(range(n), key=lambda j: (_length_key(sentences[j]), j)):
        by_len.setdefault(_length_key(sentences[i]), []).append(i)
    chunks = []
    for L in sorted(by_len):
        g = by_len[L]
        chunks.extend(g[j:j + max_size] for j in range(0, len(g), max_size))
    # Fold an undersized shard into the next chunk (keeps full same-length chunks intact while
    # merging only the small leftovers, so most shards land in [min_size, max_size]).
    shards = []
    for ch in chunks:
        if shards and len(shards[-1]) < min_size and len(shards[-1]) + len(ch) <= max_size:
            shards[-1].extend(ch)
        else:
            shards.append(list(ch))
    if len(shards) >= 2 and len(shards[-1]) < min_size \
            and len(shards[-2]) + len(shards[-1]) <= max_size + min_size:
        shards[-2].extend(shards.pop())                      # fold a too-small final shard back
    return shards


def _slurm_meta():
    e = os.environ
    return {k: e.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                                  "SLURMD_NODENAME") if e.get(k)}


# --------------------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------------------

def do_print_output_dir(a):
    print(output_dir(a))


def write_manifest(a):
    """Write OUTPUT_DIR/manifest.json (full config + provenance). Idempotent; called once at submit
    time. Records git sha, input file, sentence count, shard layout -- everything analysis needs
    without parsing the directory name."""
    sents = read_sentences(a.input)
    n = len(sents)
    n_shards = len(_shard_plan(sents, a.sort_by_length, a.min_shard_size, a.shard_size))
    od = output_dir(a)
    os.makedirs(results_dir(a), exist_ok=True)
    os.makedirs(logs_dir(a), exist_ok=True)
    manifest = {
        "created": _now_iso(),
        "git_commit": _git_commit(),
        "input_file": os.path.abspath(a.input),
        "n_sentences": n,
        "sharding": {"max_size": a.shard_size, "min_size": a.min_shard_size,
                     "sort_by_length": a.sort_by_length},
        "n_shards": n_shards,
        "config": _config_dict(a),
        "config_slug": config_slug(a),
        "output_dir": od,
        "write_viz": not a.no_viz,
    }
    _atomic_write_json(os.path.join(od, "manifest.json"), manifest)
    return manifest


def do_plan(a):
    """Compute which shards still have remaining work (for the submit-time preflight) and (re)write
    the manifest. Prints a machine-parseable block the submit script greps."""
    write_manifest(a)
    sents = read_sentences(a.input)
    n = len(sents)
    plan = _shard_plan(sents, a.sort_by_length, a.min_shard_size, a.shard_size)
    shards_with_work, remaining = [], 0
    for s, members in enumerate(plan):
        work = sum(1 for i in members
                   if _needs_work(item_path(a, i), sents[i], a.overwrite, a.skip_errors))
        if work:
            shards_with_work.append(s)
            remaining += work
    print(f"OUTPUT_DIR={output_dir(a)}")
    print(f"TOTAL_ITEMS={n}")
    print(f"NUM_SHARDS={len(plan)}")
    print(f"REMAINING_ITEMS={remaining}")
    print("SHARDS_WITH_WORK=" + ",".join(str(s) for s in shards_with_work))


def do_run(a):
    """Process this task's shard. Loads the model once, then loops sentences with per-item resume,
    atomic writes, and per-item error capture."""
    sents = read_sentences(a.input)
    n = len(sents)
    if a.shard_index is None:
        a.shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    plan = _shard_plan(sents, a.sort_by_length, a.min_shard_size, a.shard_size)
    members = plan[a.shard_index] if 0 <= a.shard_index < len(plan) else []
    mine = [(i, sents[i]) for i in members]
    todo = [(i, t) for (i, t) in mine
            if _needs_work(item_path(a, i), t, a.overwrite, a.skip_errors)]

    lens = sorted({_length_key(t) for _i, t in mine})
    span = f"{lens[0]}" if len(lens) == 1 else (f"{lens[0]}..{lens[-1]}" if lens else "-")
    print(f"[shard {a.shard_index}] {len(mine)} sentences (length-units {span}) of {n}; "
          f"{len(todo)}/{len(mine)} need work "
          f"(est ~{len(todo) * a.est_seconds_per_item // 60 + 1} min of inference + model load)",
          flush=True)
    if not todo:
        print(f"[shard {a.shard_index}] nothing to do; exiting.", flush=True)
        return

    os.makedirs(results_dir(a), exist_ok=True)

    # --- heavy imports happen ONLY here (run mode), so --plan/--print-output-dir stay GPU-free ---
    import jax
    from genjax_port import pythia_word_caprop as pwc

    # Resolve the channel / action-alpha exactly as the CLI does.
    channel = a.channel
    action_alpha = None
    if a.action_alpha is not None:
        action_alpha = tuple(float(x) for x in a.action_alpha.split(","))
        if channel == "char_copy":
            channel = "word_action"

    git = _git_commit()
    slurm = _slurm_meta()
    cfg = _config_dict(a)
    t_shard = time.time()

    for k, (idx, text) in enumerate(todo, 1):
        el = int(time.time() - t_shard)
        print(f"[shard {a.shard_index}] [{el // 60:02d}:{el % 60:02d}] "
              f"({k}/{len(todo)}) item {idx}: {text!r}", flush=True)
        t0 = time.time()
        base = {"idx": idx, "observed": text, "config": cfg, "lm": cfg["lm"],
                "git_commit": git, "slurm": slurm, "timestamp": _now_iso()}
        try:
            key = jax.random.fold_in(jax.random.PRNGKey(a.seed), idx)
            trace = [] if not a.no_viz else None
            st, lw, logZ, sl = pwc.run(
                text, key, P=a.particles, band=a.band, max_dist=a.max_dist,
                wdel=a.wdel, wins=a.wins, rejuv=a.rejuv, rejuv_lookback=a.rejuv_lookback,
                trace=trace, dedup=a.dedup, lm_temp=a.lm_temp, ins_rate=a.ins_rate,
                uniform_ins=a.uniform_ins, action_alpha=action_alpha, channel=channel,
                align_slope=a.align_slope, bd_p_stay=a.bd_p_stay, bd_mode=a.bd_mode,
                bd_attempts=a.bd_attempts, bd_funcwords=not a.no_bd_funcwords)
            top = pwc.decode(st, lw, skip=sl, top=a.top)
            hyps = [{"sentence": s, "prob": float(p)} for s, p in top]
            runtime = time.time() - t0
            rec = dict(base, status="ok", map=(hyps[0]["sentence"] if hyps else None),
                       hypotheses=hyps, logZ=float(logZ), runtime_s=round(runtime, 1))
            # viz FIRST, compact LAST: the compact file's presence implies the viz file is complete.
            if not a.no_viz:
                viz = pwc.structured_output(text, trace, float(logZ), P=a.particles, band=a.band,
                                            max_dist=a.max_dist, rejuv=a.rejuv,
                                            rejuv_lookback=a.rejuv_lookback, topk=a.viz_topk)
                _atomic_write_json(viz_path(a, idx), viz)
            _atomic_write_json(item_path(a, idx), rec)
            print(f"[shard {a.shard_index}]   ok in {runtime:.0f}s -> {rec['map']!r}", flush=True)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            rec = dict(base, status="error", runtime_s=round(time.time() - t0, 1), error=tb)
            _atomic_write_json(item_path(a, idx), rec)       # keep going; this item retries next run
            print(f"[shard {a.shard_index}]   ERROR on item {idx} (continuing):\n{tb}", flush=True)

    el = int(time.time() - t_shard)
    print(f"[shard {a.shard_index}] done in {el // 60:02d}:{el % 60:02d}.", flush=True)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # modes
    p.add_argument("--print-output-dir", action="store_true",
                   help="print the resolved config-encoded output directory and exit (no GPU).")
    p.add_argument("--plan", action="store_true",
                   help="write the manifest and print which shards still have work (no GPU).")
    p.add_argument("--manifest", action="store_true",
                   help="write OUTPUT_DIR/manifest.json and exit (no GPU).")
    # batch / IO
    p.add_argument("--input", required=True, help="text file: one observed sentence per line")
    p.add_argument("--results-root", default="results_nc", help="root of the results tree")
    p.add_argument("--shard-size", type=int, default=8,
                   help="MAX sentences per shard / array task (a shard never exceeds this)")
    p.add_argument("--min-shard-size", type=int, default=4,
                   help="with --sort-by-length, the minimum sentences per shard before closing at a "
                        "length boundary (avoids tiny shards / imbalance)")
    p.add_argument("--sort-by-length", action="store_true",
                   help="group same-length sentences into shards so each shard's process pays the JAX "
                        "trace/lower compile ~once (same-shape items reuse the in-process jit cache). "
                        "Changes shard membership only; outputs are still keyed by original index.")
    p.add_argument("--shard-index", type=int, default=None,
                   help="which shard to run (default: $SLURM_ARRAY_TASK_ID, else 0)")
    p.add_argument("--overwrite", action="store_true", help="recompute even if outputs exist")
    p.add_argument("--skip-errors", action="store_true",
                   help="treat prior error records as done (do not retry them)")
    p.add_argument("--no-viz", action="store_true", help="do not write the heavy viz-trace json")
    p.add_argument("--viz-topk", type=int, default=8, help="hypotheses kept per step in the viz json")
    p.add_argument("--top", type=int, default=5, help="top-k inferred sentences saved per item")
    p.add_argument("--est-seconds-per-item", type=int, default=200,
                   help="rough per-item runtime, only used to print an up-front estimate")
    # model knobs (mirror genjax_port.pythia_word_caprop.cli; the LM is set via NC_LM)
    p.add_argument("--channel", choices=("align", "word_action", "char_copy"), default="align")
    p.add_argument("--particles", type=int, default=128)
    p.add_argument("--band", type=int, default=2)
    p.add_argument("--max-dist", type=int, default=2)
    p.add_argument("--rejuv", choices=("off", "gibbs", "gibbs+bd"), default="gibbs+bd")
    p.add_argument("--rejuv-lookback", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lm-temp", type=float, default=1.0)
    p.add_argument("--ins-rate", type=float, default=0.02)
    p.add_argument("--uniform-ins", action="store_true")
    p.add_argument("--wdel", type=float, default=None)
    p.add_argument("--wins", type=float, default=None)
    p.add_argument("--align-slope", type=float, default=None)
    p.add_argument("--action-alpha", default=None, help="'copy,sub,ins,del' or 'align,ins,del'")
    p.add_argument("--no-dedup", dest="dedup", action="store_false", default=True)
    p.add_argument("--bd-p-stay", type=float, default=0.0)
    p.add_argument("--bd-mode", default="gibbs")
    p.add_argument("--bd-attempts", type=int, default=1)
    p.add_argument("--no-bd-funcwords", action="store_true")
    return p


def main():
    a = build_parser().parse_args()
    if a.print_output_dir:
        do_print_output_dir(a)
    elif a.manifest:
        write_manifest(a)
        print(output_dir(a))
    elif a.plan:
        do_plan(a)
    else:
        do_run(a)


if __name__ == "__main__":
    main()
