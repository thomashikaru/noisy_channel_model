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

**Harness inputs (Phase 3, planning/NOISY_CHANNEL_HARNESS_IMPLEMENTATION_PLAN.md §4).** ``--input``
may be the experiment harness's ``.jsonl`` (one ``{"sentence_id", "text", "context"}`` record per
line; ``sentence_id`` must equal the line index, which keys the per-item resume). The per-item
``context`` is fed to the model as the LM prime and stored in the record -- an edited context
recomputes the item -- and shards group by ``(context word count, unit count)``, the two XLA
compile-shape axes (the §3.5 probe: grouping by the exact pair suffices; no LCTX bucketing).
Each ok record additionally carries ``p_literal`` (posterior mass on the verbatim observed
sentence), full ``git``/``versions``/``lm_info`` provenance, and a ``words`` block of per-observed-
unit outputs -- surprisal_nc / surprisal_lm, alignment posteriors, rejuvenation statistics (see
``genjax_port.word_stats``); non-finite values are serialized as null, never as infinities. Legacy
``.txt`` inputs (one sentence per line, no context) behave exactly as before.

The ``--print-output-dir``, ``--plan`` and ``--manifest`` modes are deliberately stdlib-only (NO jax /
penzai import) so the submit script can call them cheaply on a login node without a GPU.
"""

import argparse
import glob
import json
import math
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
    if a.n_seeds > 1:                      parts.append(f"nseed{a.n_seeds}")
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


def seed_item_path(a, idx, j):
    return os.path.join(results_dir(a), f"item_{idx:05d}_s{j}.json")


def seed_viz_path(a, idx, j):
    return os.path.join(results_dir(a), f"item_{idx:05d}_s{j}.viz.json")


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


def read_items(path):
    """The input as a list of ``{"idx", "text", "context"}`` dicts (stdlib only, login-safe).

    A ``.jsonl`` input (the experiment harness) carries one ``{"sentence_id", "text", "context"}``
    record per line; blank lines and ``#`` comments are skipped, and a present ``sentence_id`` must
    equal the line index among kept lines (the worker keys results and RNG by that index, so a
    mismatch means the file was reordered and every result would silently re-map). A legacy
    ``.txt`` input is one sentence per line with no context."""
    if not path.endswith(".jsonl"):
        return [{"idx": i, "text": t, "context": ""} for i, t in enumerate(read_sentences(path))]
    items = []
    with open(path) as fh:
        for line in fh:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            rec = json.loads(t)
            if "sentence_id" in rec and int(rec["sentence_id"]) != len(items):
                raise ValueError(
                    f"{path}: kept line {len(items)} has sentence_id {rec['sentence_id']} -- the "
                    "input list is keyed by line index (append-only); a reordered/edited list "
                    "would re-map finished results onto different sentences.")
            items.append({"idx": len(items), "text": rec["text"],
                          "context": (rec.get("context") or "").strip()})
    return items


def _item_status(path, text, context=""):
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
    if (rec.get("context") or "") != (context or ""):
        return "stale"                       # the item's LM context changed -> recompute
    st = rec.get("status")
    if st == "ok":
        return "done"
    if st == "error":
        return "error"
    return "stale"


def _needs_work(path, item, overwrite, skip_errors):
    if overwrite:
        return True
    s = _item_status(path, item["text"], item.get("context", ""))
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
    """The config exactly as passed on the CLI (import-free, used by the login-side manifest). A None
    here (``wdel``/``wins``/``align_slope``/``action_alpha``) means "left at the model/channel default";
    :func:`_resolved_config` substitutes the actual effective values for the per-item records.
    ``wins_mode`` records HOW the spurious-word cost is computed, since ``wins`` (a scalar) is null
    whenever the non-scalar frequency-aware / uniform defaults are in effect."""
    return {
        "lm": _lm_name(), "channel": a.channel, "rejuv": a.rejuv,
        "particles": a.particles, "band": a.band, "max_dist": a.max_dist,
        "rejuv_lookback": a.rejuv_lookback, "seed": a.seed,
        "lm_temp": a.lm_temp, "ins_rate": a.ins_rate, "uniform_ins": a.uniform_ins,
        "wdel": a.wdel, "wins": a.wins,
        "wins_mode": ("scalar" if a.wins is not None else "uniform" if a.uniform_ins else "freq_aware"),
        "align_slope": a.align_slope, "action_alpha": a.action_alpha, "dedup": a.dedup,
        "bd_p_stay": a.bd_p_stay, "bd_mode": a.bd_mode, "bd_attempts": a.bd_attempts,
        "bd_funcwords": not a.no_bd_funcwords, "top": a.top, "n_seeds": a.n_seeds,
    }


def _resolved_config(a, pwc, channel, action_alpha):
    """:func:`_config_dict` with the model/channel defaults substituted for the knobs left at None, so
    each per-item record is self-contained (no need to cross-reference the source at ``git_commit``).
    Requires the model module (run path only) for the default constants. ``channel`` may differ from
    ``a.channel`` (char_copy + an explicit alpha resolves to word_action), and ``action_alpha`` is the
    already-parsed tuple (or None). align_slope/action_alpha are recorded as None for channels that do
    not use them (so a null there means "not applicable", distinct from a defaulted value)."""
    cfg = _config_dict(a)
    cfg["channel"] = channel
    cfg["wdel"] = a.wdel if a.wdel is not None else float(pwc.WDEL_DEFAULT)
    if channel == "align":
        cfg["align_slope"] = float(a.align_slope) if a.align_slope is not None else float(pwc.ALIGN_SLOPE)
        cfg["action_alpha"] = list(action_alpha) if action_alpha is not None else list(pwc.ALIGN_ALPHA_DEFAULT)
    elif channel == "word_action":
        cfg["align_slope"] = None                                  # not used by this channel
        cfg["action_alpha"] = list(action_alpha) if action_alpha is not None else list(pwc.ACTION_ALPHA_DEFAULT)
    else:                                                          # char_copy: no align form, no action latent
        cfg["align_slope"] = None
        cfg["action_alpha"] = None
    return cfg


def _length_key(s):
    """Cheap, stdlib-only proxy for the model's word-unit count ``M = len(obs_words)`` -- the dominant
    XLA-compile shape axis (see pairhmm_smc ``_make_kernel(seed_len, M, band, T_max, LCTX, Wmax)``).
    Counts word runs AND standalone punctuation, mirroring how the channel segments observed words.
    Exact M needs the tokenizer; this proxy clusters same-shape sentences closely enough to group
    them, and keeps ``--plan`` import-free (no transformers on the login node)."""
    return len(re.findall(r"\w+|[^\w\s]", s))


def _item_length_key(item):
    """Shard grouping key ``(context word count, unit-count proxy)`` -- BOTH compile-shape axes
    (the context sets ``seed_len``/LCTX, the text sets ``M``). The §3.5 probe measured a new
    seed_len at only ~4-5 s of one-time compile, so grouping by this exact pair is sufficient
    (no LCTX bucketing). Stdlib-only, login-safe."""
    return (len(item.get("context", "").split()), _length_key(item["text"]))


def _shard_plan(items, sort_by_length, min_size, max_size):
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
    n = len(items)
    max_size = max(1, max_size)
    min_size = max(1, min(min_size, max_size))
    if n == 0:
        return []
    if not sort_by_length:                                   # original behaviour: contiguous blocks
        return [list(range(i, min(i + max_size, n))) for i in range(0, n, max_size)]
    # Group indices by length proxy (ascending length; original order within a length), then split
    # each length group into <=max_size chunks -- so a length's sentences stay together.
    by_len = {}
    for i in sorted(range(n), key=lambda j: (_item_length_key(items[j]), j)):
        by_len.setdefault(_item_length_key(items[i]), []).append(i)
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


def _versions():
    """Installed versions of the packages the result depends on (+ the genjax checkout's commit,
    since it is installed from a local repo). Best-effort: a missing package records null."""
    import importlib.metadata as md
    out = {}
    for pkg in ("jax", "jaxlib", "penzai", "transformers", "torch", "numpy", "genjax"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    try:
        import genjax as _gj
        out["genjax_commit"] = subprocess.check_output(
            ["git", "-C", os.path.dirname(_gj.__file__), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        out["genjax_commit"] = None
    return out


def _git_info():
    """Full sha + branch + dirty flag of THIS repo (the short ``git_commit`` field is kept too)."""
    def _q(*args):
        return subprocess.check_output(["git", "-C", _REPO, *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    try:
        return {"sha": _q("rev-parse", "HEAD"), "branch": _q("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(_q("status", "--porcelain"))}
    except Exception:
        return {"sha": None, "branch": None, "dirty": None}


def _lm_info():
    """LM name + the resolved HuggingFace snapshot sha, when the hub cache layout allows."""
    name = _lm_name()
    snap = None
    try:
        from huggingface_hub import constants as _hc
        base = os.path.join(_hc.HF_HUB_CACHE, "models--" + name.replace("/", "--"), "snapshots")
        snaps = sorted(os.listdir(base))
        snap = snaps[-1] if snaps else None
    except Exception:
        pass
    return {"name": name, "hf_snapshot": snap}


def _jsonable(x):
    """Recursively make numpy values json-safe; non-finite floats become null (the words-block
    convention: an unreachable prefix mass serializes as null, never as an infinite surprisal)."""
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _unit_map(text, units, n_tokens):
    """Greedy alignment of the model's observed units to *text*'s whitespace tokens. Word units
    advance through the tokens by character consumption, so an attached punctuation unit (or a
    clitic segmented off) maps to the token it came from with ``is_punct`` = 1 for units with no
    alphanumeric character (the plan's "punctuation -> the preceding token")."""
    toks = text.split()
    out = []
    ti, pos = 0, 0                        # current whitespace token; chars of it already matched
    for i, u in enumerate(units):
        body = u.strip()
        is_punct = not any(c.isalnum() for c in body)
        if ti < len(toks) and pos > 0 and toks[ti][pos:].startswith(body):
            idx = ti                      # continuation of the current token (attached punct etc.)
        else:
            if pos > 0:
                ti, pos = ti + 1, 0      # done with that token; this unit starts a new one
            while ti < len(toks) and not toks[ti].startswith(body):
                ti += 1                   # resync guard (should not trigger on standardized text)
            idx = min(ti, len(toks) - 1)
        pos += len(body)
        if ti < len(toks):
            pos = min(pos, len(toks[ti]))
        out.append({"unit_idx": i, "text": body, "stim_word_idx": idx,
                    "is_punct": int(is_punct), "n_tokens": int(n_tokens[i])})
    return out


# --------------------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------------------

def do_print_output_dir(a):
    print(output_dir(a))


def write_manifest(a):
    """Write OUTPUT_DIR/manifest.json (full config + provenance). Idempotent; called once at submit
    time. Records git sha, input file, sentence count, shard layout -- everything analysis needs
    without parsing the directory name."""
    items = read_items(a.input)
    n = len(items)
    n_shards = len(_shard_plan(items, a.sort_by_length, a.min_shard_size, a.shard_size))
    od = output_dir(a)
    os.makedirs(results_dir(a), exist_ok=True)
    os.makedirs(logs_dir(a), exist_ok=True)
    manifest = {
        "created": _now_iso(),
        "git_commit": _git_commit(),
        "input_file": os.path.abspath(a.input),
        "input_format": ("jsonl" if a.input.endswith(".jsonl") else "txt"),
        "n_sentences": n,
        "n_with_context": sum(1 for it in items if it["context"]),
        "sharding": {"max_size": a.shard_size, "min_size": a.min_shard_size,
                     "sort_by_length": a.sort_by_length},
        "n_shards": n_shards,
        "config": _config_dict(a),
        "config_note": "as-passed; null wdel/wins/align_slope/action_alpha = channel default. "
                       "Per-item records resolve these to effective values.",
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
    items = read_items(a.input)
    n = len(items)
    plan = _shard_plan(items, a.sort_by_length, a.min_shard_size, a.shard_size)
    shards_with_work, remaining = [], 0
    for s, members in enumerate(plan):
        work = sum(1 for i in members
                   if _needs_work(item_path(a, i), items[i], a.overwrite, a.skip_errors))
        if work:
            shards_with_work.append(s)
            remaining += work
    print(f"OUTPUT_DIR={output_dir(a)}")
    print(f"TOTAL_ITEMS={n}")
    print(f"NUM_SHARDS={len(plan)}")
    print(f"REMAINING_ITEMS={remaining}")
    print("SHARDS_WITH_WORK=" + ",".join(str(s) for s in shards_with_work))


def _words_block(pwc, a, text, prime, state, log_w, ws, dg):
    """Assemble the per-observed-unit outputs (plan §4) from one run's ``word_stats`` + ``diag``:
    the prefix-mass surprisals (§3.1), the plain-LM baseline on the SAME copy spans (§3.2), the
    alignment posteriors (§3.3), and the per-slot rejuvenation statistics (§3.4 -- slot w maps to
    unit w POSITIONALLY, so under an indel-shifted parse read them as positional). Serialized via
    :func:`_jsonable` (non-finite -> null)."""
    import numpy as np
    from genjax_port import word_stats as WS
    post = WS.alignment_posteriors(state, log_w, dg)
    lm_base = pwc.lm_word_surprisals(text, prime=prime)
    units = lm_base["units"]
    umap = _unit_map(text, units, [len(sp) for sp in pwc._obs_word_spans(text)])
    M = len(units)
    S = ws["surprisal_nc"]
    e_del = post["e_del_gap"]
    sub = (ws.get("rejuv") or {}).get("sub") or {}
    attempts = (ws.get("rejuv") or {}).get("indel") or []
    indel = None
    if attempts:                                  # merge attempts: probs averaged, counts summed
        n_at = len(attempts)
        indel = {
            "n_attempts": n_at, "n_done": int(attempts[-1]["n_done"]),
            "p_noop": float(sum(at["p_noop"] for at in attempts) / n_at),
            "p_ins_gap": sum(np.asarray(at["p_ins_gap"]) for at in attempts) / n_at,
            "p_del_word": sum(np.asarray(at["p_del_word"]) for at in attempts) / n_at,
            "ins_count_gap": sum(np.asarray(at["chosen"]["ins_count_gap"]) for at in attempts),
            "del_count_word": sum(np.asarray(at["chosen"]["del_count_word"]) for at in attempts),
            "n_chosen_ins": int(sum(at["chosen"]["n_ins"] for at in attempts)),
            "n_chosen_del": int(sum(at["chosen"]["n_del"] for at in attempts)),
        }
    out_units = []
    for i in range(M):
        u = dict(umap[i])
        u.update(surprisal_nc=S[i], surprisal_lm=lm_base["surprisal_lm"][i],
                 p_copy=post["p_copy"][i], p_sub=post["p_sub"][i], p_ins=post["p_ins"][i],
                 p_err=1.0 - post["p_copy"][i], p_err_positional=post["p_err_positional"][i],
                 del_before=e_del[i])
        if i in sub:
            u["rejuv"] = dict(sub[i])
        if indel is not None:
            g = indel
            u["indel"] = {
                "p_ins_gap_before": g["p_ins_gap"][i] if i < len(g["p_ins_gap"]) else None,
                "p_del": g["p_del_word"][i] if i < len(g["p_del_word"]) else None,
                "n_chosen_ins_before": int(g["ins_count_gap"][i]) if i < len(g["ins_count_gap"]) else None,
                "n_chosen_del": int(g["del_count_word"][i]) if i < len(g["del_count_word"]) else None,
            }
        out_units.append(u)
    return _jsonable({
        "status": "ok", "prime": prime, "lm_temp": a.lm_temp, "convention": ws.get("convention"),
        "prefix_logq": ws["prefix_logq"], "surprisal_end_nc": ws["surprisal_end_nc"],
        "surprisal_end_lm": lm_base["surprisal_end_lm"], "del_after_last": e_del[M],
        "indel": indel, "units": out_units,
    })


def _run_one(pwc, a, text, context, key, channel, action_alpha, want_viz):
    """Run the model for ONE (sentence, context, key). Returns (result, viz_or_None) where result
    has status 'ok' (map/hypotheses/logZ/p_literal/words/runtime_s) or 'error' (traceback). Pure
    compute -- the caller owns file IO so the multi-seed loop can load-or-compute each seed
    independently. A non-empty context becomes the LM prime (the channel still sees only the
    target text). The words block gets its own try/except: a per-word statistics failure
    downgrades to ``words.status == "error"`` instead of failing the item."""
    import traceback
    t0 = time.time()
    prime = (context or "").strip() or pwc.PRIME
    want_words = abs(a.lm_temp - 1.0) < 1e-12     # the §3.1 estimator's convention needs lm_temp=1
    try:
        trace = [] if want_viz else None
        ws = {} if want_words else None
        dg = {} if want_words else None
        st, lw, logZ, sl = pwc.run(
            text, key, P=a.particles, band=a.band, max_dist=a.max_dist,
            wdel=a.wdel, wins=a.wins, rejuv=a.rejuv, rejuv_lookback=a.rejuv_lookback,
            trace=trace, dedup=a.dedup, lm_temp=a.lm_temp, ins_rate=a.ins_rate,
            uniform_ins=a.uniform_ins, action_alpha=action_alpha, channel=channel,
            align_slope=a.align_slope, bd_p_stay=a.bd_p_stay, bd_mode=a.bd_mode,
            bd_attempts=a.bd_attempts, bd_funcwords=not a.no_bd_funcwords,
            prime=prime, word_stats=ws, diag=dg)
        full = pwc.decode(st, lw, skip=sl, top=10 ** 9)   # full support: hypotheses + p_literal
        hyps = [{"sentence": s, "prob": float(p)} for s, p in full[:a.top]]
        p_literal = float(sum(p for s, p in full if s == text.strip()))
        res = {"status": "ok", "map": (hyps[0]["sentence"] if hyps else None),
               "hypotheses": hyps, "logZ": float(logZ), "p_literal": round(p_literal, 6)}
        if want_words:
            try:
                res["words"] = _words_block(pwc, a, text, prime, st, lw, ws, dg)
            except Exception:
                res["words"] = {"status": "error", "error": traceback.format_exc()}
        else:
            res["words"] = {"status": "skipped", "reason": "lm_temp != 1"}
        res["runtime_s"] = round(time.time() - t0, 1)
        viz = (pwc.structured_output(text, trace, float(logZ), P=a.particles, band=a.band,
                                     max_dist=a.max_dist, rejuv=a.rejuv,
                                     rejuv_lookback=a.rejuv_lookback, topk=a.viz_topk)
               if want_viz else None)
        return res, viz
    except Exception:
        return {"status": "error", "runtime_s": round(time.time() - t0, 1),
                "error": traceback.format_exc()}, None


def _merge_seeds(ok_recs):
    """Evidence-weighted merge of the successful per-seed records (each with logZ + hypotheses). The
    mixture weight of run r is proportional to Z_hat_r = exp(logZ_r), so merged P(sentence) =
    sum_r w_r * P_r(sentence) over the per-seed top-k lists -- a collapsed run (low logZ) is auto-
    down-weighted; the run(s) that found the high-evidence mode dominate. Returns (hypotheses_sorted,
    seed_weights, merged_logZ, logZ_stats). merged_logZ = log((1/R) sum_r exp(logZ_r)) =
    logsumexp(logZ_r) - log R (the unbiased combined evidence). NOTE: the merge over truncated top-k
    lists is approximate in the tail; raise --top if hypotheses are close."""
    import statistics
    logZs = [r["logZ"] for r in ok_recs]
    R = len(logZs)
    mx = max(logZs)
    w = [math.exp(lz - mx) for lz in logZs]
    sw = sum(w)
    weights = [x / sw for x in w]
    merged = {}
    for wt, r in zip(weights, ok_recs):
        for h in r["hypotheses"]:
            merged[h["sentence"]] = merged.get(h["sentence"], 0.0) + wt * h["prob"]
    hyps = sorted(({"sentence": s, "prob": p} for s, p in merged.items()), key=lambda h: -h["prob"])
    merged_logZ = mx + math.log(sw) - math.log(R)
    stats = {"per_seed": [round(z, 3) for z in logZs], "min": round(min(logZs), 3),
             "max": round(max(logZs), 3), "mean": round(statistics.fmean(logZs), 3),
             "std": round(statistics.pstdev(logZs) if R > 1 else 0.0, 3),
             "spread": round(max(logZs) - min(logZs), 3)}
    return hyps, weights, merged_logZ, stats


def _merge_words(ok_recs, weights):
    """Evidence-weighted merge of the per-seed ``words`` blocks (plan §4). Prefix masses are
    averaged in MASS space per cell (logsumexp − log R -- the formula ``_merge_seeds`` uses for
    logZ; a null cell counts as zero mass) and the surprisals RECOMPUTED from the merged masses;
    posterior expectations (p_*, deletions, indel marginals) are evidence-weighted like the
    hypotheses (w_r ∝ exp(logZ_r)); rejuvenation rates are POOLED from the raw counts;
    ``surprisal_lm`` and the unit metadata are deterministic (first contributing seed). Returns
    None when no seed produced an ok words block."""
    pairs = [(w, r) for w, r in zip(weights, ok_recs)
             if isinstance(r.get("words"), dict) and r["words"].get("status") == "ok"]
    if not pairs:
        return None
    wsum = sum(w for w, _ in pairs)
    if wsum <= 0:
        return None
    wts = [w / wsum for w, _ in pairs]
    blocks = [r["words"] for _, r in pairs]
    logZs = [r["logZ"] for _, r in pairs]
    R = len(blocks)
    first = blocks[0]

    def lse_mean(vals):
        fin = [v for v in vals if v is not None]
        if not fin:
            return None
        mx = max(fin)
        return mx + math.log(sum(math.exp(v - mx) for v in fin)) - math.log(len(vals))

    def wavg(vals):
        if len(vals) != R or any(v is None for v in vals):
            return None
        return float(sum(w * v for w, v in zip(wts, vals)))

    plq = [lse_mean([b["prefix_logq"][k] for b in blocks]) for k in range(len(first["prefix_logq"]))]
    mlz = lse_mean(logZs)
    units = []
    for i, u0 in enumerate(first["units"]):
        us = [b["units"][i] for b in blocks]
        u = {k: u0[k] for k in ("unit_idx", "text", "stim_word_idx", "is_punct", "n_tokens")}
        u["surprisal_nc"] = (None if plq[i] is None or plq[i + 1] is None else plq[i] - plq[i + 1])
        u["surprisal_lm"] = u0["surprisal_lm"]
        for k in ("p_copy", "p_sub", "p_ins", "p_err", "p_err_positional", "del_before"):
            u[k] = wavg([x.get(k) for x in us])
        rjs = [x["rejuv"] for x in us if x.get("rejuv")]
        if rjs:
            na = sum(r["n_active"] for r in rjs)
            u["rejuv"] = {"n_events": sum(r["n_events"] for r in rjs), "n_active": na,
                          "n_changed": sum(r["n_changed"] for r in rjs),
                          "stay_sum": sum(r["stay_sum"] for r in rjs),
                          "change_rate": sum(r["n_changed"] for r in rjs) / max(na, 1),
                          "stay_prob": sum(r["stay_sum"] for r in rjs) / max(na, 1)}
        ids = [x["indel"] for x in us if x.get("indel")]
        if ids:
            u["indel"] = {
                "p_ins_gap_before": wavg([d.get("p_ins_gap_before") for d in ids]),
                "p_del": wavg([d.get("p_del") for d in ids]),
                "n_chosen_ins_before": sum(d.get("n_chosen_ins_before") or 0 for d in ids),
                "n_chosen_del": sum(d.get("n_chosen_del") or 0 for d in ids)}
        units.append(u)
    return {"status": "ok", "merge": "evidence_weighted", "n_seeds": R,
            "prime": first["prime"], "lm_temp": first["lm_temp"], "convention": first["convention"],
            "prefix_logq": plq,
            "surprisal_end_nc": (None if plq[-1] is None or mlz is None else plq[-1] - mlz),
            "surprisal_end_lm": first["surprisal_end_lm"],
            "del_after_last": wavg([b.get("del_after_last") for b in blocks]),
            "units": units}


def do_run(a):
    """Process this task's shard. Loads the model once, then loops sentences with per-item resume,
    atomic writes, and per-item error capture. With --n-seeds>1, each item runs that many independent
    seeds (unique paths) and writes an evidence-weighted MERGED record as item_NNNNN.json."""
    items = read_items(a.input)
    n = len(items)
    if a.shard_index is None:
        a.shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    plan = _shard_plan(items, a.sort_by_length, a.min_shard_size, a.shard_size)
    members = plan[a.shard_index] if 0 <= a.shard_index < len(plan) else []
    mine = [(i, items[i]) for i in members]
    todo = [(i, it) for (i, it) in mine
            if _needs_work(item_path(a, i), it, a.overwrite, a.skip_errors)]

    lens = sorted({_item_length_key(it) for _i, it in mine})
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
    vers, gitinfo, lminfo = _versions(), _git_info(), _lm_info()   # once per shard
    cfg = _resolved_config(a, pwc, channel, action_alpha)   # effective values (defaults substituted)
    t_shard = time.time()

    for k, (idx, it) in enumerate(todo, 1):
        text, context = it["text"], it.get("context", "")
        el = int(time.time() - t_shard)
        print(f"[shard {a.shard_index}] [{el // 60:02d}:{el % 60:02d}] ({k}/{len(todo)}) "
              f"item {idx}: {text!r}"
              + (f" (ctx {len(context.split())}w)" if context else "")
              + (f"  ({a.n_seeds} seeds)" if a.n_seeds > 1 else ""), flush=True)
        t0 = time.time()
        base = {"idx": idx, "observed": text, "context": context, "config": cfg, "lm": cfg["lm"],
                "git_commit": git, "git": gitinfo, "versions": vers, "lm_info": lminfo,
                "slurm": slurm, "timestamp": _now_iso()}
        # The item's base RNG key (unchanged from the original single-seed path).
        item_key = jax.random.fold_in(jax.random.PRNGKey(a.seed), idx)
        try:
            if a.n_seeds <= 1:
                # Single seed: original RNG (item_key used directly) and original record schema.
                res, viz = _run_one(pwc, a, text, context, item_key, channel, action_alpha,
                                    not a.no_viz)
                if res["status"] == "ok" and viz is not None:   # viz FIRST, compact LAST
                    _atomic_write_json(viz_path(a, idx), viz)
                _atomic_write_json(item_path(a, idx), dict(base, **res))
                tag = res["map"] if res["status"] == "ok" else "ERROR"
                print(f"[shard {a.shard_index}]   {res['status']} in {res['runtime_s']:.0f}s "
                      f"-> {tag!r}", flush=True)
                continue

            # Multi-seed: run/load N independent sub-seeds to unique paths, then evidence-merge.
            per_seed = []
            for j in range(a.n_seeds):
                sp = seed_item_path(a, idx, j)
                stj = _item_status(sp, text, context)
                if not a.overwrite and (stj == "done" or (stj == "error" and a.skip_errors)):
                    with open(sp) as fh:
                        per_seed.append(json.load(fh))           # resume: reuse this seed's result
                    continue
                res, viz = _run_one(pwc, a, text, context, jax.random.fold_in(item_key, j),
                                    channel, action_alpha, not a.no_viz)
                if res["status"] == "ok" and viz is not None:
                    _atomic_write_json(seed_viz_path(a, idx, j), viz)
                rec_j = dict(base, seed_index=j, **res)
                _atomic_write_json(sp, rec_j)                    # per-seed FIRST
                per_seed.append(rec_j)
            ok = [r for r in per_seed if r.get("status") == "ok"]
            if ok:
                hyps, weights, mlogZ, stats = _merge_seeds(ok)
                merged = dict(base, status="ok", merge="evidence_weighted",
                              n_seeds=len(ok), n_seeds_requested=a.n_seeds,
                              map=(hyps[0]["sentence"] if hyps else None), hypotheses=hyps[:a.top],
                              logZ=mlogZ, logZ_stats=stats,
                              seed_weights=[round(w, 4) for w in weights],
                              runtime_s=round(time.time() - t0, 1))
                merged["p_literal"] = round(sum(w * r.get("p_literal", 0.0)
                                                for w, r in zip(weights, ok)), 6)
                mw = _merge_words(ok, weights)
                merged["words"] = (mw if mw is not None
                                   else {"status": "error",
                                         "error": "no seed produced an ok words block"})
            else:
                merged = dict(base, status="error", n_seeds=0, n_seeds_requested=a.n_seeds,
                              runtime_s=round(time.time() - t0, 1),
                              error=f"all {a.n_seeds} seeds errored")
            _atomic_write_json(item_path(a, idx), merged)        # merged LAST = completion marker
            if merged["status"] == "ok":
                print(f"[shard {a.shard_index}]   merged {len(ok)}/{a.n_seeds} seeds in "
                      f"{merged['runtime_s']:.0f}s -> {merged['map']!r}  "
                      f"(logZ {mlogZ:.2f}, spread {stats['spread']:.1f})", flush=True)
            else:
                print(f"[shard {a.shard_index}]   ERROR item {idx}: all seeds failed (continuing)",
                      flush=True)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            _atomic_write_json(item_path(a, idx),
                               dict(base, status="error", runtime_s=round(time.time() - t0, 1), error=tb))
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
    # REQUIRED, no default. This used to default to "gibbs+bd" while the interactive CLI
    # (pythia_word_caprop.py) defaulted to "off" -- so which inference regime a run got depended on how you
    # entered the code, silently. Both now force the choice. See REJUV_CHOICES in pythia_word_caprop.py.
    p.add_argument("--rejuv", choices=("off", "gibbs", "gibbs+bd"), required=True,
                   help="REQUIRED: 'off' (certified forward-only, ~15-30 s/item, cannot reach deletions) | "
                        "'gibbs+bd' (reaches deletions, +15/87 on the battery, ~180 s/item) | "
                        "'gibbs' (substitution-only; legacy, avoid for new work)")
    p.add_argument("--rejuv-lookback", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1,
                   help="run this many independent seeds per item (unique paths item_NNNNN_sJ.json) and "
                        "write an evidence-weighted MERGED record as item_NNNNN.json (logZ stats + "
                        "seed weights). 1 = original single-seed behavior. Each seed adds a config-dir "
                        "suffix so it never collides with a smaller run.")
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
