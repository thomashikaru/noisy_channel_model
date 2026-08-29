# Noisy-channel experiment harness — implementation plan (approved 2026-08-29)

Companion to `NOISY_CHANNEL_EXPERIMENT_HARNESS.md` (the goals). This file is the *how*: architecture,
per-phase work, pseudocode for the non-obvious parts, verification, and the decisions taken. The
code-facts below (paths, line numbers) were verified against branch `rejuv-birth-death @ c6a2459` on
2026-08-29 and will drift — re-grep symbols before editing.

> ## Build status — read this before trusting a section
>
> | phase | section | status |
> |---|---|---|
> | 0 branch + scaffold | — | **done** (`108beeb`) |
> | 1 stimulus harmonization | §2 | **done** (`4f02275`) — **§2 is now a design record, not a spec.** Five details below were wrong about the data and the shipped converters deviate; `experiments/README.md` is authoritative for what exists |
> | 2 per-word model outputs | §3 | **not started — this is next** |
> | 3 worker + output schema | §4 | not started |
> | 4 configs, smoke, cost probes | §5 | not started |
> | 5 cluster runs | §6 | not started |
> | 6 documentation | §7 | not started |
>
> **Where §2 was wrong** (each measured; see `experiments/README.md` for the full account):
> 1. The gibson2013/chen2023 counterpart is the plausible row of the **other** structure, not the same
>    one. The same-structure rule yields zero single-edit rows across all 120 implausible gibson rows
>    (`sub;sub` 78 / `ins;sub` 37 / `del;sub` 5); the cross-structure rule yields `ins` 58 / `del` 60.
> 2. chen2023's two-space split recovers the wrong target on 27 of 320 context rows (20 three-space
>    separators, 7 where a context sentence is one-space separated and bleeds into the target). The
>    no-context file is used as the target authority instead; §2.3's `assert` would have caught all 27.
> 3. tabor2004's relativizer is not always `"who was"` — also `that was` (14), `who were` (3),
>    `which was` (2), `who is` (2), `that were` (2). It is read off the reduced/nonreduced diff.
> 4. clark2026 also maps `1.2 -> 2.2`, the mirror of `2.1 -> 1.1` (same 36 word pairs, opposite
>    direction). Direction matters: only 62/72 off-diagonal repairs are reachable at `max_dist=2`, and
>    two are one-directional.
> 5. huang2024's `disambPosition_0idx` is **not** off by one. It is correct on all 144 rows against
>    `model_input`, where punctuation stays attached; the off-by-one only affects the space-split
>    `sentence_norm` form. §2.2's note to the contrary is wrong.
>
> The schema also gained `contrast` (design-level relation) plus `edit_ops` / `edit_from` / `edit_to`,
> because `edit_type` alone splits chen2023's voice alternation arbitrarily (90 `multi` vs 30 `sub`,
> depending only on whether the participle is irregular).

## 0. What exists, what is missing

**Exists and is reused unchanged as the compute layer**

- `slurm/run_nc_batch.py` — per-shard worker: one model load per shard, length-bucketed sharding, per-item
  resume (done ⇔ file exists ∧ stored `observed` matches ∧ `status=="ok"`), atomic writes, per-item
  try/except, `--n-seeds` with an evidence-weighted merged posterior + `logZ_stats`, and every record stamped
  with git sha, fully resolved config, SLURM ids, UTC timestamp. Stdlib-only `--plan` / `--manifest` /
  `--print-output-dir` modes are safe on the login node.
- `slurm/submit_nc_batch.sh` — env-var → sbatch generator; `INPUT` and `REJUV` are required (rejuv has NO
  default anywhere — deliberate; never add one); `DRYRUN=1` previews and prints `REMAINING_ITEMS`.
- `.claude/skills/orcd-cluster/SKILL.md` — SSH/MFA/sync/submit/monitor/fetch runbook (user runs
  `ssh -fN orcd` once per session; always `ssh -O check orcd` first; never `ssh -O exit`; never run
  inference on the login node).
- `planning/bd_mem_probe.py "<sentence>" <P> [rejuv]` — one-run runtime + peak-RSS probe (x86 cluster RSS
  is 1.7–2× the Mac's; size `MEM` from the longest input, then double).
- `calibration_word_action_smc._wellform` (capitalize + terminal period) and `calibration_gate.word_change`
  (difflib edit classification) — reused by the converters.
- Output contract `planning/TRACE_SCHEMA.md` (the `.viz.json` step trace) and the compact per-item record.

**Missing (the harness fills these)**

1. A dataset layer: converters from the raw `data/<study>` formats to one schema + model-input lists.
2. Stable item identity (the worker keys by line index; factorial designs repeat sentences).
3. The requested per-word outputs — they do not exist in the model; the needed quantities are computed and
   discarded inside `pairhmm_smc.run` / `pairhmm_rejuv` (see §3).
4. Multi-dataset orchestration, a smoke-test tier, a cost-probe step, version capture (package versions,
   branch, dirty flag), a dataset-agnostic aggregator, and a committed launch log.

## 1. Architecture

```
experiments/                         # NEW, tracked. The harness. Nothing here reads human data.
  README.md                          # end-to-end reproduction (another cluster, a new dataset)
  RUNLOG.md                          # append-only: every launch (date, commit, config, job ids, MEM, outcome)
  converters/                        # raw data/<study>/… -> common schema (one module per study)
    __init__.py  common.py  gibson2013.py  chen2023.py  ryskin2021.py  qian2023.py
    huang2024.py  clark2026.py  tabor2004.py  moses.py
  build_stimuli.py                   # runs converters -> stimuli/, checks invariants, writes MANIFEST.json
  stimuli/                           # tracked (small): the harmonized stimuli = the reproducibility anchor
    <dataset>.stimuli.csv            # one row per (item, condition): common schema
    <dataset>.input.jsonl            # model input: UNIQUE (context, text) pairs, append-only order
    smoke.input.jsonl  probe.input.jsonl   # pipeline smoke set / worst-case-length cost-probe set
    MANIFEST.json                    # per dataset: source files + sha256, counts, converter commit, build time
  configs/                           # named model configs as env files for slurm/submit_nc_batch.sh
    smoke.env  main.env  main_off.env  main_bd.env
  run.sh                             # fetch-tabor | build | smoke-local | probe | submit | status | pull | collect
  collect.py                         # results_nc/**/item_*.json -> outputs/<config>/<dataset>/*.csv.gz
  outputs/                           # tidy results (.csv.gz) — UNTRACKED (gitignored); raw item_*.json stay in results_nc/
src/genjax_port/word_stats.py        # NEW: per-word quantities (host-side); small hooks in pairhmm_smc / pairhmm_rejuv / pythia_word_caprop
slurm/run_nc_batch.py                # worker extensions: jsonl input with context, `words` block, versions
```

Data flow: `data/<study>` → converters → `stimuli/*.stimuli.csv` + `*.input.jsonl` → git push/pull to the
cluster → `submit_nc_batch.sh INPUT=experiments/stimuli/<ds>.input.jsonl` + config env → per-item JSON in
`results_nc/<ds>.input/<config_slug>/results/` → rsync pull → `collect.py` → `outputs/`.

Identity: each stimulus row has `stim_uid = <dataset>/<subset>/<item_id>/<condition>` and a `sentence_id` =
line index into the dataset's `input.jsonl` (rows with identical `(context, model_input)` share one
`sentence_id`). Results join on `(dataset, sentence_id)`; the stimuli CSV carries everything else.

## 2. Phase 1 — stimulus harmonization

### 2.1 Common schema (`<dataset>.stimuli.csv`)

| field | notes |
|---|---|
| `dataset`, `subset` | e.g. `gibson2013`, `dopo_to`; `subset=""` when none |
| `item_id`, `condition`, `stim_uid` | strings; `stim_uid = dataset/subset/item_id/condition` |
| `sentence_orig` | original orthography when the source has it (gibson2013, clark2026, huang2024 pivot, chen2023, moses, tabor); empty for ryskin2021 / qian2023 (lost upstream) |
| `sentence_norm` | lowercase, punctuation split — the old pipeline's convention, kept for joins to legacy files |
| `model_input` | **what the model reads** — standardized: initial capital, punctuation attached, terminal `.`/`?` (the calibration-battery convention; fixes the leading-opener artifact) |
| `context` | clean preceding text for the LM prime (chen2023 supportive / non-supportive); empty otherwise |
| `sentence_id` | index into `<dataset>.input.jsonl` |
| `plausibility`, `is_grammatical` | per-dataset derivations (table below) |
| `intended_uid`, `intended_text`, `edit_type` | the plausible/control counterpart row; its `model_input`; difflib class (`sub`/`ins`/`del`/`none`/`multi`) |
| `critical_word_idx` | 0-based index into whitespace tokens of `model_input` (huang2024 recomputed; clark2026 from `CriticalWord`; tabor: the RC participle) |
| `comprehension_q`, `correct_answer` | gibson2013 / chen2023 (a normative answer key, not human data) |
| `meta` | JSON of dataset-specific columns |

### 2.2 Per-dataset mapping (converters open ONLY the listed files)

| dataset | source | item / condition | intended counterpart | notes |
|---|---|---|---|---|
| gibson2013 (dopo_to, dopo_for, transitive_intransitive) | `materials.csv` (`Item,Type,Structure,Plausibility,Sentence,Question,Answer`) | `Item`; `Structure_Plausibility` | same Item + Structure, `plausible` | 20 items × 4 per subset. transitive_intransitive's `Plausibility` label is inverted relative to `Answer` — carry both, flag in README |
| chen2023 (dopo_to, active_passive) | 6 Linger `.txt` (`no-context`, `supportive`, `non-supportive`) | `#` header fields 3–4; `context ∈ {none, supportive, non_supportive}` | as gibson2013 | CRLF; `dopo-to-supportive.txt` is mojibake (`‚Äô` = a double-encoded `’`); context + target on ONE line separated by two spaces; 48 fillers per file (`condition == filler`) excluded; the no-context dopo_to targets are byte-identical to gibson2013/dopo_to (kept, noted) |
| ryskin2021 | `materials.csv` (`Item,Condition,sentence`) | `Item`; `Condition ∈ {Control,SemCrit,Sem,Synt}` | same Item, `Control` | 126 items × 4 (Item ids 1–159 with gaps); lowercase-only |
| qian2023 | `materials.csv` (`…,condition,sentence`) | `item`; `condition` (`sss`…`ppp`: N1, N2, verb number) | same item, verb number := N1 number | all 480 rows; `is_grammatical = cond[0]==cond[2]` |
| huang2024 (SAP ClassicGP) | `items_ClassicGP.pivot.csv` (original case) | `item`; `condition` (NPS/NPZ/MVRR × AMB/UAMB) | the `*_UAMB` sibling | 24 × 6; recompute `critical_word_idx` on `model_input` tokens — the stored `disambPosition_0idx` is off by one on all 24 `NPZ_UAMB` rows once the comma is split |
| clark2026 | `materials.csv` (+ `raw_materials.csv` segments) | `Item`; `Label` | `Typo1.1→1.1`, `Typo2.2→2.2`, `2.1→1.1`, else none | 36 × 10; **never opens `exp_data_merged.csv` or `lists/`** |
| tabor2004 | `data/tabor2004/items.csv` (OSF `f8qwh`, file `y4872`; sha256 pinned; `run.sh fetch-tabor`) | `item_nr`; `reduced_rel_coherence` | the `nonreduced` sibling | `;`-separated, 128 rows = 32 items × 4 (2×2 `reduced_rel` × `coherence`); Paape, Smith & Vasishth (2025)'s adaptation of Tabor et al. (2004) |
| moses | `raw_materials.csv` | 1 item | none | single demo item |

### 2.3 Pseudocode

```python
# experiments/converters/common.py
HOLDOUT_PATHS = ("data/clark2026/exp_data_merged.csv", "data/clark2026/lists/")   # human data: never opened
SOURCES_SEEN = {}                                   # path -> sha256, filled by open_source, dumped into MANIFEST

def open_source(path, encoding="utf-8", errors="strict"):
    p = str(path)
    assert not any(p.startswith(h) for h in HOLDOUT_PATHS), f"hold-out file: {p}"
    SOURCES_SEEN[p] = sha256(file bytes)
    return open(p, encoding=encoding, errors=errors, newline="")   # newline="" so CRLF survives for csv

PUNCT = r"([.,?!;:\"])"
def normalize(s):                                   # the old pipeline's sentences.txt convention (verified)
    s = re.sub(PUNCT, r" \1", s.lower()); return " ".join(s.split())

def standardize(s):                                 # -> model_input (battery convention)
    s = " ".join(s.split())
    s = re.sub(r"\s+([.,?!;:])", r"\1", s)          # re-attach punctuation
    s = re.sub(r"\s+'", "'", s)                     # re-attach clitic apostrophes if a source split them
    if s and s[0].islower(): s = s[0].upper() + s[1:]
    if s and s[-1] not in ".?!": s += "."
    return s

def edit_type(observed, intended): return calibration_gate.word_change(observed, intended)  # reuse

def critical_index(model_input, word):              # index of `word` among whitespace tokens (punct stripped)
    toks = [t.strip(".,?!;:") for t in model_input.split()]
    return toks.index(word) if word in toks else None

@dataclass
class StimRow:  dataset; subset; item_id; condition; stim_uid; sentence_orig; sentence_norm; model_input;
                context; sentence_id; plausibility; is_grammatical; intended_uid; intended_text; edit_type;
                critical_word_idx; comprehension_q; correct_answer; meta
```

```python
# experiments/converters/chen2023.py   (the only from-scratch parser)
def read_linger(path):
    raw = open_source(path, encoding="utf-8", errors="surrogateescape").read()
    if "‚Äô" in raw:                                # the mojibake signature (‚Äô)
        raw = raw.encode("cp1252", errors="surrogateescape").decode("utf-8")   # undo the double encoding
    raw = raw.replace("’", "'").replace("\r", "")
    records = []
    for line in raw.split("\n"):
        if line.startswith("# "):  group, item, cond = line[2:].split()[:3]; cur = dict(group=group, item=item, cond=cond)
        elif line.startswith("? "): q, ans = line[2:].rsplit(" ", 1); cur.update(question=q, answer=ans); records.append(cur)
        elif line.strip():          cur["text"] = line.strip()
    return records

def convert(subset):                                                  # dopo_to | active_passive
    by_ctx = {ctx: read_linger(f"data/chen2023/{subset}/{subset_file(ctx)}") for ctx in ("none", "supportive", "non_supportive")}
    target_of = {(r["item"], r["cond"]): r["text"] for r in by_ctx["none"] if r["cond"] != "filler"}
    for ctx, recs in by_ctx.items():
        for r in recs:
            if r["cond"] == "filler": continue                        # user decision: fillers excluded
            pieces = [p for p in r["text"].split("  ") if p]          # context sentences + target, two-space separated
            target = pieces[-1]; context = " ".join(pieces[:-1])
            assert target == target_of[(r["item"], r["cond"])], "target must match the no-context file"
            yield StimRow(dataset="chen2023", subset=subset, item_id=r["item"], condition=r["cond"],
                          sentence_orig=target, sentence_norm=normalize(target), model_input=standardize(target),
                          context=standardize_context(context), plausibility=r["cond"].split("_")[1],
                          is_grammatical=True, intended_uid=plausible_sibling(r), comprehension_q=r["question"],
                          correct_answer=r["answer"], meta=json.dumps(dict(group=r["group"], context=ctx)))
```

```python
# experiments/build_stimuli.py
for ds, convert in CONVERTERS.items():
    rows = list(convert())
    inputs = OrderedDict()                                            # (context, model_input) -> sentence_id
    for r in rows: r.sentence_id = inputs.setdefault((r.context, r.model_input), len(inputs))
    resolve intended_text/edit_type by stim_uid lookup; assert every intended_uid resolves
    existing = read_jsonl(f"stimuli/{ds}.input.jsonl") if exists
    if existing and not args.rebuild:
        assert existing == new[:len(existing)], "input list is append-only (worker resume is keyed by line index)"
    write stimuli csv (sorted columns), input jsonl [{"sentence_id", "text", "context"}], MANIFEST entry:
        {sources: SOURCES_SEEN, n_rows, rows_per_condition, n_inputs, converter_commit, built_at}
emit smoke.input.jsonl (the harness doc's 7 examples + 1 chen2023 context item) and probe.input.jsonl
(the longest model_input per dataset + the longest-context chen2023 item)
invariants: huang2024 critical token == expected word for every row; no hold-out path in SOURCES_SEEN
```

## 3. Phase 2 — model-side per-word outputs (`src/genjax_port/word_stats.py` + hooks)

All opt-in and host-side: `pairhmm_smc.run(..., word_stats=None, diag=None)` (like `trace` / `rejuv_stats`).
`None` ⇒ bit-identical (gated by a test). The certified files gain only argument plumbing and extra *return
values of arrays they already compute* — no new arithmetic inside jitted code, so XLA fusion cannot change
the certified numbers.

### 3.1 Noisy-channel per-observed-word surprisal — prefix-mass estimator from ONE run

Definition. `S_k = −log(Q_k / Q_{k−1})` for observed unit k = 1..M, where `Q_k` is the total mass of
generative paths whose *last event emitted `o_k`* — by an intended word (the diag arc of
`word_dp._word_row_update`) or as a spurious insertion (the insertion sweep, possibly after a deletion).
Each path emits `o_k` exactly once, so these events partition the paths; deletion-terminated states are
excluded (they would double-count the state before the deletion). This is the observed-prefix marginal
under the model's own partial-mass convention (see the caveat).

Estimator (per SMC step s, computed **before** the resample test, right after `log_w += incr`,
`pairhmm_smc.py:800`; particle mapping is the identity at that point):

```python
# word_stats.py
def _emission_row(alpha_prev, emit_col, wdel, wins):
    """Second output of the SAME scan as word_dp._word_row_update: the part of the new row whose last
    event emitted the observed unit (diag arc, or an insertion into this cell). em[0] = -inf."""
    M = emit_col.shape[0]
    diag = alpha_prev[0:M] + emit_col
    up = alpha_prev[1:M + 1] + wdel
    beta = concat([(alpha_prev[0] + wdel)[None], logsumexp(stack([diag, up]), axis=0)])
    def step(left, x):
        b, w, d = x
        cell = logsumexp(stack([b, left + w]))        # identical to _word_row_update
        em = logsumexp(stack([d, left + w]))          # diag + insertion into this cell
        return cell, em
    _, em = lax.scan(step, beta[0], (beta[1:], broadcast(wins, (M,)), diag))
    return concat([[-inf], em])
# identity (tested): logaddexp(em[1:], alpha_prev[1:] + wdel) == _word_row_update(...)[1:]

def emission_masses(alpha_prev, word_surf_post, n_words_prev, emit_full, copy_mask, lp_copy, lp_sub, wdel_p, wins_p, band_mask):
    surf = word_surf_post[p, n_words_prev[p]]                                  # the word this step appended
    cols = emit_full[:, surf].T + lp_sub[:, None] + (lp_copy - lp_sub)[:, None] * copy_mask[:, surf].T   # same formula as _caprop_scores / channel_carry
    em = vmap(_emission_row)(alpha_prev, cols, wdel_p, wins_p)
    return band_mask(em, n_words_prev + 1)                                    # mask AFTER the sweep, like the kernel

class PrefixAccumulator:                                                     # host side, float64
    def __init__(self, a0p):                                                 # (P, M+1), the leading-insertion init
        a = np.asarray(a0p, float64); P = len(a)
        self.logq = np.full(M + 1, -inf); self.logq[0] = 0.0
        self.logq[1:] = logsumexp(a[:, 1:] - logsumexp(a, axis=1)[:, None], axis=0) - log(P)   # RELATIVE, like logZ
    def add(self, em, alpha_post, done_post, log_w, logZ_acc):
        g = np.asarray(em, float64) - logsumexp(np.asarray(alpha_post), axis=1)[:, None]
        g[np.asarray(done_post)] = -inf                                      # EOS is not an observed-unit emission
        contrib = logZ_acc + np.asarray(log_w)[:, None] - log(P) + g           # (P, M+1)
        self.logq = np.logaddexp(self.logq, logsumexp(contrib, axis=0))
    def finish(self, logZ_final):
        S = self.logq[:-1] - self.logq[1:]                                   # S_k, k = 1..M
        return dict(prefix_logq=self.logq, surprisal_nc=S, surprisal_end_nc=self.logq[M] - logZ_final)
```

Hook points in `pairhmm_smc.run`: signature (`:468`); `acc = PrefixAccumulator(a0p)` after `a0p` is built
(`:617`); `acc.add(emission_masses(log_alpha, state[4], n_words, …costs in force this step…), state[5],
state[6], log_w, logZ)` between `:800` and `:801` (loop-top `log_alpha`, `n_words` are the pre-extension
state; `lp_copy/lp_sub/wdel_p/wins_p` are the costs the step used — the θ refresh happens later at `:847`);
`word_stats.update(acc.finish(float(logZ)))` after `:900`. Cost: one extra vmapped row per step, P×(M+1).

Why this is right (design-review conclusions): after `log_w += incr` the cloud is properly weighted for the
step-s target ∝ `P_LM(x_{1:s}) · Σ_k alpha_s[k]` on live particles (done particles carry the full joint and
add nothing new; `incr = 0` for them); the unnormalized estimator of `Σ_x target(x)·g(x)` is
`exp(logZ_acc) · (1/P) Σ_p exp(log_w_p) g_p`; the sweep-then-refresh at resample events changes states
only *after* this point and is target-invariant. Band: only steps `s ∈ [k−band, k+band]` contribute to
`Q_k`. `Σ_k S_k + S_end = −logZ` holds by construction.

Caveats to document in the output (`convention` field): (a) the form channel is an unnormalized edit kernel
(`Σ_o e^{K·d(o,x)}` ≈ 1.05–1.4 per word), so `S_k` carries a near-constant offset (~0.05–0.3 nats) plus a
small word-dependent part — a regression intercept absorbs it; it is the same convention `logZ` already
uses. (b) Requires `lm_temp == 1` (assert; the default). (c) `prefix_logq[k] = −inf` (unreachable under the
band) is written as `null`, never as an infinite surprisal.

### 3.2 Plain-LM per-word surprisal (baseline)

`pythia_word_caprop.lm_word_surprisals(observed, prime)`: seed `[EOS] + encode(prime)` exactly as
`_pythia_model` does (`:353`); units and spans from `_obs_word_units` / `_obs_word_spans` (`:273–284`);
**score the model's COPY spans** — factor the sentence-initial leading-space restoration (`:306–313`) into a
shared `_copy_span(body, obs_span)` so the baseline scores exactly the token spans every verbatim-copy
particle scores; one `lm_penzai.seq_token_logprobs` forward (`lm_penzai.py:119`) over
`seed + concat(spans) + [EOS]`; unit surprisal = −Σ of its token logprobs; `surprisal_end_lm = −log P(EOS | …)`.

### 3.3 Per-observed-word P(error) — host-side forward-backward on the final cloud

After `run` returns, a `diag` dict (filled in place after `:900`) exposes `emit_full, copy_mask, lp_copy,
lp_sub, wdel_p, wins_p, theta, band, M, obs_words` (locals of `run`; `emit_full` is a few MB — used
in-process, never serialized). `word_stats.alignment_posteriors(state, log_w, diag)` runs, per particle
(no dedup — θ differs per particle after a refresh; the lattice is (M+3)×(M+1) cells, vectorized over P in
numpy), a forward pass that stores rows and a backward pass **on the unmasked lattice with the band applied
only to row-to-row arcs** (the insertion sweep is unmasked within a row and the mask is applied after it —
mirroring the kernel; a naive masked backward does not sum to 1):

```
forward (per particle; n = number of intended words; col_i = emit_full[:, surf_i] + lp_sub + (lp_copy−lp_sub)·copy_mask[:, surf_i]):
  Au[0] = a0 (unmasked);  A[0] = mask(Au[0], 0)
  for i in 1..n:  beta[0] = A[i−1][0] + wdel;  beta[k] = lae(A[i−1][k−1] + col_i[k−1], A[i−1][k] + wdel)
                  Au[i][k] = lae(beta[k], Au[i][k−1] + wins[k−1])   (left-to-right, unmasked)
                  A[i] = mask(Au[i], i)
  total = A[n][M]     (== the particle's log_alpha[M])
backward:
  Bu[n][M] = 0 if inband(n, M) else −inf;   Bu[n][k] = Bu[n][k+1] + wins[k]
  for i = n−1..0:  out[k] = inband(i, k) ? lae(Bu[i+1][k+1] + col_{i+1}[k], Bu[i+1][k] + wdel) : −inf
                   Bu[i][M] = out[M];  Bu[i][k] = lae(out[k], Bu[i][k+1] + wins[k])   (right-to-left)
arc posteriors (− total):
  diag[i][k] = A[i−1][k−1] + col_i[k−1] + Bu[i][k]     o_k emitted by x_i  (copy iff copy_mask[k−1, surf_i], else sub)
  up[i][k]   = A[i−1][k]   + wdel       + Bu[i][k]     x_i deleted at gap k
  ins[i][k]  = Au[i][k−1]  + wins[k−1]  + Bu[i][k]     o_k spurious after row i   (i = 0..n; row 0 = leading)
invariants (asserted): per k, Σ_i P(diag[i][k]) + Σ_i P(ins[i][k]) = 1;  per i<n, lse_k(A[i][k] + out_i[k]) = total
```

Per-unit outputs `p_copy, p_sub, p_ins` (sum to 1) and per-gap `E[deletions]`, averaged over particles with
`softmax(log_w)` (terminal-corrected). The cheap positional approximation
(`1 − copy_mask[i, word_surf[p, i]]`, what `_action_counts` uses; exact only when no indel shifted the
alignment) is recorded as `p_err_positional` for comparison.

### 3.4 Per-word rejuvenation statistics

"Acceptance rate" is ≡ 1 for the production Gibbs moves (they always apply; `move_logw ≈ 0`). The
meaningful per-word statistics are computed and discarded today:

- Substitution sweep: `pairhmm_rejuv._apply_move` (`:361`) already has `s_new` (slot 0 = keep the current
  word) and `target`; return them as extra outputs; `step` / `move` (`:397–443`) pass them through; the
  `sweep` closure (`:533`) takes `stats=None` and appends `(w, s_new, target, active = w < n_words)` per
  slot. Host-side: `change_rate[w] = mean(s_new ≠ 0 | active)`, `stay_prob[w] = mean(softmax(target)[:, 0])`,
  `n_events[w]`. Surfaced in `rejuv_info` (`pairhmm_smc.py:831`, so the trace/`TRACE_SCHEMA.md` get them)
  and accumulated in `word_stats["rejuv"]`. Slot w ↔ observed unit w positionally (the pool is positional).
- Gibbs indel move: `_indel_apply` (`:1086`) returns `idx`; `make_gibbs_indel_sweep.sweep` (`:1198`) takes
  `stats=`; from the host-visible `logits`: `p_noop`, `p_ins_gap[g] = Σ_c p[:, 1 + g·Kc + c]`,
  `p_del_word[i] = p[:, 1 + Wmax·Kc + i]`, plus counts of the chosen edits (decoded with the same arithmetic
  as `:1096–1100`); weighted by `softmax(log_w)` at the call site (`:890`). `bd_mode="mh"` (non-default) not
  instrumented.

### 3.5 Context prime (chen2023)

`pythia_word_caprop.run(prime=context)`; `_pythia_model` rebuilds only `seed_ids` per prime (cheap — the
vocab tables are prime-independent). The real cost is indirect: `seed_len` sets `LCTX`, and every distinct
`LCTX` is a new XLA compile shape (transformer forward, kernel, rejuv steps, KV setup) — tens of extra
compiles per shard over ~160 contexts. Mitigation (a): the worker sorts each shard by exact
`(seed_len, M)`. Mitigation (b), only if the probe says it is needed: opt-in `lctx_round` bucketing
(right-pad with EOS past `ctx_len` — exact under causal attention; three places: `run :593`,
`_tail_inputs`, `_kv_setup`; default `None` ⇒ bit-identical). Probe first: one item at three seed lengths,
first vs second call (~2 min).

### 3.6 Tests (`src/genjax_port/tests/test_word_stats.py`; register in `tests/run.py`)

Reuse `_toy_model`, `_emit_table`, `_peaked`, `_bigram_table`, `_a0_const`, `WDEL/WINS`,
`_wa_emit_copymask_costs` from `test_pairhmm_exact.py`.

1. Row identity (property test): `logaddexp(_emission_row(...)[1:], alpha_prev[1:] + wdel) ==
   _word_row_update(...)[1:]` on random inputs.
2. Exact prefix masses vs SMC on the toy (`"teh cat sat"`, band=1): enumerate intended prefixes up to
   `M+1` words, LM prefix probability with NO EOS term, masked forward + `_emission_row`; compare
   `mean over 4 seeds of prefix_logq[k]` to the exact value (relative to `_a0_const`) at the existing logZ
   gate's tolerance, P=6000; also `S_end`.
3. Backward pass vs brute-force alignment enumeration (n ≤ 5, M ≤ 5; band=None and band=1; the band is
   checked at end-of-row only, exactly as the kernel); the two invariants above.
4. Hooks off ⇒ `np.array_equal` on every state leaf, `log_w`, `logZ`, for `char_copy/off` and the WA `gibbs`
   and `gibbs+bd` configs (covers the extra jit outputs).
5. Pythia smoke (in `test_pythia_word_caprop.py`): finite `surprisal_nc/lm`, `p_*` sum to 1,
   `Σ S_k + S_end ≈ −logZ`, positional-vs-DP agreement reported.

## 4. Phase 3 — worker and output schema

`slurm/run_nc_batch.py`:

```python
def read_items(path):                    # stdlib only (keeps --plan/--manifest jax-free)
    if path.endswith(".jsonl"):  items = [json.loads(l) for l in lines if l.strip() and not l.startswith("#")]
    else:                        items = [{"sentence": s} for s in read_sentences(path)]      # legacy .txt
    for i, it in enumerate(items): it["idx"] = i; it.setdefault("context", None)
    return items
# _item_status: also compare rec.get("context") == item["context"]  (edited context => recompute)
# _length_key: (context word count, unit count)  so shards group by both compile axes
# _run_one(text, context, ...): prime = (context or "").strip() or pwc.PRIME
#     st, lw, logZ, sl = pwc.run(text, key, ..., prime=prime, word_stats=ws, diag=dg)
#     words = assemble(ws, alignment_posteriors(st, lw, dg), lm_word_surprisals(text, prime), unit_map(text))
```

Per-item record additions (per seed and merged):

```
words: { prime, lm_temp, convention, prefix_logq[M+1], surprisal_end_nc, surprisal_end_lm, del_after_last,
         units: [ {unit_idx, text, stim_word_idx, is_punct, n_tokens, surprisal_nc, surprisal_lm,
                   p_copy, p_sub, p_ins, p_err, p_err_positional, del_before,
                   rejuv: {n_events, change_rate, stay_prob},
                   indel: {p_ins_gap_before, p_del, n_chosen_ins_before, n_chosen_del}} ] }
p_literal            # posterior mass on the verbatim observed sentence
versions             # jax, jaxlib, penzai, transformers, torch, genjax (+commit), numpy
git: {sha (full), branch, dirty}     lm: {name, hf_snapshot (if resolvable)}
```

Multi-seed merge (`_merge_words` beside `_merge_seeds`, `:379`): `prefix_logq` / `logZ` — plain mean of
masses (`logsumexp_r − log R`, the formula `merged_logZ` already uses; then recompute `S_k`, `S_end`);
posterior expectations (`p_*`, deletions, indel marginals) — evidence-weighted (`w_r ∝ exp(logZ_r)`, as the
hypotheses merge); rejuv rates — pooled counts; `surprisal_lm` — deterministic (seed 0). `--top` raised to 20
for the main runs.

`experiments/collect.py` (pandas): walks `results_nc/<ds>.input/<slug>/results/`, joins
`(dataset, sentence_id)` to `stimuli/<ds>.stimuli.csv`, writes `outputs/<slug>/<ds>/sentences.csv.gz`
(logZ, logZ_std, MAP, map_prob, p_literal, edit ops vs observed, runtime, git sha, status),
`posterior.csv.gz` (rank, hypothesis, prob, edit ops), `words.csv.gz` (long format: one row per unit ×
seed/merged), and `status.md` (per dataset × config: ok / error / missing). Unit ↔ stimulus-word map: greedy
alignment of `_obs_word_units(model_input)` to whitespace tokens (punctuation units → the preceding token,
`is_punct = 1`).

## 5. Phase 4 — configs, smoke tests, cost probes (before any large run)

- `configs/smoke.env`: `CHANNEL=align REJUV=off PARTICLES=16 N_SEEDS=1 WRITE_VIZ=0 TOP=10`
- `configs/main.env` (shared): `CHANNEL=align PARTICLES=64 N_SEEDS=4 REJUV_LOOKBACK=6 BAND=2 MAX_DIST=2 SEED=0 WRITE_VIZ=0 SORT_BY_LENGTH=1 TOP=20`
  — the stability benchmark's general-purpose point (P=64, lb6, ≥4 seeds evidence-merged).
- `configs/main_off.env` = `main.env` + `REJUV=off`; `configs/main_bd.env` = `main.env` + `REJUV=gibbs+bd`.
  **The off-vs-rejuv comparison keeps everything else constant** (user decision); `run.sh` refuses a pair of
  configs that differ in anything but `REJUV`.
- Local smoke (Mac, ≤10 min): worker on `smoke.input.jsonl` with `smoke.env` (~8 sentences × ~5 s + model
  load), then `collect.py` — verifies build → run → words block → tidy outputs end to end; one
  `gibbs+bd P=16` sentence to exercise the indel stats; then the seed-length compile probe (§3.5).
- Cost probe (Mac, 1 seed each): `planning/bd_mem_probe.py` (add a context argument) over
  `probe.input.jsonl` at the main config; set cluster `MEM ≈ 2× Mac peak RSS`, `SECONDS_PER_ITEM ≈ 3× Mac
  runtime`. Expected worst cases: tabor (16 units), chen2023 with context (prime +≈40 tokens).
- Cluster smoke: `smoke.input.jsonl` with `main_bd.env` on ORCD (DRYRUN first) → pull → collect → inspect.

Budget (to be replaced by probe numbers): ≈2,340 unique inputs (gibson 240, chen 480, ryskin 504, qian 480,
huang 144, clark 360, tabor 128, moses 1). `main_off` (P64 × 4 seeds) ≈ 1 min/item/seed on x86 → ~150–200
job-hours (≈8–10 h wall at 20 parallel). `main_bd` (P64 × 4 seeds) ≈ 25–40 min/item → ~1,000–1,500
job-hours (≈2–3 days wall).

## 6. Phase 5 — cluster runs

- Sync: commit; `git push origin experiment-harness` (ask first — outward-facing); on the cluster
  `git pull --ff-only`; confirm `git rev-parse --short HEAD` matches local. Always `ssh -O check orcd` first;
  never `ssh -O exit`; login node only for `--plan`, `squeue`, `sacct`, `tail`.
- `run.sh submit <dataset> <config>`: composes `INPUT=experiments/stimuli/<ds>.input.jsonl` + the config
  env + `MEM` / `SECONDS_PER_ITEM` / `SENTENCES_PER_SHARD` / `CPUS` from the probe; `DRYRUN=1` first
  (prints `REMAINING_ITEMS`), then submits; appends a RUNLOG entry (date, commit, dataset, config slug,
  sbatch line, job id). Order: `main_off` on all datasets, then `main_bd` dataset by dataset, shortest first.
- `run.sh status`: `squeue` / `sacct` summaries + DRYRUN remaining counts per (dataset, config).
- `run.sh pull`: rsync `results/` + `manifest.json` + `logs/submit.sbatch` into `results_nc/`; then collect.
- OOM: re-run the remaining items at a higher `MEM` (the battery runbook); log it in RUNLOG.

## 7. Phase 6 — documentation

`experiments/README.md`: purpose; phenomena → dataset table with citations (incl. the Tabor provenance);
the common schema; standardization rules; the blind-protocol statement; the exact command sequence
(fetch-tabor → build → smoke → probe → submit → pull → collect); adding a dataset (one converter + one
MANIFEST entry); reproducing on another SLURM cluster (`slurm/setup_env.sh`, `cluster.env.example`); the
output schema incl. the surprisal-convention caveat; the RNG contract (`fold_in(PRNGKey(seed), sentence_id)`
then `fold_in(·, seed_index)`). Also: `planning/TRACE_SCHEMA.md` (per-slot rejuv stats), `slurm/README.md`
(jsonl input, `words` block).

## 8. Verification

1. `conda run -n ncgenjax python -m pytest -q src/genjax_port/tests/` — existing 43 + new gates pass;
   `word_stats=None` bit-identical.
2. `python experiments/build_stimuli.py` — MANIFEST counts: gibson 240, chen 480, ryskin 504, qian 480,
   huang 144, clark 360, tabor 128, moses 1; every `intended_uid` resolves; huang2024 `critical_word_idx`
   tokens are the expected words; no hold-out file opened.
3. Local smoke: `outputs/smoke/…/words.csv.gz` has one row per unit with finite surprisals,
   `Σ S_k + S_end ≈ −logZ`, `p_*` sum to 1; "The mother gave the candle the daughter." shows deletion mass
   at the gap before "the daughter" under `gibbs+bd`.
4. Probe numbers recorded in RUNLOG before any submit; cluster smoke pulled and collected before full runs.
5. Full runs: `status.md` 100 % `ok` per (dataset, config); spot-check the harness-doc examples
   (candle/daughter → "to"; inflection → infection; licked → kicked).

## 9. Decisions and assumptions

Decisions taken with the user (2026-08-29): both `main_off` and `main_bd`, identical except `REJUV`
(P=64, 4 seeds, lb6); chen2023 context fed as the clean LM prime (channel sees only the target); uniform
battery standardization for all datasets; `experiments/stimuli/` tracked, `experiments/outputs/`
untracked; qian2023 all 480 rows; chen2023 fillers excluded; no punctuation-pool config.

Assumptions: surprisal = the prefix-mass estimator from one run (no separate prefix runs; no
"posterior-expected LM surprisal of the inferred word" column for now); "rejuv acceptance rate" = per-word
change rate + stay probability + event count; the Tabor family = the Paape/Smith/Vasishth (2025)
replication items, labelled `tabor2004`, fillers excluded; hypothesis 3 is evaluated under default
parameters ("that" is in the indel move's fixed insertion pool, "," is not — comma restoration is reachable
only via the filter's top-J LM bridges; reported as a model property, not patched); laptop use limited to
the smoke, the compile probe and one cost probe per dataset's longest input.

## 10. Sources (Tabor materials)

- Paape, D., Smith, G., & Vasishth, S. (2025). *Do local coherence effects exist in English reduced relative
  clauses?* Journal of Memory and Language, 140. OSF: https://osf.io/f8qwh/ — `Materials/items.csv`
  (download https://osf.io/download/y4872/; 128 rows; header
  `item_type;item_nr;reduced_rel;coherence;src_orc;sentence;question;ans1;ans2;ans3;corr_ans`;
  sha256 `f4030aadc67662f22c4d14a8ee5ee6a6b1d089a038bdb8c321079059bd01175e`). Local copy:
  `data/tabor2004/items.csv` (gitignored) with `data/tabor2004/SOURCE.md`.
- Paape, D., Vasishth, S., & Engbert, R. (2021). Open Mind — German materials only (not usable here).
- Tabor, W., Galantucci, B., & Richardson, D. (2004). JML 50(4), 355–370 — the original; its appendix is not
  openly accessible.
