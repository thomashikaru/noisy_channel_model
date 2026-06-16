# Plan: rich structured (JSON) output from a model run

**Status:** not started. Written 2026-06-16 as a cold-start handoff for a fresh session.
**Context to read first:** memory `genjax-native-migration.md` (the ALIGNED CONDITIONAL REJUVENATION +
perf sections) and `genjax-port-settled-decisions.md` (resample every word ⇒ ESS_THRESH=Inf ⇒
**post-resample particle weights are uniform**, so empirical particle counts ARE the posterior probs —
this is why the prefix distribution can be plain counts). The filter is
`src/genjax_port/smc_substitution.py` (`run_smc_substitution`); the production rejuv path is
`rejuv_bridge.run_smc_conditional_rejuv_aligned`; the CLI is `src/genjax_port/run.py`.

**Goal:** emit a JSON artifact alongside the human-readable run output, capturing three quantities the
filter already computes internally. Everything is keyed by **observed-word index** `wi` (0-based), so
the natural shape is one record per observed word plus a per-event rejuvenation log. Gate the whole
feature behind a new `--output_json PATH` flag (default off → zero change to the existing fast path).

The three features (with the decisions made WITH the user, 2026-06-16):
1. **Per-word surprisal** — the particle-filter estimate already used by the surprisal gate:
   `surprisal = -step_lmw` where `step_lmw = logsumexp(log_w) - log P` is the word's log-evidence
   (`smc_substitution.py:214`). One scalar per observed word (shared across particles).
2. **Per-location rejuvenation acceptance — PER (event, word) detail** (a sparse `t × word` view):
   for each firing event `t`, for each target word in its lookback window, the attempts and accepts.
   Only when `--conditional_rejuv` is on.
3. **Distribution over the inferred latent prefix at each incremental position — TOP-K per position**:
   at the end of each word step, the K most common decoded intended-prefixes across particles, with
   counts, plus a residual mass for the dropped tail.
Plus (decided: **yes, include**) the **auxiliary gate diagnostics** already computed: each word's
**unigram surprisal**, the **gate fire-probability** `custom_sigmoid(surprisal − unigram_surp, …)`,
and the **per-step min-ESS**.

---

## Where each quantity already lives (no new computation, just extraction)

- **Surprisal:** `smc_substitution.py:214` (`step_lmw`); the negation `-step_lmw` is already passed to
  the hook at `:251-253`. Per observed word `wi`.
- **Per-step ESS:** `smc_substitution.py:216` (`1.0 / sum(softmax(log_w)**2)`) — currently only folded
  into the running `min_ess`; capture the per-step value too.
- **Prefix distribution:** the particle buffers `intended_buf [P,M]` + `i_len [P]` at the end of each
  `wi` iteration (after emission AND after the rejuv hook). Decode exactly like the final sentences
  (`smc_substitution.py:255`): `decode(intended_buf[p, 1:int(i_len[p])]).strip()`, then `Counter`.
- **Rejuv acceptance:** today the hook accumulates only AGGREGATE `stats["accepts"]/["attempts"]`
  (`rejuv_bridge.py:621-622`). Per-(event,word) needs (a) the fused move to return **per-window-word**
  accepts instead of the single `acc_tot[P]`, and (b) the hook to log one event record per firing.
- **Gate fire-prob:** computed in the hook as `p_fire = custom_sigmoid(surprisal - unigram_surp[t],
  center, spread)` (`rejuv_bridge.py` in `_make_aligned_subflip_hook`, ~`:597`). Scalar per word `t`.
- **Unigram surprisal:** `unigram.unigram_surprisal(word_str)` — deterministic from the word string,
  so compute it in `run.py` from `noise_word.segment_words(obs)` for BOTH arms (the forward-only arm
  has no hook). The hook already precomputes the same list.

---

## JSON schema (concrete target)

```json
{
  "observed": "The little boy licked the ball into the net.",
  "config": {"lm": "EleutherAI/pythia-70m", "particles": 100, "max_dist": 2,
             "max_deletions": 1, "allow_insertion": true, "conditional_rejuv": true,
             "lookback": 2, "logprob_thresh": 0.0, "logprob_spread": 1.0, "n_sweeps": 2,
             "seed": 0, "json_topk": 5},
  "log_marginal": -50.0,
  "min_ess": 57.4,
  "accept_rate": 0.971,
  "posterior": [["The little boy kicked the ball into the net.", 81, 0.81],
                ["The little boy licked the ball into the net.", 12, 0.12]],
  "words": [
    {"index": 3, "word": "licked",
     "surprisal": 11.8, "unigram_surprisal": 12.6, "gate_p": 0.31, "step_min_ess": 64.0,
     "prefix_topk": [["The little boy kicked", 70], ["The little boy licked", 25]],
     "prefix_residual_count": 5},
    "... one per observed word ..."
  ],
  "rejuv_events": [
    {"t": 3, "gate_p": 0.31, "fired_particles": 40,
     "targets": [{"word": 1, "attempts": 80, "accepts": 50},
                 {"word": 2, "attempts": 80, "accepts": 33},
                 {"word": 3, "attempts": 80, "accepts": 71}]}
  ]
}
```

Notes: `attempts` per (event `t`, target word `w`) = `fired_particles × n_sweeps` (each sweep is one
attempt at `w`); `accepts` = accepted moves at `w` summed over gated particles and sweeps. `posterior`
is the final top-K (reuse `run.py`'s `summarize`). `words[*].gate_p` is the same scalar as the matching
`rejuv_events[t].gate_p` (merge for convenience); omit `gate_p`/`rejuv_events` entirely when rejuv off.

---

## Implementation steps

1. **A `record` accumulator threaded through the filter.** Add `record=None` to
   `run_smc_substitution`. When not `None`, after each `wi` iteration append to `record["words"]` a
   dict `{index, word, surprisal=-step_lmw, step_min_ess, prefix_topk, prefix_residual_count}` and
   decode the prefix top-K there (it has `decode` imported, `Counter` from collections). Capture the
   per-step ESS at `:216` into a local before folding into `min_ess`. **K** comes in via the param
   (default 5). Build `prefix_topk` from `Counter(prefixes).most_common(K)`; residual = `P - sum(topk
   counts)`. This is the ONLY place that decodes per step.
2. **Per-(event,word) rejuv detail.** Change the fused move
   `rejuv_bridge._aligned_window_move_fn` (`:473`) to accumulate accepts **per window column** and
   return `(key, buf, accs[nwin])` instead of `(key, buf, acc_tot[P])` — accumulate
   `accs[j] += jnp.sum(jnp.where(gate, acc, 0))` inside the `for j` loop. (Keep it RNG-faithful — do
   NOT change the key threading; this is the path the perf fix just landed, guarded by
   `test_manual_subflip_detailed_balance` + `test_aligned_conditional_composes_with_forward_deletions`.)
   In `_make_aligned_subflip_hook` (`:591`), derive the aggregate stats from `accs` (`stats["accepts"]
   += int(jnp.sum(accs))`, `stats["attempts"] += int(sum(gate))*n_sweeps*len(win)` — unchanged total),
   AND when `record is not None` append one `rejuv_events` entry `{t, gate_p, fired_particles=int(sum
   gate), targets=[{word: win[j], attempts: int(sum gate)*n_sweeps, accepts: int(accs[j])} for j]}`.
   Pass `record` into `_make_aligned_subflip_hook` and `run_smc_conditional_rejuv_aligned` (which also
   forwards it to `run_smc_substitution`). `win` is in sweep order, so `accs[j] ↔ win[j]`.
3. **CLI + assembly in `run.py`.** Add `--output_json PATH` and `--json_topk` (default 5). Build a
   `record = {"words": [], "rejuv_events": []}` and pass it down both the `--conditional_rejuv` branch
   (`:129`) and the plain `--filter native` branch (`:180`). After the run, in `run.py`: merge
   `unigram_surprisal(word_str)` per word (from `segment_words`), merge each `rejuv_events[t].gate_p`
   into `words[t]["gate_p"]`, attach `config` / `log_marginal` / `min_ess` / `accept_rate` /
   `posterior` (reuse `summarize(sentences, top_k=args.json_topk)`), and `json.dump` to the path.
   Print a one-line "wrote <path>" after the existing human output. Forward-only runs simply have no
   `rejuv_events` and no `gate_p`.

**Scope note — start with the two native paths that matter** (`--conditional_rejuv` and bare
`--filter native`). The post-sweep R2 / `@gen` paths (`--add_delete`, `--rejuvenate`) are
reference/deprecated (see `genjax-native-migration` STRATEGIC PIVOT); wire `record` into them only if
asked. The unified filter (`else` branch) is the old reference and out of scope.

---

## Watch-outs

- **Host sync cost.** Recording decodes `P` prefixes every step ⇒ a device→host pull of `intended_buf`
  / `i_len` per word + `P×W` short decodes. Cheap vs the LM forwards, but it is real host work — keep
  it strictly behind `--output_json` so the default fast path is untouched. (The eval/benchmark do NOT
  set it.)
- **The fused-move signature change touches the perf-critical path.** Returning `accs[nwin]` instead of
  `acc_tot[P]` is a small change but it is the move the perf fix just optimized. Keep the key threading
  byte-identical (RNG-faithful) and re-run `tests/test_rejuv_bridge.py` — the aggregate `accept_rate`
  must be unchanged and detailed balance must still hold. `sum(accs) == old sum(where(gate, acc_tot))`.
- **Top-K residual.** Always emit `prefix_residual_count` so a heavy tail isn't silently dropped; a
  reader can reconstruct total `P` = `sum(topk counts) + residual`.
- **Prefix is recorded POST-rejuv** (end of the `wi` loop, after the hook) — it is the distribution the
  filter actually carries forward, so it reflects rejuvenation's effect at that position. If a fresh
  session wants the pre-rejuv distribution too, capture it before the hook call; default is post.
- **Counts == probabilities only because of resample-every-word** (uniform post-resample weights). If
  ESS_THRESH is ever made finite, the prefix counts and the per-word "posterior" would need the
  particle weights — note this in the JSON (`config`) so the artifact isn't misread later.
- **Multi-token words** are skipped by rejuvenation (not in `win`), so they get no `rejuv_events`
  targets — correct. Their surprisal + prefix are still recorded by the filter loop.
- **Indexing:** `surprisal[wi]` is word `wi`'s evidence; `prefix_topk[wi]` is the distribution AFTER
  word `wi` is processed; `rejuv_events` is keyed by the firing event `t` (= current word index), whose
  window targets earlier words. Keep 0-based throughout and document it in the schema.

## Validation / done criteria

- **LM-independent suite stays green:** `PYTHONPATH=. python -m src.genjax_port.tests.run` (esp.
  `test_rejuv_bridge` — the move signature changed; `accept_rate` and detailed balance unchanged).
- **New unit test** (`tests/test_structured_output.py`, add to `tests/run.py`): run a short sentence
  with a `record`, assert `len(record["words"]) == W`; each `prefix_topk` counts + residual == `P`;
  surprisals finite; with `--conditional_rejuv` on, `rejuv_events` nonempty and every target's
  `attempts == fired_particles * n_sweeps`. Cheap on 70m.
- **Round-trip:** `json.load(open(path))` parses; on `he went too the store` the surprisal on `too`
  is high and the prefix distribution shows the `too/to` ambiguity resolving across positions.
- **Default path unchanged:** a run WITHOUT `--output_json` produces byte-identical console output and
  no perf change (the `record is None` branches are no-ops).

## Pointers index (file:line — verify, they drift)
- filter loop / surprisal / ESS / prefix decode: `smc_substitution.py:194-258` (`:214` step_lmw,
  `:216` ESS, `:251-253` hook, `:255` decode, `:256-257` return_state)
- fused move (return per-word accepts): `rejuv_bridge.py:473` (`_aligned_window_move_fn`, `:494` return)
- aligned hook (gate_p, event log): `rejuv_bridge.py:591` (`_make_aligned_subflip_hook`),
  entry point `:628` (`run_smc_conditional_rejuv_aligned`)
- `custom_sigmoid`: `rejuv_bridge.py:499`; unigram: `src/genjax_port/unigram.py`
- CLI routing + printing: `run.py:122-211` (`:129` rejuv call, `:180` forward-only call, `:208-211`
  summarize/print); `summarize` is in `run.py`
- Gen.jl reference for the per-word reporting (surprisal/unigram/gate printed per word):
  `src/gen_inference.jl:407-410`; argmax-prefix list `:401`
```
