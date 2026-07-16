# Why `rejuv=gibbs+bd` is slow (vs `rejuv=off`) — causes, ranked, with mitigations

> ## ⚠️ CORRECTION (2026-07-06) — this report's top recommendation was MEASURED and is WRONG. Read this first.
>
> The report below ranks **Cause 1** (the indel move's whole-sentence rescore) as ~70–80% of the gap and
> calls the **suffix-tail KV rewrite** the "highest-leverage, lowest-risk, exact" fix. **Both claims are
> false.** Measured 2026-07-06 (`planning/bd_kv_probe.py` + `bd_kv_surgical.py`, pythia-70m, CPU):
>
> 1. **The indel move is EXEC-bound, not compile-bound.** Surgical double-call split: COMPILE 3.8 s vs
>    EXEC 28.6 s at P=64. So **Cause 3 (per-run recompile) is ~4 s — negligible, do NOT bother.**
> 2. **The suffix-tail KV rewrite was actually BUILT (exact, gated) and measured ~3.6× SLOWER on CPU**
>    (indel exec 28.6 → 103.5 s; full run 87.5 → 260.9 s). It was reverted. The forward-count cost model
>    below is *wrong for the KV path*: a KV candidate feed over a FULL-length exact tail costs about as much
>    as a full forward, AND penzai's KV-caching transformer has large per-call overhead on CPU (variable
>    bind/unbind, per-candidate prefix-KV copy, cache setup, prefill over a ~2× padded buffer). The
>    substitution sweep's KV win only worked because its tails are WINDOWED to ~5 tokens — the indel move
>    cannot window (it must consider insertions at gap 0, whose tail is the whole sentence).
>
> **Bottom line: there is NO exact CPU speedup for the indel LM forwards.** The plain `seq_token_logprobs`
> whole-sentence forward is kept and is the right choice. Remaining levers are all approximations (fewer
> particles / smaller candidate pool / fewer `bd_attempts`) or cluster-side parallel fan-out across items.
> The mitigations under Causes 1–3 below are retained for the record but are **superseded by this banner.**

Scope: explain the wall-clock gap between `rejuv="off"` and `rejuv="gibbs+bd"` (with the default
`bd_mode="gibbs"` indel move), rank the causes by contribution, and propose a mitigation for each.
**No code was changed** (at the time of the original writing; the KV mitigation was later built + reverted,
see the correction banner). Headline numbers are from `planning/GIBBS_INDEL_RESULTS.md` (~15–30 s/item
for `gibbs`, ~180 s/item for `gibbs+bd`, plus a ~175 s first-item compile); the per-cause breakdown
below is derived from the code paths and a forward-count cost model, and the split between causes was
an **estimate** — now MEASURED (see the banner): exec-bound, and the Cause-1 KV mitigation makes it worse.

Calibration operating point assumed throughout (from `calibration_word_action_smc.py` →
`pythia_word_caprop.run`): pythia-70m, P=256, band=2, channel `align`, `rejuv_dedup=True`,
`bd_mode="gibbs"`, `bd_attempts=1`, `bd_bridge_j=0`, `bd_funcwords=True` (15 closed-class words).

---

## 1. What each setting actually runs

The forward filter is **identical** for all three settings (the loop in `pairhmm_smc.run`,
[pairhmm_smc.py:770-790](../src/genjax_port/pairhmm_smc.py#L770-L790)): M+slack steps, each one batched
LM forward over the cloud + cheap channel-DP scoring. The settings differ only in the rejuvenation work
bolted on:

| setting | forward filter | substitution sweep (per resample event) | indel move (once, post-loop) |
|---|---|---|---|
| `off`       | yes | — | — |
| `gibbs`     | yes | yes | — |
| `gibbs+bd`  | yes | yes | **yes** |

Two facts set the ranking:

- `rejuv="gibbs+bd"` is a **superset** of `gibbs`: it runs the same per-resample substitution sweep
  (`rejuv in ("gibbs","gibbs+bd")`, [pairhmm_smc.py:630](../src/genjax_port/pairhmm_smc.py#L630)) **and
  then** the indel move. So everything that makes `gibbs` slower than `off` is also present, plus the
  indel move on top.
- The indel move is the new, dominant cost: the doc's own numbers put `gibbs` at ~15–30 s and
  `gibbs+bd` at ~180 s, i.e. the indel move alone adds **~150 s/item (~85% of the gap)**.

### The unit of cost: a "full forward"

Every LM score is one penzai transformer forward producing a `[batch, LCTX, vocab=50304]` logit tensor
(`lm_penzai._raw_logits`). That tensor — and the `log_softmax` over the 50304-wide vocab — is the
expensive thing. Counting these forwards is a good proxy for wall time. Let:

- M = observed words; Wmax = M + slack (slack=3); LCTX ≈ seed_len + Wmax tokens.
- P = particles (256); U = **unique** post-resample particles; the post-resample cloud is degenerate,
  measured U/P ≈ 0.02–0.06 ([pairhmm_rejuv.py:1126](../src/genjax_port/pairhmm_rejuv.py#L1126)); Ub = U
  padded up to a fixed bucket rung.
- Kc = indel insertion-pool size = (unique observed surfaces, ≤ M) + (≤15 funcwords) + (bd_bridge_j
  bridges = 0) ≈ **M + ~13** ([pairhmm_smc.py:647-708](../src/genjax_port/pairhmm_smc.py#L647-L708)).
- R = number of ESS-triggered resample events; A = `bd_attempts` (1).

**Forward counts:**

- `off` (filter only): ≈ (M+slack) forwards over the cloud.
- substitution sweep: at each of R events, a **windowed** sweep over the last `rejuv_lookback`(=3) words,
  each candidate scored by the **KV suffix-tail** scorer (prefix prefilled once per word, only the suffix
  tokens fed) over the Ub unique rows — roughly R·3 cheap partial forwards. Already optimized
  ([pairhmm_rejuv.py:490-495](../src/genjax_port/pairhmm_rejuv.py#L490-L495)).
- indel move: `A · (1 + Wmax·Kc + Wmax)` **full-sentence** forwards over Ub. For M=8:
  Wmax=11, Kc≈21 → **≈ 243 full forwards per attempt** ([pairhmm_rejuv.py:1041-1077](../src/genjax_port/pairhmm_rejuv.py#L1041-L1077)).

The indel coefficient (~243·A) is one to two orders of magnitude larger than the filter's ~11 or the
sub-sweep's ~R·3 — and each indel forward is a **whole-sentence** rescore, not a suffix tail. That is the
whole story; everything below is the decomposition.

---

## 2. Causes, ranked by contribution

### Cause 1 — The indel move re-scores the **whole sentence** for every (gap, candidate), with no suffix-tail cancellation  ⟵ dominant (~70–80% of the gap)

> **⚠️ 2026-07-06: the mitigation below (suffix-tail KV rewrite) was BUILT and measured ~3.6× SLOWER on
> CPU — see the correction banner at the top. This cost is real (exec-bound) but is NOT exactly reducible
> via KV on CPU. The whole-sentence forward is kept.**

`gibbs_indel_move` scores the move set `{no-op} ∪ {insert c@gap g} ∪ {delete word i}` by calling the
target `score(...)` once per element ([pairhmm_rejuv.py:1041-1077](../src/genjax_port/pairhmm_rejuv.py#L1041-L1077)).
`score` = `_make_bd_score_fn` → `_lm_logprior` → `model.seq_token_logprobs(bufs)`
([pairhmm_rejuv.py:968-985](../src/genjax_port/pairhmm_rejuv.py#L968-L985),
[pairhmm_rejuv.py:188-192](../src/genjax_port/pairhmm_rejuv.py#L188-L192)), which is **one forward over
the entire candidate sentence**. So each of the ~Wmax·Kc insertion candidates pays a full re-score of all
M-ish words — even though, for a fixed gap g, every candidate shares the identical prefix `tokens[:g]`.

The substitution sweep already solved exactly this: its conditional `q(x) ∝ LM(x|prefix) +
LM(suffix|prefix,x) + channel(x)` cancels the prefix LM (identical across candidates) and scores only the
candidate-dependent **suffix tail** via a KV-cached scorer that prefills the prefix once
([pairhmm_rejuv.py:490-502](../src/genjax_port/pairhmm_rejuv.py#L490-L502), `lm_penzai.batch_tail_logprobs`).
The indel path does not use it — the code says so directly: *"no suffix-tail KV cancellation yet; that is
the Phase-2 perf win"* ([pairhmm_rejuv.py:948](../src/genjax_port/pairhmm_rejuv.py#L948)).

This is the single biggest lever and it is a **debt the codebase flags as deferred** — exactly the
"apply known, already-proven performance wins up front" case in `CLAUDE.md`.

**Mitigation (pure optimization, exact):** route the indel candidate scoring through the same suffix-tail
KV scorer the substitution sweep uses. For an insertion at gap g, prefill `tokens[:g]` once and score the
tail `[candidate word, suffix words w>g, EOS?]`; for a deletion at i, prefill `tokens[:i]` and score the
tail with word i removed. Because only the ratio `logπ(y') − logπ(y)` enters (and the prefix cancels), this
is exact. Expected win: the per-candidate cost drops from O(M tokens) to O(suffix tokens), and the Kc
candidates at a gap share one prefill instead of Kc.

**Mitigation (approximation, larger win):** bound the rescored suffix to a **lookback/lookahead window**
(as the substitution sweep already does with `rejuv_lookback`/`max_tail`,
[pairhmm_smc.py:638](../src/genjax_port/pairhmm_smc.py#L638)). A dropped/duplicated function word changes
the LM mostly within a few words; scoring a fixed window instead of the whole suffix trades far-context
exactness for a constant-bounded tail.

---

### Cause 2 — The candidate **grid is scored sequentially** (`lax.map`), at a tiny batch, with invalid gaps/positions computed then masked  ⟵ ~10–15%

`_indel_logits` builds the grid with nested `jax.lax.map` over Wmax gaps × Kc candidates and over Wmax
deletions ([pairhmm_rejuv.py:1054-1073](../src/genjax_port/pairhmm_rejuv.py#L1054-L1073)). Two costs hide
here:

- **No candidate batching.** The `lax.map` runs the ~243 forwards one after another (this is deliberate:
  *"a full Kc·N batch would materialise a Kc-times-larger vocab-logit tensor"* and **thrashed 32 GB**, so it
  was reverted — [pairhmm_rejuv.py:1049](../src/genjax_port/pairhmm_rejuv.py#L1049),
  `GIBBS_INDEL_RESULTS.md` "Perf / caveats"). Sequential forwards at the small deduped batch Ub (≈16–32
  rows) badly under-utilize the GPU, so wall-time per forward is worse than its FLOPs suggest.
- **Wasted work on invalid slots.** Insertion gaps are scored for all Wmax positions and only then masked
  to −inf for `g > n_words` ([pairhmm_rejuv.py:1064-1066](../src/genjax_port/pairhmm_rejuv.py#L1064-L1066));
  deletions likewise for pad slots ([pairhmm_rejuv.py:1072-1073](../src/genjax_port/pairhmm_rejuv.py#L1072-L1073)).
  With Wmax=11 and n_words≈8, ~20–30% of the grid is computed and thrown away.

**Mitigation (pure optimization, memory-bounded):** the "obvious further perf win" the doc names — a
**chunked batched scorer**: process C candidates per forward (C chosen to fit memory), instead of one
(thrash) or all Kc (OOM). This recovers batch parallelism without the 32 GB blow-up. Combined with Cause 1
(suffix-only tails), the per-forward logit tensor is `[C·Ub, suffix, vocab]` rather than
`[Kc·Ub, LCTX, vocab]`, so a much larger C fits.

**Mitigation (approximation):** prune the grid before scoring. Insertion gaps can be restricted to a band
around the alignment frontier or to gaps where the bigram surprisal spikes (a dropped word shows up as a
locally improbable transition); deletions to in-pool / duplicate / low-LM-contribution words only. Fewer
columns scored, at the cost of possibly missing a restoration far from any surprisal cue.

---

### Cause 3 — The indel sweep is **rebuilt and re-JIT-compiled on every `run()`** (no structural caching, data baked in)  ⟵ potentially large; the most uncertain term — measure it

`make_gibbs_indel_sweep` is **not** memoized and defines its `_logits`/`_apply` as fresh inline `@jax.jit`
functions each call, closing over `score` (which captures `ctx.emit_full`, `ctx.a0`, …) and the concrete
`cand_tok/cand_len/cand_surf` arrays ([pairhmm_rejuv.py:1161-1184](../src/genjax_port/pairhmm_rejuv.py#L1161-L1184)).
Because those per-sentence arrays are **baked into the jaxpr as constants** and the decorated function
object is new every call, two sentences — even of the **same length** — produce different XLA programs and
**recompile**. And the program being compiled is huge (a transformer forward inside a Wmax·Kc `lax.map`),
so each compile is expensive.

This is the **opposite** of what the substitution sweep does. `make_sweep` builds its step through the
`@functools.lru_cache` factories `_build_step` / `_build_dedup_steps`, keyed on the **structural signature**
only, and threads all per-run data (`emit_full`, `a0`, pool, `wdel`, `wins`, `seed_ids`) as **traced args**
— so same-shape runs reuse one compile ([pairhmm_rejuv.py:378-389](../src/genjax_port/pairhmm_rejuv.py#L378-L389),
[pairhmm_rejuv.py:517](../src/genjax_port/pairhmm_rejuv.py#L517)). The docstring there explicitly calls the
per-run recompile *"the fix for the per-run recompile that dominated R3 wall-clock."* The indel move never
got that treatment.

Why "uncertain": `GIBBS_INDEL_RESULTS.md` says ~180 s/item *"once warm"* with the first item paying ~175 s
of compile — which could mean later items mostly skip compile, or could mean each item still eats a large
recompile that the doc lumped into the per-item figure. The code says recompile-per-`run()` should happen;
the magnitude needs a measurement (below).

**Mitigation (pure optimization, exact, proven sibling):** give the indel sweep the same treatment as the
substitution sweep — an `lru_cache`'d factory keyed on `(sl, Wmax, T, M, Kc, …)` that returns a jitted
`_logits`/`_apply`, with `emit_full`, `a0`, the candidate pool, and `theta` costs passed as **traced args**
instead of closure constants. Then a run reuses the compile from any earlier same-length item. This is
directly the lesson in the `reuse-proven-optimizations` memory and the `CLAUDE.md` "reuse proven work"
norm. Pairs naturally with length-bucketed sharding (`SORT_BY_LENGTH`), which groups same-Wmax items so the
cache hits.

---

### Cause 4 — The whole `gibbs` substitution sweep + theta refresh, inherited by `gibbs+bd`  ⟵ ~5–15% (this is the entire `gibbs`-over-`off` gap)

Everything that makes `rejuv="gibbs"` slower than `off` is still in `gibbs+bd`: after **every** resample
event (`ess_pre < 0.5·P`, [pairhmm_smc.py:793](../src/genjax_port/pairhmm_smc.py#L793)) it runs the
windowed substitution sweep ([pairhmm_smc.py:804-820](../src/genjax_port/pairhmm_smc.py#L804-L820)) and,
on the action channels, a Dirichlet theta refresh + `a0p`/`log_alpha` recompute
([pairhmm_smc.py:831-846](../src/genjax_port/pairhmm_smc.py#L831-L846)). This sweep is already the
better-engineered path — windowed, deduped over unique buffers, KV suffix-tail — so it is a modest cost,
but it scales with the number of resample events R and is pure overhead relative to `off`.

**Mitigation (tradeoff):** raise the resampling threshold below 0.5·P or cap the number of sweeps, so the
sub-sweep fires fewer times (fewer R) — trades some cloud diversity for speed. Or shrink `rejuv_lookback`
(3→2) / `rejuv_Ke` (8→smaller) to make each sweep cheaper. The posterior-stability memo found the
substitution rejuv is where particles buy *exactness*, so this is genuinely a quality/speed dial.

---

### Cause 5 — Per-candidate channel-DP recompute and host-side dedup bookkeeping  ⟵ small (<5%)

- Each `score` call also runs `channel_carry`, a forward DP over the M observed words with the band, so the
  channel DP is recomputed ~Wmax·Kc·A times ([pairhmm_rejuv.py:978-979](../src/genjax_port/pairhmm_rejuv.py#L978-L979)).
  It is O(M·band) and fused inside the same jit as the LM forward, so it is dwarfed by the transformer
  forward — but it rides the same grid, so it shrinks automatically once Causes 1–2 cut the grid.
- `_dedup_indel_logits` builds a Python dict over all P particle states (byte-keyed) on the host each sweep
  ([pairhmm_rejuv.py:1135-1158](../src/genjax_port/pairhmm_rejuv.py#L1135-L1158)). O(P) pure-Python work per
  attempt; negligible at P=256 but it grows with P and with A (and the dedup gets **less** effective on
  later attempts, since applying different edits diversifies the cloud, raising U for attempt ≥2).

**Mitigation:** none needed in isolation; both shrink as a side effect of fixing Causes 1–3. If `bd_attempts`
is raised, note the dedup degrades across attempts (Cause 6).

---

### Cause 6 — `bd_attempts` is a straight multiplier

The whole indel grid (Causes 1, 2, 5) re-runs for each of `bd_attempts` sweeps
([pairhmm_rejuv.py:1190-1196](../src/genjax_port/pairhmm_rejuv.py#L1190-L1196)). At the calibration default
A=1 this is neutral, but `GIBBS_INDEL_RESULTS.md` notes ~4–5 attempts are wanted for *full* restoration of
multi-dropped-word sentences — that would 4–5× the dominant cost.

**Mitigation (tradeoff):** keep A=1 as default (one Gibbs move already amplifies a single dropped word
fully); only raise it for sentences plausibly missing ≥2 words. Or make A **adaptive** — stop early once a
sweep returns no-op for (nearly) the whole cloud, which is the common case on clean sentences.

---

## 3. Summary ranking and recommended order of attack

**MEASURED verdict (2026-07-06) overrides the "est. share" and "best mitigation" columns below** — see the
correction banner. Kept for the record; the recommendations in this table are NOT to be followed.

| rank | cause | est. share of the `off→gibbs+bd` gap | best mitigation | type |
|---|---|---|---|---|
| 1 | Indel move re-scores the **whole sentence** per candidate (no suffix-tail KV) | dominant (exec) | ~~reuse the KV suffix-tail scorer~~ **TRIED → ~3.6× SLOWER, REVERTED** (KV overhead on CPU; can't window). No exact fix. | — |
| 2 | Sequential `lax.map` grid at tiny batch + invalid slots scored | ~10–15% | memory-bounded **chunked batched** scorer; prune gaps/deletions (GPU-util issue — minor CPU-only) | pure opt / approx |
| 3 | Indel sweep **recompiled per `run()`** (data baked in, no `lru_cache`) | ~~uncertain~~ **MEASURED ~4 s — negligible** | (not worth fixing) | — |
| 4 | Inherited `gibbs` substitution sweep + theta refresh (per resample event) | ~5–15% | fewer resamples / smaller window & pool | tradeoff |
| 5 | Per-candidate channel DP + host dedup dict | <5% | (cheap) | — |
| 6 | `bd_attempts` multiplier | ×A | keep A=1 / adaptive early-stop | tradeoff |

Global dials that cut the cost (all APPROXIMATIONS / quality trades — the only real levers left, since the
exact Cause-1 fix failed):

- **Fewer particles (P=256→128).** The posterior-stability work found P=128 ≈ P=256 for the align/sub path;
  the filter and sub-sweep scale ~linearly in P, and the deduped indel grid's Ub shrinks with U. Trades
  inference *exactness*, not reachability.
- **Fewer `bd_attempts` / smaller candidate pool.** Direct multipliers on the dominant indel exec cost.
- **Cluster parallel fan-out.** Per-item latency is irreducible exactly; throughput comes from running many
  items concurrently (the `/orcd-cluster` harness), not from a faster single item.
- **LM size is already minimized.** pythia-70m is the floor; 410m is ~5–6× slower and is an explicit
  anti-pattern (`calibration-improvement-antipatterns` memory) — not a mitigation.

**Original recommendation (SUPERSEDED, do not follow):** "the highest-leverage, lowest-risk move is Cause 1
(exact)…". Cause 1's KV mitigation was built and measured ~3.6× slower; Cause 3 is negligible. The lesson:
a forward-count cost model does not capture the KV-caching transformer's per-call overhead — measure before
building. The measurement recipe in §4 is what settled it.

---

## 4. How to measure (to firm up the estimated split, especially Cause 3)

Without editing model code:

- **Per-phase wall clock + recompile check:** run two **same-length** battery items back-to-back under
  `NC_REJUV=gibbs+bd` and time each `run()`. If item 2 is roughly as slow as item 1 (and slower than a
  `gibbs`-only run of the same item), the per-`run()` recompile (Cause 3) is real and large. A third item
  of a *new* length isolates the compile spike.
- **Forward-count instrumentation already exists:** pass `rejuv_stats={...}` to `run`
  ([pairhmm_smc.py:643-646](../src/genjax_port/pairhmm_smc.py#L643-L646),
  [pairhmm_smc.py:823-830](../src/genjax_port/pairhmm_smc.py#L823-L830)) to read `filter_lm_calls`,
  `sweep_prefills`, `uniq_frac`, and the dedup `rows_in`/`rows_computed`. That quantifies Cause 4 and the
  dedup effectiveness (Cause 5) directly. The indel grid count is `A·(1 + Wmax·Kc + Wmax)` with Kc read from
  the assembled pool — log it next to those if a finer split is wanted.
- **Confirm Kc per item:** Kc = unique observed surfaces + (≤15) funcwords; it is the `cs` list length at
  [pairhmm_smc.py:712-713](../src/genjax_port/pairhmm_smc.py#L712-L713).

A two-item timing probe (~a few minutes once the model is loaded) settles the only uncertain row in the
ranking (Cause 3) before committing engineering effort.
