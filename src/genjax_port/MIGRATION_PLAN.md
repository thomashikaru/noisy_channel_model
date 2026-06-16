# Genjax-native migration plan

> **Decision made (2026-06-14):** migrate the hand-rolled JAX SMC port to **genjax `@gen`
> generative functions** with native SMC + `Rejuvenate`. Two de-risking spikes cleared every
> fundamental risk (see §2). This doc is the build plan; it is written to be picked up cold in a
> fresh session.

---

## 0. How to use this doc (fresh-session agent: read this first)

**Read, in order:** this file → `src/genjax_port/README.md` (what the model does + how to run) →
`src/genjax_port/REJUVENATION_PLAN.md` (the feature this migration unlocks; §0 has the
architecture decision + spike results). The persistent memory index (`MEMORY.md`) points to
`genjax-port-design`, `genjax-port-bpe-substitution`, `genjax-port-cache-trie`,
`genjax-native-migration`, and `genjax-conda-env`.

**Environment** (from `genjax-conda-env` memory — non-obvious, all default Pythons are x86/Rosetta
and won't run jax):
```bash
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh
conda activate ncgenjax          # arm64, python 3.12, genjax + jax 0.5.2, penzai
export TOKENIZERS_PARALLELISM=false
# iterate fast with the small LM:  NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python ...
```
genjax source is editable at `/Users/thomasclark/mit/genjax` — grep it freely.

**Spikes** that prove the approach were at `/tmp/genjax_spike.py` and `/tmp/genjax_spike2.py`
(may be gone in a new session — the essential snippets are embedded in §4 below, so you can
recreate them). Both ran green on Pythia-70m.

---

## 1. Where the project is now (the thing we are migrating FROM)

A **hand-rolled** JAX SMC over the penzai Pythia LM. It works and is validated. Key files in
`src/genjax_port/`:

- `particle_filter_unified.py` — **the current model/filter** (run.py entry). Word-scan SMC: per
  observed word, Phase A lookahead deletion gap + Phase B copy / substitution (SymSpell N:1) /
  insertion. Hand-computes log-evidence, proposal, weights; manual particle buffers
  (`intended_buf [P,M]`, `i_len [P]`, `log_action_prior`). Resamples every word.
- `noise_word.py` — punctuation-aware word segmentation + SymSpell substitution candidates
  (Damerau-Levenshtein, `max_dist` default 2, not hard-capped). `word_sub_loglik(d)=d·log SUB_PARAM`.
- `noise.py` — token-level edit helpers + `SUB_PARAM`, `sub_candidates`.
- `lm_penzai.py` — Pythia (GPT-NeoX) via penzai; `next_token_logits(bufs[P,M], i_lens[P])->[P,V]`;
  `NC_LM`-selectable, default `pythia-410m`; `EOS_ID=0` buffer seed; `vocab_size()`.
- `cache_dedup.py` — exact prefix/row dedup of LM forwards (~1.5–2×; default-on in the lookahead
  filter). **Likely lost under genjax vmap — accept that.**
- `particle_filter.py` / `particle_filter_lookahead.py` — superseded baselines kept for A/B; host
  shared constants (`ACTION_ALPHAS=[3,1,1]`, `MAX_DELETIONS=1`, `P_DELETE_PRIOR=0.02`,
  `P_DELETE_PROPOSAL=0.20`, `LOOKAHEAD_K=6`) the unified filter imports.
- `model.py`, `proposal.py`, `tokenizer.py`, `run.py`.

**Validated behavior (these are the regression targets — see §7):** at P≈64, Pythia-410m:
experimemt→experiment ~100%; deletion "he wants _ go"→"to" ~0.5; doubled-word removal ~0.5;
clean text stays literal; punctuation preserved + period acts as EOS (suppresses the spurious
"medics who treated…" insertion → 96.9% correct). Substitution distance is a real param
(SymSpell), not capped.

---

## 2. Why migrate (decision + spike evidence)

Hand-rolling discarded exactly what genjax is for — `@gen`'s automatic random-choice tracking and
the functional scan interface — and that is precisely what **rejuvenation** needs (an addressable
per-choice trace + MH/SMCP3 machinery). Rather than hand-build a trace (`REJUVENATION_PLAN.md` §5)
and the MH/reversible-jump math (§6/§8), go native and get them for free.

**Spike 1** (`/tmp/genjax_spike.py`): penzai wraps as a genjax `exact_density` distribution inside
a `@gen` `Scan` model; `simulate` runs; **the trace auto-records every choice**; `importance`
weight matches a manual chain-rule exactly; **`vmap` batches the penzai forward** (P=64 ≈ 6× P=1,
not 64×).

**Spike 2** (`/tmp/genjax_spike2.py`): a substitution **noisy channel** (`x~LM; o~channel(x)`)
expresses correctly (fully-constrained `importance` = manual joint, −11.431); **native
`Rejuvenate` performs reanalysis** — a trace started at the literal reading `x0=" too"` flipped to
the higher-posterior `" to"` under an MH-with-Rejuvenate move.

**Conclusion:** every fundamental risk cleared. What remains is migration engineering, not
feasibility.

---

## 3. Target architecture (the thing we are migrating TO)

A genjax `@gen` model + native inference, mirroring the unified filter:

| current hand-rolled piece | genjax-native replacement |
|---|---|
| manual `intended_buf`/`i_len` carry + Python `for t` loop | `@gen` kernel + `kernel.scan(n=W)` (carry = `(buf, i_len)`) |
| penzai `next_token_logits` called manually | `lm_token = exact_density(sample, logpdf)` custom distribution |
| `step_log_evidence` copy/sub/insert | per-step addressed choices: intended `x @ "x"`, action, observed `o @ "o"` |
| `noise_word.word_sub_candidates` (host) | same, passed in as **scanned inputs** (candidate ids + logliks, padded) |
| `proposal.propose` (local posterior) | custom proposal `q` (a `SampleDistribution`) for `Importance(target, q)` |
| manual resample-every-step | genjax SMC (`Importance`/`ImportanceK` + `ChangeTarget`, or a manual SMC loop reusing `model.importance`) |
| **(absent) rejuvenation** | `Rejuvenate` edit requests over scan addresses |
| variable-length deletes | `Scan` + `Mask` (genjax masking) |
| dedup cache | (probably dropped; revisit if perf needs it) |

Granularity: keep the **word-scan** (matches the unified filter and the deterministic word
segmentation). Substitution stays **N:1** (intended word = single BPE token); deletions N:1
(omitted word = single token). M:N is future work either way.

---

## 4. Verified genjax API facts + working snippets (load-bearing — embed/keep)

```python
# --- imports that actually work ---
import genjax, jax, jax.numpy as jnp
from genjax import ChoiceMap as C
from genjax import exact_density                      # else: from genjax._src.generative_functions.distributions.distribution import exact_density
from genjax._src.generative_functions.static import StaticRequest
from genjax.inference.requests import Rejuvenate      # also HMC, SafeHMC
from genjax import Diff                               # Diff.no_change(...)
# SMC: genjax.Target, genjax.inference.smc.Importance / ImportanceK

# --- penzai LM as a custom distribution (single-sequence; vmap batches it) ---
from src.genjax_port import lm_penzai as L
def lm_logp(buf, i_len):                              # buf:[M], i_len: scalar -> [V] log-probs
    return jax.nn.log_softmax(L.next_token_logits(buf[None, :], jnp.asarray([i_len]))[0])
lm_token = exact_density(
    lambda key, buf, il: jax.random.categorical(key, lm_logp(buf, il)).astype(jnp.int32),
    lambda tok, buf, il: lm_logp(buf, il)[tok],
    "lm_token")

# --- @gen Scan model (autoregressive carry) ---
@genjax.gen
def kernel(carry, scanned_in):                        # -> (new_carry, y)
    buf, i_len = carry
    tok = lm_token(buf, i_len) @ "tok"
    return (buf.at[i_len].set(tok.astype(jnp.int32)), i_len + 1), tok
model = kernel.scan(n=T)
init = (jnp.full(M, L.EOS_ID, jnp.int32), jnp.array(1, jnp.int32))
tr = model.simulate(key, (init, None))                # trace auto-records "tok" per step
toks = tr.get_choices()[:, "tok"]                     # vectorized scan-address read

# --- importance with observations (vectorized scan choicemap) ---
vchm = C.empty().at[:, "tok"].set(jnp.asarray(obs_ids, jnp.int32))
tr, w = model.importance(key, vchm, (init, None))     # w = log weight; matches chain-rule

# --- table-channel observation distribution (candidates passed as args) ---
def _obs_logpdf(o, x, cand_x, cand_l):                # P(o|x): lookup x in candidate set
    m = cand_x == x
    return jnp.where(jnp.any(m), jnp.max(jnp.where(m, cand_l, -jnp.inf)), -jnp.inf)
obs_dist = exact_density(lambda key, x, cx, cl: cx[0], _obs_logpdf, "obs")
# candidates(o) -> (ids[K], logliks[K]) padded; loglik = log P(o|x=cand): copy=log(0.95),
# sub=log(0.05)+word_sub_loglik. The intended x is restricted to this set via the proposal.

# --- Rejuvenate MH move on an address (the reanalysis move) ---
cand_prop = exact_density(                             # propose x from candidates ~ LM*channel
    lambda key, buf, il, cx, cl: cx[jax.random.categorical(key, lm_logp(buf,il)[cx]+cl)].astype(jnp.int32),
    lambda x, buf, il, cx, cl: (lambda s: jnp.where((cx==x).any(),
        jnp.max(jnp.where(cx==x, s, -jnp.inf)) - jax.scipy.special.logsumexp(s), -jnp.inf))(lm_logp(buf,il)[cx]+cl),
    "cand_prop")
req = StaticRequest({"x0": Rejuvenate(cand_prop, lambda chm: (buf0, ilen0, cx0, cl0))})  # arg_map may close over context
new_tr, wgt, _, _ = req.edit(key, tr, Diff.no_change(tr.get_args()))   # argdiffs MUST match model args
accept = jnp.log(jax.random.uniform(k)) < wgt                         # MH accept/reject
tr = jax.tree_util.tree_map(lambda a, b: jnp.where(accept, a, b), new_tr, tr)
```

**Gotchas that cost time in the spikes:**
- `req.edit(key, tr, argdiffs)` — argdiffs must match the model's args: use `Diff.no_change(tr.get_args())`, NOT `()`.
- `Rejuvenate`/`StaticRequest` are not top-level: import paths above.
- `exact_density` warns if `name` omitted; pass one.
- For vmap batching, write the LM logpdf/sample on a **single sequence** ([M]); let `vmap` add the
  particle axis (don't pre-batch inside).
- beartype is strict about types (return `int32` tokens, float scores).
- SMC custom proposal: `Importance(target, q)` where `q` is a `SampleDistribution` that takes a
  `Target` and returns a choicemap over the un-constrained latent addresses (GenSP interface,
  `random_weighted`/`estimate_logpdf`). This is the heaviest unbuilt piece — see §6.

---

## 5. Milestones (each ends at a validation gate vs the current filter)

**M0 — Scaffolding (in-repo, not /tmp). ✅ DONE (2026-06-15).** Added `lm_genjax.py` (the
`lm_token` `exact_density` distribution + `lm_logits_single`/`lm_logp` helpers) and
`genjax_model.py` (the `make_lm_scan_model` LM-prior Scan factory + the table-channel `obs_dist`
/ `token_candidates` building blocks from spike 2 — M1 assembles these). Spikes 1–2 are frozen
as `src/genjax_port/tests/` (`test_lm_genjax.py`, `test_noisy_channel.py`) with an aggregate
runner `tests/run.py` (no pytest in `ncgenjax`; run as a script). Gate met: all 5 assertions
green on pythia-70m — importance == LM chain-rule, channel joint == manual, vmap batches,
`Rejuvenate` flips ` too`→` to`. Run: `NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python -m
src.genjax_port.tests.run`.

**M1 — Native substitution model + custom proposal (the core).** `@gen` word-scan model: per word,
`x ~ lm_token @ "x"`, `o ~ obs_dist(x, cand_x, cand_l) @ "o"`; candidates from
`noise_word.word_sub_candidates` passed as scanned inputs. Build the **custom proposal `q`**
(local posterior over candidates) and run SMC (`Importance(target,q)` per step with resampling, or
a manual SMC loop calling `model.importance` with `q`-sampled `x`). Gate: **posterior matches the
idealized behaviors** on the substitution suite (experimemt→experiment dominant, recieve weak,
clean literal) within Monte-Carlo noise — *soft* intuitive targets, NOT bit-parity with the
hand-rolled filter (confirmed with the user 2026-06-15). Golden reference at this seed/P:
`tests/golden_targets.json`.

**M1 ✅ DONE (2026-06-15).** (i) **representation LOCKED** — the per-word `Switch`
(`make_word_model`, §6.3-resolved) with exact branch importances + C-way verified in
`tests/test_word_model.py`. (ii) **SMC driver built + validated** — `smc_substitution.py`
(`run_smc_substitution`): hand-rolled outer loop (§6.1(b)) vmapped over particles, per word
`word_log_evidence` (one LM forward + gather) → local-posterior `propose` → `logsumexp` weight →
resample → emit. `word_log_evidence` is cross-checked == `make_word_model` branch importances
(`tests/test_smc_substitution.py`), so the lean filter and the native model agree by construction.
Behavioral gate met at P=64/410m vs golden: experimemt→experiment 100% (golden 93.8%),
recieve→receive 23.4% weak (17.2%), clean literal 100% (98.4%) — all match the ideals within MC
noise. **Note:** the filtering sweep computes evidence directly (scales to many candidates); the
`@gen` `Switch` model is the trace carrier for M5/R1 rejuvenation, where per-particle traces
(branch index + token choices) are materialized/edited. **Formal writeup:** `docs/model.tex`
(compiles to 10pp) proves SMC proper-weighting + the Rejuvenate/SMCP3 + MH correctness.

**M2 — Deletions. ✅ DONE (2026-06-15).** Ported the lookahead deletion gap (Phase A) into the
native filter as `smc_substitution.deletion_gap`, wired into `run_smc_substitution` behind
`max_deletions` (default 0 = pure-M1, unchanged). Per gap slot: Bernoulli delete? proposed at
`P_DELETE_PROPOSAL` priced at `P_DELETE_PRIOR`; omitted token proposed from top-`LOOKAHEAD_K` LM
reweighted by one-step lookahead toward the word's first token; gap weight folds into the step
weight (a properly-weighted SIS pre-extension — proof in `docs/model.tex` §5, eq. gapweight). Gate
met at P=64/410m: "he wants go home" → "he wants **to** go home" 81% (+to come/have/love ≈97%
reconstruct "to"), minESS 4.0 (high-variance, as expected); M1 cases unchanged. The filtering sweep
ports Phase A directly (vmapped); the genjax `Mask` representation is the trace-carrier concern for
M5. Note: stronger reconstruction than the unified golden (55%) because no INSERT branch competes
yet (M3).

**M3 — Insertion. ✅ DONE (2026-06-15).** The INSERT action (observed word spurious, emit nothing,
scored `log π_ins + n·(−log V)`) is one extra column in `word_log_evidence`, behind
`run_smc_substitution(allow_insertion=True)`; the emission loop already treats a zero-length column
as "emit nothing" (no change). Gate met at P=64/410m: "the boy handed handed the pencil to the girl"
→ doubled "handed" removed 64.1% (golden 46.9%; stronger because no deletion competes), minESS 48.6;
M1/M2 cases unchanged. Doc §5 gives INSERT as an evidence column. **The filtering sweep (sub+del+ins)
is now feature-complete; M4 wires it into `run.py` and M5 adds rejuvenation.**

> **Latency note (measured 2026-06-15, pythia-410m, P=64):** load 3.4s (once/proc); **JIT
> compile ~8s _per distinct tensor shape_** (the `[P,M]` forward, the `[P·K,M]` lookahead batch, and
> a fresh compile for every new `max_intended` ⇒ every new sentence length); warm exec 0.56s/forward,
> ~3.0s/sentence. So a cold one-sentence run ≈13.5s is **~10.5s compile + ~3s exec**. **Compilation
> dominates, not load or exec.** JAX's persistent compilation cache does NOT help (cache-hit 9.1s ≈
> cold-compile 8.1s — the cost is graph tracing/lowering, which the XLA cache doesn't cover; enabling
> it made the cold run far worse). Real levers: (1) **shape stability / bucketing** — pad all
> sentences to a common `max_intended` and run them in ONE process so they share the compiled graph
> (first ≈13s, rest ≈3s each); (2) **keep a warm process** for the dev loop; (3) use pythia-70m for
> the LM-independent tests (already the runner default). A corpus-scale bucketed batch layer is the
> right home for fix (1).

**M4 — Full parity. ✅ DONE (2026-06-15).** `run.py` takes `--filter native|unified` (+
`--max_deletions`, `--no_insertion`); native = `run_smc_substitution` with deletion+insertion on.
Parity harness `tests/capture_native.py` runs the full README suite through the native filter
(all ops, one fixed bucket via `run_smc_batch`) vs `golden_targets.json`. Gate met at P=64/410m —
all six idealized behaviors reproduced: experimemt→experiment 92% (golden 94%), recieve→receive
44% (17% — native more willing, both "partial"), he-wants reconstructs "to" (high-var ESS~5),
doubled-word removed 56% (47%), clean literal 100% (98%), and the **critical medics case:
inflection→infection 89%, period kept as EOS, no spurious "who"** (golden 97%). Quantitative gaps
are within MC noise + the soft-target band (native has no dedup ⇒ different effective sampling).
Runtime: with bucketing the 6-sentence suite is one compile + warm execs (see latency note).

**M5 — Rejuvenation (the payoff).** Per `REJUVENATION_PLAN.md` §11 phasing:
- R1: **substitution-flip** `Rejuvenate` over a word's intended-token address, **unconditional**.
  **✅ DONE (2026-06-15)** — `rejuvenation.py` (`make_chain_model`, `flip_request`, `rejuv_step`,
  `rejuv_sweep`) on an unrolled per-word chain (single-token words); editing `x_k` re-scores the
  suffix via `Update`, so later context votes. `tests/test_rejuvenation.py` (pythia-70m): reanalysis
  flip too→to ✓; the flip's MH weight is ≈0 without the trailing words but ≈+5.5 with them (suffix
  vote) ✓; **detailed balance** — MH histogram == brute-force exact posterior to ≤1e-3 ✓ (Thm 2
  empirically). Gotcha (cost time): a `Rejuvenate` `argument_mapping` sees only the LOCAL sub-trace
  at the edited address, so the position-k context must be rebuilt from the full trace + closed over.
  Spike at `/tmp/genjax_spike4_rejuv.py`. Doc §4.4 updated.
- R2: **add/delete** reversible-jump rejuvenation move. **✅ DONE (standalone, 2026-06-15)** —
  `rejuvenation_r2.py` (`make_gap_chain`, `add_delete_step`/`_sweep`) on a **masked deletion-gap
  chain** (addresses `del{t}`, `gap{t}/xd`, `x{t}`, `o{t}`): each gap is a `genjax.mask(lm_token)`
  whose `del{t}` flip is the trans-dimensional move. **Keystone (spike 6, `/tmp/genjax_spike6_maskflip.py`):**
  `MaskCombinator.edit` supplies the reversible-jump birth/death weight for free — flipping `del{k}`
  via `Update` gives `w_upd` == exact `log p(t')/p(t)` (Bernoulli del-prior + LM(omitted) + automatic
  suffix re-score; Jacobian unity), and delete == −add (round-trips bit-exactly). Unlike R1 it does
  NOT use `Rejuvenate` (forward=add draws a token, backward=delete draws none → asymmetric proposals);
  the SMCP3 weight `W = w_upd + s_bwd − s_fwd` is assembled by hand, proposal = the LM-lookahead
  omitted-token `q` (mirrors `smc_substitution.deletion_gap`). `tests/test_rejuvenation_r2.py`
  (pythia-70m): reanalysis (add recovers omitted "to" → "he wants to go home"), suffix-participates,
  **detailed balance** (add/delete MH histogram == enumerated exact posterior ≤0.07). Design +
  remaining vectorization in `R2_PLAN.md`. **NEXT for R2:** Phase 2 — vmap the move over particles +
  interleave into the sweep (`VECTORIZED_REJUV_PLAN.md`); the masked fixed-address rep keeps the
  batched trace rectangular. *(Multi-token-word add/delete still future, as with R1.)*
- R3: **surprisal-conditioned trigger** + tunable lookback. **✅ DONE for substitution (2026-06-15)**
  — see the bridge below; R3's add/delete needs R2.
Gate (R1): a sentence the filtering sweep gets wrong (early commitment) is fixed by rejuvenation — MET.

**FILTERING-SWEEP BRIDGE ✅ DONE (2026-06-15), VECTORIZED — `rejuv_bridge.py`.** R1 ran on a
standalone chain; the bridge integrates rejuvenation into the real filter, **vmapped over particles**
(the point of the port — Phase 0 spike `/tmp/genjax_spike5_vmap_rejuv.py` proved `Rejuvenate.edit`
vmaps and batches: P=64 = 4.3× P=1, bit-parity with a per-particle loop). Scope v1: single-token
words, substitution-only (homogeneous trace shape ⇒ rectangular batched trace). Out-of-scope
sentences (multi-token words) **degrade gracefully** to the plain substitution filter (no error,
`accept_rate=0`) — rejuvenation is never less capable than `--filter native` (experimemt→experiment
still corrected, it's M1 substitution not deletion). One vectorized
primitive `vmapped_window_move` (jitted, cached per window length) materializes a chain trace per
particle from the sweep's buffers, runs the MH sub-flip, writes back — candidate tables built from the
sweep's evidence so the move targets the sweep's posterior (keystone test). Two modes:
`run_smc_rejuv` (post-sweep / second-pass; full-sentence window) and `run_smc_conditional_rejuv`
(**interleaved**, R3: after each word's resample a per-particle surprisal-gated `[P]` Bernoulli mask
drives a windowed move over the last `lookback` words — `custom_sigmoid` trigger, mirrors
`gen_inference.jl`). Wired in `run.py` (`--rejuvenate` / `--conditional_rejuv --lookback
--logprob_thresh --logprob_spread --rejuv_sweeps`); `run_example_native.sh` demos it on 70m. Tests in
`tests/test_rejuv_bridge.py` (6, in the runner). **Vectorization design + remaining phases in
`VECTORIZED_REJUV_PLAN.md`.** Resample-every-word throughout (settled). Open: Phase 2 = vectorized
trans-dimensional moves (ragged `W` → `Mask`/padding), coupled with R2. Perf TODO: `make_chain_model`
recompiles per window length (bucket it); surprisal gate uses an absolute threshold (no unigram yet).

**M6 — Cleanup.** Retire the hand-rolled filters (or keep one as the A/B baseline); update README +
memory; move the shared constants (`ACTION_ALPHAS`, `MAX_DELETIONS`, `P_DELETE_*`, `LOOKAHEAD_K`)
out of `particle_filter.py`/`particle_filter_lookahead.py` into a `constants`/`config` module so the
port doesn't import from the filter it replaces.

- **✅ DONE (2026-06-15) — dedup re-added to the port's filtering sweep (perf).** `run_smc_substitution`
  now takes `dedup=True` (default-on, like the reference; + optional `dedup_stats`), wiring
  `cache_dedup.make_dedup_fns()` into the `next_logprobs_fn`/`next_logits_fn` seams that
  `word_log_evidence` and `deletion_gap` already route every LM call through; `run.py --no_dedup`
  toggles it off for A/B timing. **Closes the gap and then some:** medics sentence, P=100, 410m,
  cold process each — **dedup 39s vs no-dedup 176s (~4.5×), and faster than the reference 54s**;
  output BIT-IDENTICAL (logP −54.497, posterior 87/13, minESS 21.9 all match). Numerically exact +
  fires verified by `tests/test_smc_substitution.py::test_dedup_matches_plain_and_fires` (LM-dependent;
  in the runner). NB: dedup *does* get hard again if/when the filtering-sweep bridge is closed the
  fully-native way (vmap the `@gen` model) — that's the case the original "dedup lost" concern applies to.

---

## 6. Hard parts / open risks (in priority order)

1. **Custom proposal as a genjax `SampleDistribution` (M1).** `Importance(target, q)` needs `q` to
   implement the GenSP interface (`random_weighted(key, target) -> (log_w, choicemap)` and
   `estimate_logpdf`). For a scan model the proposal is autoregressive (threads the buffer like the
   model). Options: (a) implement a proper GenSP `q`; (b) **simpler fallback — keep a hand-rolled
   SMC outer loop** (Python `for word`) that calls `model.importance` with the `x` proposed by our
   own local-posterior sampler, then resample; this still gets `@gen` traces + `Rejuvenate` while
   sidestepping the GenSP API. Recommend trying (b) first — it preserves the win (traces +
   rejuvenation) with far less API surface, and we keep our proven proposal.
2. **Variable-length deletions via `Scan`+`Mask` (M2).** Read genjax masking
   (`docs/cookbook/inactive/expressivity/masking.ipynb`, `tests/generative_functions/test_mask_combinator.py`).
3. **Word-span N:1 emission inside a scan step** — a word may emit n>1 BPE tokens (copy of a
   multi-token word). **✅ RESOLVED (2026-06-15) via a per-word `Switch`** (spike 3 →
   `genjax_model.make_word_model`): branch 0 = COPY emitting n live `lm_token` choices
   `"t0".."t{n-1}"`, branches 1..S = SUB emitting 1; each branch ends with a deterministic
   channel `factor` ("ch") carrying action-prior + `word_sub_loglik`; branches return the threaded
   `(buf, i_len)` so words compose. Branch importance == manual LM chain-rule + channel exactly
   (copy −21.953, sub −8.626 on `experimemt`), C-way (>2 branch) switch verified
   (`tests/test_word_model.py`). The branch index is the addressable action R1 rejuvenation flips.
4. **`Rejuvenate` over scan-indexed addresses (M5).** Spike 2 did it on a non-scan address; the
   HMC scan test (`tests/inference/test_requests.py:314-359`) rejuvenates scan addresses with a
   `jax.lax.scan` of edits — use that pattern.
5. **Dedup loss** (~1.5–2×) and **per-length recompiles** — measure; reimplement dedup only if needed.

---

## 7. Validation strategy (golden regression)

Before/while migrating, capture the current `particle_filter_unified.py` outputs as **golden
targets** (fixed seed, P, sentences) and assert each native milestone reproduces them within
Monte-Carlo tolerance. Reuse the existing test sentences:
- `the boy did an experimemt today` → experiment (~1.0)
- `did you recieve the message` → receive (weak, ~0.1 at 410m)
- `he wants go home` → "to" (~0.5)
- `the boy handed handed the pencil to the girl` → dup removed (~0.5)
- `the boy did an experiment today` → literal (~1.0)
- `The medics treated the wound to prevent an inflection.` → infection, period kept, no "who"
Use Pythia-70m for fast iteration, confirm key results on 410m. Percentages need P≥200 to be stable
(P=48 is noisy, esp. deletion ESS).

---

## 8. Decisions already made / still open

**Made:** go genjax-native; word-scan granularity; N:1 substitution+deletion (M:N later); MH
accept/reject as default rejuvenation flavor (SMCP3 via `Rejuvenate` weight also available);
rejuvenation phasing R1→R3 (sub-flip+unconditional → add/delete → surprisal+lookback).

**Open:** (1) custom proposal — full GenSP `q` vs hand-rolled SMC outer loop (lean: hand-rolled
outer loop first, §6.1); (2) keep or drop dedup; (3) how much of the old hand-rolled filter to
retire vs keep as baseline; (4) lookback reach vs LM-suffix-rescoring cost.

---

## 9. Fresh-session checklist

1. Activate `ncgenjax`; `PYTHONPATH=. python -c "import genjax, jax; print(jax.default_backend())"` → `cpu`.
2. Re-read §4 snippets; recreate the two spikes if `/tmp` is empty and confirm green (Pythia-70m).
3. Capture golden outputs from `particle_filter_unified.py` (§7).
4. Build M0→M1; the M1 gate (substitution parity) is the real proof the migration is on track.
5. Keep `run_example.sh` / `run.py` pointed at the hand-rolled filter until M4 parity.
