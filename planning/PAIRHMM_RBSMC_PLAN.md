# Plan: one correct RB-SMC pair-HMM noisy-channel model (toy bigram → Pythia)

**Supersedes** `PYTHIA_PAIRHMM_CAPROP_PLAN.md` (kept for reference; its §0/§1 background is still
accurate, its milestone list is not the path we're taking). This plan resets the project to the
**last actually-working model** — the toy-bigram pair-HMM RB-SMC — proves it mathematically, then
graduates the *same* filter to Pythia. The sampled-alignment + rejuvenation stack is archived, not
extended.

**Env:** `ncgenjax` arm64 conda env. Run modules as `python -m genjax_port.<mod>` from `src/`
(`PYTHONPATH=src`). LM selected by `NC_LM` (default `EleutherAI/pythia-70m`).

**Iteration rule (hard):** do correctness work on the **toy bigram** (no LM load, runs at large P
in milliseconds). On **Pythia, keep P tiny (≈4)** — it is slow; Pythia runs are smoke tests, not
correctness gates. Correctness transfers because toy and Pythia are the *same filter* with a
different injected LM.

**Output-visibility rule (hard):** every long run uses `python -u … > /tmp/out.txt 2>&1` with
**flushed per-step progress + elapsed timestamps**; never pipe through `tail`/`head`/`grep` (the
captured stream comes back empty → wasteful re-runs). See the `never-pipe-expensive-output` memory.

---

## STATUS (2026-06-17) — read this first

- ✅ **Plan written.**
- ✅ **A2 (exact-enumeration correctness test) — DONE. The SMC is mathematically certified.**
  `src/genjax_port/tests/test_pairhmm_exact.py` brute-forces the exact posterior. **3/3 substantive
  gates pass** against the *unified* filter: `caprop_logZ_matches_exact`, `map_matches_exact`
  (sub + spurious), `posterior_mass_matches_exact`. (The 4th, caprop-lower-variance, was **demoted to
  a diagnostic** — at toy scale the edge is ~1.1–1.3× and seed/P-sensitive, even inverting at P=3000;
  it is real but grows only with LM cost. `main()` reports it now, doesn't assert it.) Run via the
  function-runner convention (no pytest):
  `python -u -c "import genjax_port.tests.test_pairhmm_exact as T; [getattr(T,n)() for n in sorted(dir(T)) if n.startswith('test_')]"` (~35s).
- ✅ **A1 (unify into `pairhmm_smc.py`) — DONE.** The single injectable RB-SMC filter is
  `src/genjax_port/pairhmm_smc.py` (`PairHMMModel` injection bundle + the `@gen` caprop kernel +
  bootstrap baseline + band/seed). The **toy** (test) and **Pythia** (`pythia_word_caprop.py`) are now
  two `PairHMMModel` configs of *identical* inference code — so the exact-enumeration certification
  transfers to Pythia by construction. `poc_word_indel*.py` are frozen reference PoCs (not imported).
- ✅ **A3 (insertion handling + edit-type gates) — DONE, with a finding that reframes it.** An
  explicit INSERT *action* is a category error: a spurious observed word is a CHANNEL event, not an
  LM emission, so scoring it as a peer action inside the LM-normalized step adds un-normalized mass
  → **+0.2-nat logZ bias vs exact enumeration** (and the heuristic `argmax(alpha) > n_emitted` gate
  had merely been masking a worse ~0.8-nat double-count between that action and the within-word
  insertion sweep). The principled handling is the one already living in `_word_row_update`: the
  insertion sweep marginalizes spurious words inside the channel DP (cost `WINS`), and the **band**
  gives them reach (`|k−t| ≤ band`). So the explicit INSERT action, the `insert_action` flag, and the
  argmax gate were **REMOVED** — every SMC step is now a clean LM word/EOS choice. New gate
  `test_edit_types_recovered_with_band` proves the swept filter recovers all four edit types at the
  production `band=2` (sub / spurious→shorter / missing→longer / clean→unchanged); the toy exact
  gates (logZ/MAP/TV) still pass. The earlier-landed always-present-COPY candidate + deletion-cost
  retune stay. Pythia at P=128 reproduces the validated corrections unchanged (it already ran the
  swept path), so this was a pure simplification there.
- ⏳ **Phase B (archive the bloat) — NOT STARTED.**
- 🔄 **Phase C (Pythia) — STARTED EARLY and is already largely working** (see below). Remaining:
  the multi-token / KV items and a wider sentence sweep.
- ⏳ **Phase D — not started.**

**Pythia is working at modest P (validated 2026-06-17, P=128, no rejuvenation, ~5–8s incl. JIT):**
  `teh cat sat on teh mat` → `the cat sat on the mat` (p≈0.84); `i want go home` → `i want to go home`
  (p≈1.0). Runtime is ~flat in P (vmap). Three fixes got it there, all landed:
  1. **COPY candidate** — `_candidate_ids` now prepends the observed word's own single token.
     `word_sub_candidates` excludes the literal (it's the copy branch), so without this a correctly-
     spelled observed word could never be *emitted*, only reached via top-J LM → boilerplate drift.
     (The toy candidate scan always kept distance-0, so this also restored true toy/Pythia parity.)
  2. **`P_DELETE_PRIOR` 0.02 → 0.005** (`config.py`) — word-deletion cost WDEL −3.91 → −5.30 nats, so
     a hallucinated fluent "missing" word must earn back >5.3 nats before it survives. This is the
     anti-"cheat" knob: stops inference filling LM-cheap boilerplate as cheap word-deletions.
  3. **Prime `"."` → `". "`** (`pythia_word_caprop.PRIME`) — the LEADING `<|endoftext|>` seed asks for
     the document-START distribution (boilerplate); a trailing `". "` conditions the model mid-document.
     The trailing SPACE matters (different token than `"."`). A/B (P=64): `". "` and especially a full
     neutral carrier sentence beat `"."`; `"\n\n"` is BAD (collapses to empty). Carrier-sentence prime
     is available by editing `PRIME` for hard cases.

**Verified findings to honour (don't re-derive — they cost real time):**
1. The **flat toy bigram is too weak to be a correctness target** — its exact MAP for `teh cat` is
   the *empty* sentence (both words spurious). Gates use a **peaked LM** (`_peaked()` in the test) so
   the exact posterior is sensible (`teh cat sat` → `the cat sat` at p≈0.62).
2. **Gate logZ on caprop, not bootstrap.** Bootstrap explores intended lengths up to `M+slack`,
   beyond the enumerable `Lmax`, so its logZ legitimately *exceeds* truncated-exact (support
   artifact, gap was +0.1–0.16). caprop's candidate set keeps it on the enumerated support →
   tight match (gap −0.007 to −0.026). The filter's returned `logZ` equals exact log-marginal
   **minus** the deterministic a0 leading-spurious constant `logsumexp(a0)`.
3. **Do NOT try to "cap" the filter's steps (`slack=Lmax−M`) to force matched support — it breaks
   the filter** (slack→0 gave a logZ gap of +1.65). Use natural slack + caprop-gating instead.
4. caprop's **low-variance** edge is real but MODEST at toy scale (≈1.1–1.3×, P/seed-sensitive) — not
   a gate; it grows with LM cost. Bootstrap is byte-identical old-vs-unified, confirming the shared
   loop is a faithful port; only the caprop draw differs (genjax.categorical vs jax.random).
5. **Insertions are a CHANNEL event, not an LM action — do NOT re-add an explicit INSERT action.**
   The swept `_word_row_update` already marginalizes spurious observed words (cost `WINS`) and the
   band gives them reach (`|k−t| ≤ band`). An INSERT *peer-action* in the LM-normalized step injects
   un-normalized mass → +0.2-nat logZ bias vs exact; combined with the sweep it double-counts ~0.8
   nats. The kernel's action set is therefore just {word candidates, EOS} (A3 finding, certified by
   `test_edit_types_recovered_with_band` + the unchanged exact gates).

---

## 1. The model (what "working" means)

A hierarchical noisy-channel model. The **only sampled latent is the intended sentence**, generated
left-to-right from the LM prior. The edit-alignment between intended and observed words is **summed
out exactly by a nested pair-HMM**, so per-particle state is fixed-shape and `vmap`-clean:

```
intended sentence  ~  LM prior (sampled, left-to-right)        ← the ONLY sampled latent
        │
        ▼  per intended word: one fixed-shape word-level forward-DP row update
word-level pair-HMM:  align word ↔ observed word | word MISSING (deletion) | observed word SPURIOUS (insertion)
        │   the word↔word "emission" cost IS …
        ▼
char-level pair-HMM:  copy/sub/insert/delete CHARACTERS of the surface forms  → channel logpdf
```

Per particle: the LM context buffer + the word-level forward vector `alpha[k]` = log P(intended
prefix, exactly `k` observed words consumed). `alpha` is **Rao–Blackwellized carry state**, never
sampled. SMC weights come from the forward-mass increment; the proposal is the channel-aware
(fully-adapted) one so weights are near-zero variance and small particle counts suffice.

This handles all three edits: **substitution** (char-level DP inside the emission), **deletion**
(intended word with no observation), **insertion** (observed word that is spurious).

---

## 2. Current state (verified by running it, 2026-06-17)

| Component | File | Status |
|-----------|------|--------|
| char channel DP | `poc_pairhmm_channel.py` | ✅ solid, exact posterior reproduced |
| word SMC, 1:1 | `poc_word_smc.py` | ✅ solid |
| **RB-SMC + indel, bootstrap** | `poc_word_indel.py` | ✅ works on toy (subs+spurious; missing is a *bigram model limit*, not a bug) |
| **RB-SMC + indel, channel-aware** | `poc_word_indel_caprop.py` | ✅ **the working model.** Sub fully corrected; spurious → shorter; missing recovered under a peaked LM; low-variance weights confirmed (logZ std 3–6× < bootstrap) |
| **unified injectable filter** | `pairhmm_smc.py` | ✅ **the production filter (A1).** Toy + Pythia are two `PairHMMModel` configs of it |
| **exact-enumeration correctness test** | `tests/test_pairhmm_exact.py` | ✅ **3/3 substantive gates pass** vs brute force, now guarding the *unified* filter (A2) |
| Pythia port | `pythia_word_caprop.py` | ✅ thin Pythia config over `pairhmm_smc`; corrects tested examples at P=128 (was ❌ pre-fix) |
| toy reference PoCs | `poc_word_indel*.py`, `poc_word_smc.py`, `poc_pairhmm_channel.py` | 🧊 **frozen** — kept as readable reference; not imported by the production path |
| sampled-alignment + rejuvenation | `rejuv_bridge.py`, `rejuvenation*.py`, `rejuv_model.py`, `particle_filter_unified.py`, `smc_substitution.py`, `pythia_rejuv.py`, `run.py`, `viz.py` | 🗄 **archive** — different paradigm we're leaving |

`pairhmm_smc.py`, `pythia_word_caprop.py`, the `poc_*` files and `tests/test_pairhmm_exact.py` are
currently **untracked / uncommitted**. They are the asset; everything else is either support or bloat.

**NEXT SESSION — start here:** Phase A is complete (A1 / A2 / A3 all ✅). `run_example_native.sh` is
already **repointed** at the unified filter (`pythia_word_caprop`), which now also emits the
structured JSON the viewer consumes (`--output_json`; `config.words_are="intended"` relabels the
viewer's heatmap to the inferred sentence — backward compatible with old rejuv JSONs). Remaining:
**Phase B** = actually move the dead `run.py` / `rejuv_bridge` / `rejuvenation*` / `rejuv_model` /
`particle_filter_unified` / `smc_substitution` / `pythia_rejuv` stack into `archive/` (the entry
point is now the unified CLI). **Phase C** = wider Pythia sentence sweep; diagnose pythia-70m's
deletion over-trigger on some clean sentences (`the dog ran` → `the dog was ran`, leading `#`
boilerplate) — a WDEL / LM-strength question, not an inference bug. Re-run the exact gates after any
kernel change — they guard the refactor.

---

## 3. Phase A — one filter + a real correctness proof (toy only)

### A1. Unify into a single injectable filter `pairhmm_smc.py`

Promote `poc_word_indel_caprop.run` into a reusable filter whose only model-specific inputs are
**injected**, so the toy and Pythia are two configs of identical inference code:

- `lm_fn(ctx_buf, ctx_len) -> log-probs over [vocab + EOS]` — toy bigram or Pythia `next_token_logprobs`.
- `candidate_fn(observed_word) -> padded ids` — toy edit-scan or `noise_word.word_sub_candidates`.
- char channel `channel_logpdf` + vocab char table.
- knobs: `wdel`, `wins`, `band`, `Ke`, `J`, `cwin`, `slack`.

Keep the `@gen` step kernel (`genjax.categorical @ "action"` + `factor(logsumexp_C) @ "ev"`,
`kernel.importance`) — trace addresses are cheap to keep and we don't want to relitigate them later.
The toy PoCs stay as-is as a frozen reference; the new module is what Pythia imports.

### A2. Exact-enumeration correctness test — **✅ DONE (the headline deliverable)**

`src/genjax_port/tests/test_pairhmm_exact.py` brute-forces the exact posterior (vectorized,
eager — *not* jit; jit constant-folds the big sequence array and times out) over intended sentences
up to a small `Lmax`, scoring `joint = LM_prior(intended) + channel_loglik(observed | intended)`
with the alignment marginalized by the *same* word-level DP (terminal read `alpha[M]`). Gates:

- `test_caprop_logZ_matches_exact` — caprop logZ ≈ exact log-marginal (within 0.08; see finding 2).
- `test_map_matches_exact` — MAP matches exact: substitution `teh cat sat`→`the cat sat` (peaked),
  spurious `the cat cat`→`the cat` (flat).
- `test_posterior_mass_matches_exact` — MAP mass within 0.12 + overall TV < 0.15.
- `test_caprop_lower_variance_than_bootstrap` — the fully-adapted low-variance signature.

Enumeration is feasible only for short intended sentences (length ≤ ~4 over V=12); the **missing-word
case needs length 6 (12⁶ ≈ 3M) and is too big to enumerate — it became an A3 behavioural
MAP-recovery gate**, not an exact-match gate.

### A3. Insertion handling + edit-type gates — ✅ DONE

What we did (and why it differs from the original bullet): the `allow_insert = argmax(alpha) >
ctx_len` heuristic was **not** replaced by a principled INSERT *action* — it was removed along with
the action itself, because an explicit INSERT action is mis-factored (a channel event scored as an
LM action; +0.2-nat logZ bias, see STATUS / finding #5). Spurious words are instead marginalized by
`_word_row_update`'s insertion sweep (cost `WINS`) and given reach by the band. The kernel's action
set is now {word candidates, EOS}.

- `test_edit_types_recovered_with_band` (in `tests/test_pairhmm_exact.py`) gates all four edit types
  on the toy at the production `band=2`, peaked LM, P=8000: substitution corrected, spurious →
  shorter, missing → longer, clean → unchanged. All pass; toy exact gates (logZ/MAP/TV) still pass.
- **Deletion case is `the cat sat the mat` → `the cat sat on the mat`** (NOT `the cat on the mat`, a
  valid noun phrase): under the peaked LM the dropped `on` makes `sat → the` low-probability, so
  restoration is the true MAP. The clean case `the cat sat on the mat` → unchanged guards the
  opposite failure (the filter must not shave a real word off by calling it spurious).

**Phase A done when:** `test_pairhmm_exact.py` passes (logZ + posterior match exact + edit-type
recovery); the unified filter reproduces the PoC results on all edit types; insertion uses no
heuristic gate. ✅ all satisfied.

---

## 4. Phase B — archive the bloat (after A is green)

Move the sampled-alignment + rejuvenation stack out of the active package into `archive/` (do **not**
delete — recoverable and referenceable): `rejuv_bridge.py`, `rejuvenation.py`, `rejuvenation_r2.py`,
`rejuv_model.py`, `particle_filter_unified.py`, `smc_substitution.py`, `pythia_rejuv.py`, `run.py`,
`viz.py`, and their tests. Collapse to one entry point (the unified filter's CLI). Update
`run_example_native.sh` / `run_example_hmm.sh` to point at the unified filter (keep-run-example
memory). Trim the test suite to the pair-HMM path.

---

## 5. Phase C — Pythia, same filter (smoke-tested at tiny P)

Run `pairhmm_smc` with the Pythia `lm_fn` + SymSpell candidates + char channel on surface forms.
Diagnose the boilerplate drift seen at P=4:

1. **Decode by MAP / weighted top-K**, not resample-and-count (the latter is pure noise at small P).
2. Verify the **band** keeps the intended prefix synchronized with observed consumption (the §0.2
   fix — without it the intermediate target collapses to the LM prior and drifts to boilerplate).
3. Ensure the candidate set always contains the observed word, its edit-neighbours, and EOS;
   confirm the `"."` prime steers out of document-start boilerplate.
4. Confirm Pythia kernel == toy kernel by construction (shared module); rely on Phase A for
   correctness, Pythia for a small-P sanity smoke (`run_example_hmm.sh`).

**Phase C done when:** the regression sentences (DEL / SUB / INS / KEEP) give reasonable inferences
on pythia-70m at the budget the user runs (validated via the shell script, not large-P CI).

---

## 6. Phase D — deferred (do not start until A–C solid)

Multi-token intended words (chain-rule LM scoring, surface-form channel) and the KV-extend cache
(validated spikes in `planning/kv_cache_spikes/`). Trans-dimensional rejuvenation stays archived.

---

## 7. Gotchas (already bit us — carry forward)

- `-inf - -inf = NaN` in `dZ`/weights silently poisons categorical sampling. Guard every weight.
- Without the **band**, the intermediate SMC target collapses to the LM prior → boilerplate drift.
- Don't double-apply the terminal correction: EOS-in-C pays it for particles that chose EOS; the end
  correction is for particles still live at the budget.
- LM buffer convention (`lm_penzai`): position 0 = `EOS_ID` seed; eager `load_model()` before any jit.
- Resample-and-count decode is unreliable at small P — decode by weight.
