# Code consolidation / de-bloat plan (src/genjax_port)

**Status:** drafted 2026-06-19, **no code changed yet**. Purpose: separate the live pair-HMM RB-SMC path
from the superseded M-series token-filter path and the toy/reference substrate, so a fresh agent can
archive the dead code without breaking the certified path. Read this top-to-bottom before touching anything.

## 0. How the classification was derived (so you can re-check it)
- **Import closure** from the live entry points (`pythia_word_caprop.cli`, `calibration_word_action_smc`,
  and the certification `tests/test_pairhmm_exact`). Edges were extracted by grepping `from genjax_port…import`.
- **git last-touched date** per module: the live pair-HMM generation is dated **2026-06-18**; the dead
  M-series generation is **2026-06-14…16**.
- **Test mapping**: which test exercises which module, and what `tests/run.py` actually runs.
- **This session's validations** (2026-06-19): `test_pairhmm_exact` passes (OFF path bit-identical);
  the word-action ON path + the case-insensitive copy fix were validated on Pythia-70m.

Re-derive the import edges any time with:
`for f in src/genjax_port/*.py; do echo "== $f"; grep -nE "from genjax_port|import genjax_port" "$f"; done`

---

## 1. LIVE CORE — keep (the current pair-HMM RB-SMC noisy-channel model)
The production path. Entry: `pythia_word_caprop.cli` (the `--sentence` CLI; `run_example_native.sh` drives it,
and it is `git --skip-worktree` — do not commit edits to it, see [[keep-run-example-script-current]]).

| module | role | certified/validated by |
|---|---|---|
| `pairhmm_smc.py` | the one unified RB-SMC filter (toy + Pythia share it) | `test_pairhmm_exact` (13 exact-enum gates) |
| `pythia_word_caprop.py` | Pythia config of the filter (channel, candidates, prime, word-action) | `test_pythia_word_caprop` + this session |
| `pairhmm_rejuv.py` | Gibbs/SMCP3 rejuvenation sweep + KV suffix scorer | `test_pairhmm_exact` (rejuv path), R2 tests |
| `lm_penzai.py` | the LM backend (penzai/Pythia), `next_token_logprobs`, KV cache | used by 14 modules |
| `tokenizer.py` | BPE tokenizer surface | — |
| `noise.py` | `insertion_loglik` (+ toy helpers) | used live + tests |
| `noise_word.py` | SymSpell candidate generation (single + multi-token), `segment_words`, `_damerau_levenshtein` | — |
| `unigram.py` | `unigram_surprisal` (frequency-aware insertion cost) | `test_unigram` |
| `config.py` | shared constants | — |
| `cache_dedup.py` | exact post-resample LM-forward dedup | R3 |

**Two live functions are currently MIS-HOUSED in "poc"/old modules — see §2 and §4.**

---

## 1b. Make the WORD-ACTION channel the DEFAULT inference path (the calibration-amenable model)
The deployed inference model must be the **word channel with per-word Dirichlet action-probability latents**,
not either superseded scoring. There are **three** channel concepts in the tree; only the third is
calibration-amenable:

1. **OLD M-series TOKEN-action channel** — `config.ACTION_ALPHAS=[3,1,1]` consumed in `smc_substitution.py`
   / `particle_filter_unified.py`. Token-granular action prior; part of the dead cluster (§3) — archived.
2. **BUNDLED CHAR-COPY channel** — `pairhmm_smc` **OFF path** (`action_alpha=None`); scores
   `copy^matched · sub^changed` at the character level. It **over-certifies**: a clean long word's ~38
   char-copies swamp the evidence and *veto* real-word corrections (the antidote→anecdote failure,
   WORD_ACTION_CHANNEL_PLAN §0). **Not amenable to calibration** — this is the entire reason the word-action
   channel exists.
3. **WORD-ACTION channel** — `action_alpha` set: per-word Dirichlet `(p_copy,p_sub,p_insert,p_delete)`
   latent + the pair-HMM demoted to scoring substitution **form only** (`channel_form_logpdf`, COPY_LP=0).
   Word-granular certification (mild), so it does NOT over-certify. **This is the deployment model**, built
   and validated this effort (incl. the 2026-06-19 case-insensitive `copy_mask` fix — keep it).

**Action — flip the default path:** today `pythia_word_caprop.run` / `--word_action` default to **OFF (#2)**,
with word-action opt-in. Make the word-action channel (#3, `action_alpha=ACTION_ALPHA_DEFAULT`) the **default**
for the production entry points (`pythia_word_caprop.run`, the CLI, and the `run_example_native.sh` invocation
— flag that change to the user since the script is `--skip-worktree`, see [[keep-run-example-script-current]]);
make running the char-copy channel the explicit opt-out.

**Keep #2 as the CERTIFICATION ANCHOR ONLY (do not delete):** `tests/test_pairhmm_exact` proves the SMC/DP
*machinery* is exact (bit-identical to brute-force enumeration) via the OFF path; it is the regression guard
and the concentrated-α limit, **not** a deployment option. State this role explicitly in the `pairhmm_smc.run`
/ `pythia_word_caprop.run` docstrings ("OFF = exact-enumeration certification anchor; the word-action channel
is the deployment model").

**Dependency / do NOT conflate with the calibration constants:** flipping the default *path* is a
code-cleanup step. The deployed **`ACTION_ALPHA_DEFAULT` value and rejuvenation policy are NOT settled** —
`(3,1,1,1)` was found **edit-happy** this session and needs concentration; the final α comes from the
**deferred full-battery re-tune** (to be run now that the leading-opener/`copy_mask` confound is fixed; see
[[word-action-channel-status]]). So: do the flip with the α left as a clearly-marked deployment constant +
a TODO pointing at the re-tune; do not ship `(3,1,1,1)` as the silent deployment default. The `--selftest`
smoke expectations are α-dependent and must be updated to the word-action behavior at the chosen α (so this
step is best finished *after* the battery re-tune fixes α, or with the selftest temporarily pinned to the
OFF anchor + a separate word-action smoke).

---

## 2. EXTRACT-THEN-ARCHIVE blockers (live code trapped in dead/toy modules)
These must be moved BEFORE the §3 archive, or the archive breaks the live path.

1. **`factor`** — `genjax_model.py:118` (`factor = exact_density(…)`, a 1-line genjax combinator). Imported
   live by `pairhmm_smc.py:45` and `pairhmm_rejuv.py:59`. The rest of `genjax_model.py` is **old M-series
   word-scan model code** (`make_word_model`, `token_candidates`, `_make_copy_branch`, …) that depends on
   `lm_genjax` and is used only by the §3 cluster + its tests. **Action:** move `factor` (and the
   `exact_density` it needs) into a tiny core module (e.g. `genjax_factor.py`) or into `pairhmm_smc.py`
   directly; repoint the two live imports. Then `genjax_model.py` + `lm_genjax.py` become archivable.

2. **`_word_row_update`, `_ess`** — `poc_word_indel.py:77,117`. The live word-DP recurrence and ESS,
   imported by `pairhmm_smc.py:46` and `pairhmm_rejuv.py:59`. The rest of `poc_word_indel.py` (BOS/EOS,
   `LOG_BIGRAM`, `lm_logits`, the demo `run`) is **toy-bigram fixture** used by `test_pairhmm_exact`.
   **Action:** move `_word_row_update` + `_ess` into a clean core module (e.g. `word_dp.py`); repoint the
   two live imports. This also severs the production path's transitive import of `poc_pairhmm_channel`
   (`poc_word_indel` imports `channel_logpdf, encode` from it at module load — see §4).

After (1) and (2), the production import closure contains **no `poc_*`, no `genjax_model`, no `lm_genjax`.**
Guard: re-run `python -m genjax_port.tests.test_pairhmm_exact` (must stay bit-identical) after each move.

---

## 3. DEAD — archive (the superseded M-series token-filter + its rejuvenation, dated 06-14…16)
**Confirmed:** no file in the live closure imports any of these (verified by grep over the live core).
They are kept alive only by the stale `tests/run.py` (§5). Superseded by the unified `pairhmm_smc` filter.

| module | what it was | imported by (all dead/test) |
|---|---|---|
| `model.py` | M-series word/obs model | smc_substitution, particle_filter_unified |
| `proposal.py` | data-driven proposal (old) | smc_substitution, particle_filter_unified |
| `lm_genjax.py` | OLD LM backend (`lm_token`, `lm_logp`) — superseded by `lm_penzai` | genjax_model(body), rejuvenation* |
| `genjax_model.py` (BODY) | M-series word-scan @gen model | rejuvenation*, smc_substitution (after `factor` extracted in §2) |
| `smc_substitution.py` | OLD substitution SMC | tests only |
| `particle_filter_unified.py` | OLD "unified" token PF | tests only |
| `rejuvenation.py` | OLD rejuvenation | rejuvenation_r2, rejuv_model |
| `rejuv_model.py` | OLD rejuv model | tests only |
| `rejuvenation_r2.py` | OLD R2 rejuv | tests only |
| `pythia_rejuv.py` | R1 one-off VALIDATION harness (not production; says so in its docstring) | nobody |

**Note:** the WORD_ACTION plan cited `particle_filter_unified.py:78` / `smc_substitution.py:184` as the
*reference* for the Dirichlet `jax.random.dirichlet` draw idiom and for `config.ACTION_ALPHAS` (3-way). That
is a documentation reference, not a runtime dependency — the live word-action draw is already in
`pairhmm_smc.run`. Capture the idiom in a comment if desired, then archive.

Archive **their tests too**: `tests/test_smc_substitution.py`, `test_word_model.py`, `test_rejuvenation.py`,
`test_rejuv_model.py`, `test_rejuvenation_r2.py`, `test_lm_genjax.py`.

---

## 4. TOY / REFERENCE SUBSTRATE — keep, but RENAME out of "poc_" (load-bearing for the certification)
These are **not dead** — `tests/test_pairhmm_exact.py` (the mathematical-correctness gate for the live
filter) is built on them. But they are misnamed "poc" and tangled with the live functions in §2.

| module | provides (used by `test_pairhmm_exact`) |
|---|---|
| `poc_pairhmm_channel.py` | `channel_logpdf`, `encode` (toy char channel; **modified this session** for the FORM variant mirror) |
| `poc_word_smc.py` | `V, VOCAB, VOCAB_IDS, VOCAB_LEN, WORD2IDX`, `edit_channel` (toy vocab) |
| `poc_word_indel.py` (REMAINDER after §2) | `BOS, EOS, LOG_BIGRAM, lm_logits` (toy bigram LM) |
| `poc_word_indel_caprop.py` | `caprop` (toy caprop reference) |

**Action (lower priority, do after §2/§3):** after `_word_row_update`/`_ess` are extracted (§2), rename
these to a clear `toy_*` scheme (e.g. `toy_channel.py`, `toy_vocab.py`, `toy_bigram.py`, `toy_caprop.py`)
or move them under `tests/fixtures/`. Update `test_pairhmm_exact` imports. They are exercised ONLY by that
gate, so the rename is low-risk once §2 severs the production transitive import.

---

## 5. TEST SUITE — reconcile (currently inverted)
`tests/run.py` runs `t1..t9 = {test_lm_genjax, test_noisy_channel, test_word_model, test_smc_substitution,
test_rejuvenation, test_rejuv_model, test_rejuvenation_r2, test_unigram}` — i.e. it runs the **dead** §3
path and **omits the live certification** (`test_pairhmm_exact`) and `test_pythia_word_caprop`.

**Action:** rewrite `tests/run.py` to run the LIVE suite — `test_pairhmm_exact`, `test_pythia_word_caprop`,
`test_unigram`, and `test_noisy_channel` if it still targets live `noise` — and drop the §3 tests along with
their modules. Keep `capture_golden.py` / `capture_native.py` / `golden_targets.json` only if a live test
still consumes them (verify; likely tied to the old path).

---

## 6. CALIBRATION / DIAG SCRIPTS — research artifacts, all **untracked** (`git status ??`), triage separately
These are analysis scripts, not the model. Lower stakes; archive/keep is a judgment call. Dependency chain:
`calibration_identifiability` ← `calibration_marginalize` ← {`calibration_word_action_preview`,
`calibration_word_action_prior_search`}.

- **KEEP (current word-action calibration chain):** `calibration_identifiability.py`,
  `calibration_marginalize.py`, `calibration_word_action_preview.py` (§4 preview),
  `calibration_word_action_prior_search.py` (prior search), `calibration_word_action_smc.py`
  (close-the-loop harness, actively edited this session), `calibration_battery_analyze.py` (NEW analyzer,
  the matched-pair lens), `calibration_gate.py` (P3 edit-gate), `diag_leading_opener.py` (NEW prime/case probe).
- **LIKELY SUPERSEDED (verify, then archive):** `calibration_prior_preview.py` (the CHAR-COPY offline
  preview — superseded by `calibration_word_action_preview`, kept only as the contrast column in the plan);
  `calibration_fit_intuitive.py` (2026-06-18 dial-fitting approach, predates the word-action prior search).
- Decide whether to **commit the keepers** (they're untracked) or keep them as a scratch/ area. Recommend a
  `src/genjax_port/calibration/` subpackage or `scripts/` dir to separate research from the model.

---

## 7. Suggested sequence (safe order) and the guard
1. **Tag a restore point**: `git tag pre-cleanup-archive` (so deletions are retrievable; archiving = delete +
   rely on git history, OR `git mv` into `archive/` if you prefer them visible — recommend the tag+delete for
   a clean tree).
2. **§2 extractions** (`factor`, `_word_row_update`/`_ess`) → repoint live imports → run
   `test_pairhmm_exact` (must stay bit-identical: caprop logZ ≈ −7.955/−7.995/−9.230, TV 0.261/0.259/0.118).
3. **§3 archive** the M-series cluster + its tests → run `test_pairhmm_exact` + `test_pythia_word_caprop`.
4. **§5** rewrite `tests/run.py` to the live suite → run it green.
5. **§4 rename** the toy substrate out of `poc_*` → update `test_pairhmm_exact` imports → run it.
6. **§1b flip the default** to the word-action channel once the deferred battery re-tune has settled the
   deployed α (keep the OFF char-copy path as the certification anchor; update the `--selftest` smoke).
   This step is gated on the calibration re-tune, not on the archive — it can land in a later commit.
7. **§6** triage calibration scripts (separate commit; non-blocking).

**Invariant after every step:** `NC_LM=EleutherAI/pythia-70m PYTHONPATH=src conda run -n ncgenjax python -u
-m genjax_port.tests.test_pairhmm_exact` passes with unchanged values, and a Pythia smoke
(`pythia_word_caprop --selftest`) runs. The OFF path is the bit-identical certified anchor; never regress it.

## 8. Expected outcome
Live core drops from ~34 modules to ~13 (core) + ~4 renamed toy fixtures (under tests/). Archived: ~10
M-series modules + 6 old-path tests. The "poc"/"genjax_model" naming confusion is gone; the test runner
certifies the path that's actually in production; and the **default inference path is the word-action channel**
(Dirichlet action latents), with the char-copy channel retained only as the exact-enumeration certification
anchor. ~2,000+ lines leave the live tree.

## 9. Open verifications for the executing agent (don't assume)
- Confirm `factor`'s only live consumers are `pairhmm_smc` + `pairhmm_rejuv` (grep `import factor`).
- Confirm nothing live imports `genjax_model` beyond `factor`, or `poc_word_indel` beyond `_word_row_update`/`_ess`.
- Confirm `capture_golden`/`golden_targets.json` are old-path only before archiving.
- Confirm `calibration_prior_preview` / `calibration_fit_intuitive` are not imported by a keeper.
- Re-run the full live closure import (`python -c "import genjax_port.pythia_word_caprop"`) after §2/§3.
