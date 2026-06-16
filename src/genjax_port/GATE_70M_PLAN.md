# Plan: default to 70m + unigram-relative surprisal gate

**Status:** ✅ EXECUTED 2026-06-16. Part A (70m default) + Part B (unigram-relative gate via
`wordfreq`, `src/genjax_port/unigram.py`) done; gate kept at Gen.jl thresh=0/spread=1/lookback=2 (user
choice, no further tuning). New `tests/test_unigram.py` (suite 35/35 green on 70m). Behavioral eval
(70m): mean +3.1% (too→to + of→off improve; form→from −20.3% / clean −10.9% still regress, now via the
gate flipping common neighbors got→get/did→had at thresh=0 on the weak LM — accepted as the honest
weak-LM baseline). GOLDEN: kept as a 410m reference (NOT re-baselined; runner has no hard golden
assertions — only the manual capture_native/capture_golden scripts compare, now annotated to pass
NC_LM=410m). Original handoff text preserved below.
**Context to read first:** memory `genjax-native-migration.md` (the BEHAVIORAL EVAL DONE + NEXT
sections) and `genjax-port-settled-decisions.md`; this repo's `src/gen_inference.jl` (the Gen.jl
reference) and `src/gen_lm.jl` (its unigram table). The behavioral eval that motivated this work is
`src/genjax_port/tests/eval_rejuv.py`.

**Why:** the 410m behavioral eval found interleaved sub-rejuvenation is ~neutral on the strong LM
(mean −1.0%), with one −18.8% regression (`form→from`) where the gate fired too freely (96% accept).
Two corrective changes:
1. Make **pythia-70m the default** for all runs — stop treating 410m as the authoritative LM. 70m
   already shows reasonable correctness/behavior and is ~6× cheaper; the user can still override.
2. Make the surprisal gate fire on **contextual surprisal *relative to* unigram surprisal**
   (`surprisal − unigram_surp`), matching the Gen.jl reference, so it fires only on words that are
   *more* surprising in context than out of context (not on words that are just rare).

---

## Part A — make 70m the default LM

**The one functional change:**
- `src/genjax_port/lm_penzai.py:29` — `MODEL_NAME = os.environ.get("NC_LM", "EleutherAI/pythia-410m")`
  → default to `"EleutherAI/pythia-70m"`. Update the comment at lines 25-28 accordingly. All Pythia
  sizes share arch + tokenizer, so nothing else in the model path changes.

**Already correct (no change):** `run_example_native.sh:30` already defaults `NC_LM` to 70m. The
rejuvenation tests (`test_rejuvenation.py`, `test_rejuv_bridge.py`, `test_rejuvenation_r2.py`,
`test_rejuv_model.py`) already run on 70m. The LM-independent tests (importance==manual-joint,
chain-rule, detailed-balance histograms) hold for any LM.

**Docstring / command-string updates (cosmetic, do them so the docs aren't misleading):**
`tests/eval_rejuv.py` header, `tests/test_smc_substitution.py:7-11,113,117,124`,
`tests/capture_native.py:8`, `tests/capture_golden.py:9,31`. Reword "use 410m to match the golden"
once Part A's golden decision below is made.

**THE NUANCE — golden_targets.json is a 410m artifact** (`tests/golden_targets.json:2`,
`"model": "EleutherAI/pythia-410m"`). The behavioral suites that compare against it
(`tests/capture_native.py`, the behavioral half of `tests/test_smc_substitution.py`) will mismatch
once the default is 70m. The golden "ideal" notes even say things like "weak at 410m". Decide one of:
- **(recommended) Re-baseline golden to 70m:** run `NC_LM=EleutherAI/pythia-70m python -m
  src.genjax_port.tests.capture_golden` to regenerate `golden_targets.json`, then **review each
  case** — 70m mangles short words (memory: go→to/home→come on the deletion case), so some idealized
  behaviors (esp. the `recieve→receive` and `he wants go home` deletion case) will NOT reproduce.
  Relax/annotate those `ideal` strings to what 70m actually does, and loosen the corresponding
  assertions in `test_smc_substitution.py`'s behavioral suite. This is the honest "70m is the
  baseline" move.
- **(alternative) Keep golden as a 410m reference** and gate the golden-comparison suites behind an
  explicit `NC_LM=...410m` opt-in (skip with a clear message otherwise). Less work, but then "all
  tests on 70m" isn't literally true for the behavioral suite.

Confirm the intended reading of "all runs" with the eval results in hand before regenerating golden.

---

## Part B — unigram-relative surprisal gate (wordfreq)

**Reference (do this exactly):** Gen.jl computes, per observed word `t` (`gen_inference.jl:407-409`):
```julia
surprisal     = -log_mean_weight                       # contextual surprisal of word t
unigram_surp  = -log(unigram_probs[get_vocab_idx(utt[t])])
cond_rejuv_p  = custom_sigmoid(surprisal - unigram_surp, logprob_thresh, logprob_spread)
```
Its unigram table (`gen_lm.jl:90-92`) is a normalized word-frequency distribution with floors:
unknown words → `min_freq` (⇒ high unigram surprisal ⇒ gate input small/negative ⇒ fire LESS — a
rare word is *expected* to be surprising, so don't over-rejuvenate it); punctuation/EOS → `max_freq`.
Gate defaults (`config.jl:40-48`): **`logprob_thresh=0.0`, `logprob_spread=1.0`, `lookback=2`** —
note thresh=0 because subtracting the unigram baseline recenters the gate input around 0.

**What we already have:**
- Our `surprisal = -step_lmw` (`smc_substitution.py:214`, passed to the hook at line 253) is the
  per-word log-evidence = the exact analog of Gen.jl's `-log_mean_weight`. It is a **scalar per
  word, shared across particles**. So the gate input stays scalar — the change is local and cheap.
- The gate is applied in `custom_sigmoid(surprisal, center, spread)` inside the hooks:
  - `rejuv_bridge.py:564` in `_make_aligned_subflip_hook` (**the production path** —
    `run_smc_conditional_rejuv_aligned`, what `run.py --conditional_rejuv` routes to).
  - `rejuv_bridge.py:492` in `_make_rejuv_hook` (the older `@gen`-window `run_smc_conditional_rejuv`;
    superseded but still tested — update it too for parity, or leave with a note).
- The hook closures already hold `words = NW.segment_words(...)`, a list of `(token_ids, unit_str)`
  where **`unit_str` is the space-stripped surface** (`noise_word.py:37` docstring). So the observed
  word string for index `t` is `words[t][1]` — **no decoding needed**.

**Implementation steps:**
1. **Add `wordfreq` to the env** — it is NOT installed in `ncgenjax`. `pip install wordfreq` in the
   arm64 `ncgenjax` conda env (memory `genjax-conda-env.md`). Record it wherever deps are tracked.
2. **New helper** (suggest `src/genjax_port/unigram.py`, or a small function in `rejuv_bridge.py`):
   `unigram_surprisal(word: str) -> float` = `-log(max(wordfreq.word_frequency(word, "en"), FLOOR))`.
   Mirror Gen.jl's floors: a `FLOOR` (min freq) for unknown/zero-freq words, and treat
   punctuation-only units and empty/EOS as very common (low surprisal) — reuse `noise_word._is_punct`
   to detect punctuation units. Pick `FLOOR` so unknown words land near the corpus min (start from
   wordfreq's smallest representable freq; tune). Cache per string (`functools.lru_cache`).
3. **Precompute per-word at hook build time** (NOT inside vmap — it's per observed word, scalar):
   in `_make_aligned_subflip_hook` (and `_make_rejuv_hook`), build
   `unigram_surp = [unigram_surprisal(surf) for _, surf in words]` once. Then in the hook body change
   `p_fire = custom_sigmoid(surprisal, center, spread)` →
   `p_fire = custom_sigmoid(surprisal - unigram_surp[t], center, spread)` (`t` is the hook's first
   arg = `wi`, the observed-word index).
4. **Re-tune the gate defaults.** The current defaults assume raw surprisal:
   `run.py` `--logprob_thresh` default `5.0` (line ~99), `run_smc_conditional_rejuv_aligned` default
   `logprob_thresh=5.0` (`rejuv_bridge.py:587`), `run_example_native.sh:27` `LOGPROB_THRESH=4.0`.
   With the unigram-relative input, move toward Gen.jl's **`thresh≈0.0`, `spread≈1.0`**, and consider
   `lookback=2` (Gen.jl default) vs our 4. These are defaults to set, then tune empirically in step 6.

**Watch-outs:**
- `surprisal` here is the noisy-channel word *evidence* (mixture over copy/sub/del/ins), not a pure LM
  next-token surprisal — same as Gen.jl's `log_mean_weight`, so the comparison to a *word* unigram is
  apples-to-apples at the word level. Multi-token words: the forward filter handles them and the
  aligned hook already skips them for the *move*; the gate still computes per word — `unit_str` is the
  whole word surface, so `word_frequency` on the full word is correct.
- Tokenizer surfaces carry a leading space; `unit_str` is already space-stripped (good for wordfreq).
- Keep `custom_sigmoid` overflow-safe as-is (`rejuv_bridge.py:471`); the input can now go negative,
  which it already handles.

---

## Validation / done criteria

- **LM-independent tests still green:** `NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python -m
  src.genjax_port.tests.run` (the aggregate runner; ncgenjax has no pytest — run as a script).
- **Re-run the behavioral eval** on the gate change: `tests/eval_rejuv.py` (now default-70m). Target
  outcome: the `form→from` regression shrinks/flips (gate should fire less on `form`, which is a
  fairly common word ⇒ low contextual-vs-unigram surprise), the genuine garden-path `of→off` win is
  retained, clean control unaffected. **Add a unit assertion** that a rare-but-expected word gets a
  LOW gate prob while a common word that is surprising in context gets a HIGH one (locks in the
  unigram-relative semantics; LM-independent if you feed a synthetic `surprisal`).
- If re-baselining golden (Part A recommended path): `capture_native` posteriors match the new 70m
  golden within MC noise.

## Pointers index (file:line)
- LM default: `lm_penzai.py:29` (+comment 25-28)
- surprisal source: `smc_substitution.py:214` (`step_lmw`), passed `smc_substitution.py:253`
- gate call sites: `rejuv_bridge.py:564` (aligned, production), `rejuv_bridge.py:492` (older)
- `custom_sigmoid`: `rejuv_bridge.py:471`
- entry point + defaults: `rejuv_bridge.py:586` (`run_smc_conditional_rejuv_aligned`),
  `run.py:96-101` (CLI), `run_example_native.sh:25-29`
- word strings: `noise_word.py:37` (`segment_words` → `(ids, unit_str)`), `_is_punct` in same file
- Gen.jl gate: `src/gen_inference.jl:407-409`; unigram table `src/gen_lm.jl:90-101`; defaults
  `src/configs/config.jl:40-50`
- golden: `tests/golden_targets.json:2`; regenerate with `tests/capture_golden.py`
- eval: `tests/eval_rejuv.py`
