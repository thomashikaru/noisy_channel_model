# Genjax Noisy-Channel Model (BPE token-level port)

A token-level **noisy-channel model of language comprehension**, implemented as a hand-rolled
Sequential Monte Carlo (SMC) particle filter over a real in-graph language model. Given an
observed (possibly noisy / mistyped / garbled) sentence, it infers a posterior over the
**intended** sentence the speaker meant — correcting typos, reconstructing omitted words, and
dropping spurious ones, all in proportion to how much each edit improves language-model
plausibility.

This is the GPT-2-class successor to the original word-level Gen.jl model, rebuilt on
**[genjax](https://github.com/genjax-dev/genjax) / JAX** with **Pythia** (via
[penzai](https://github.com/google-deepmind/penzai)) as the in-graph LM.

---

## Overview of the model

The generative story: a speaker draws an **intended** sentence from a language model prior, then
a **noisy channel** corrupts it into the **observed** sentence through word-level edit
operations. Inference runs that story backwards — a posterior over intended sentences given the
observation — with SMC.

The filter scans the observed sentence **one word at a time** (segmentation is deterministic, so
all particles stay in lockstep). Per observed word it weighs four explanations, each scored by
the LM and a noise cost:

| operation | meaning | emission |
|-----------|---------|----------|
| **copy** | intended word == observed word | emit the word's *n* BPE tokens (LM chain-rule scored) |
| **substitution** | intended word is a near spelling (a *single* vocab token) | emit 1 token, scored `LM(x) + d·log SUB_PARAM` |
| **insertion** | the observed word is spurious | consume it, emit nothing |
| **deletion** | an intended word was omitted from the observation | reconstruct it before the word (lookahead-guided) |

The key idea that makes typo correction work: a multi-token *implausible* observed word (e.g.
`experimemt` → 4 BPE tokens) loses to a single-token *plausible* neighbor (`experiment` → 1
token) once the LM gain beats the per-edit cost. Edits happen **∝ plausibility gain** — there is
no a-priori "clean vs noisy" label; positing an edit is just an alternative interpretation that
wins when the LM likes it enough.

**Substitution candidates** come from a SymSpell deletion index over the single-token vocabulary,
so the edit distance is a real (Damerau-Levenshtein) parameter, not hard-capped — the likelihood
`SUB_PARAM**d` down-weights far candidates. **Punctuation** is its own unit, which both keeps it
out of substitutions and lets the LM use a sentence-final period as an EOS signal (up-ranking
complete sentences over fragments).

---

## Getting started

### Hardware requirements

- **Apple Silicon (arm64) Mac** for the documented setup below. Inference runs on **CPU** —
  JAX has no usable Metal/GPU backend on macOS (Apple's `jax-metal` is incompatible with the
  pinned `jax==0.5.2`), so the M-series GPU is *not* used. This is fine: the default 410M model
  is ~1.6 GB and runs short sentences in seconds.
- ~2 GB RAM free for the default model (more for larger Pythia sizes).
- On a Linux + CUDA box the same code would use the JAX GPU backend, but the environment setup
  differs and is not covered here.

### Dependencies

- `jax==0.5.2` + matching `jaxlib` (arm64 wheels)
- `genjax` (installed editable)
- `penzai` (in-graph transformer)
- `transformers` + `torch` — **load time only** (to fetch + convert the HF Pythia checkpoint;
  no torch in the inference loop)
- `numpy`, `tqdm`

### Setup (the non-obvious part)

Every pre-installed Python on this machine (`python3`, Homebrew, `pyenv`, `uv`, …) is an
**x86_64 / Rosetta** build, and `jaxlib` 0.5.x has no Intel-mac wheels. You must use the native
**arm64 conda** toolchain. The working environment is the `ncgenjax` conda env under
`miniforge3_arm`:

```bash
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh
conda activate ncgenjax
```

If recreating from scratch:

```bash
conda create -y -n ncgenjax python=3.12
conda activate ncgenjax
pip install -e /Users/thomasclark/mit/genjax     # genjax + jax 0.5.2 (pulls arm64 jaxlib)
pip install penzai transformers torch tqdm
```

The first run downloads the Pythia checkpoint from HuggingFace (~160 MB for 70m, ~1.6 GB for
410m) into `~/.cache/huggingface`; subsequent runs load from cache.

### Quick check

```bash
cd /Users/thomasclark/mit/noisy_channel_model
source /Users/thomasclark/miniforge3_arm/etc/profile.d/conda.sh
conda activate ncgenjax
PYTHONPATH=. python -c "import jax; print(jax.default_backend(), jax.devices())"
# expect: cpu [CpuDevice(id=0)]
```

---

## Running

### Command line

```bash
export TOKENIZERS_PARALLELISM=false
PYTHONPATH=. python -m src.genjax_port.run \
  --sentence "the boy did an experimemt today" \
  --particles 64 --max_dist 2
```

```
observed : the boy did an experimemt today
particles: 64   log P(observed) ~= -45.6   min ESS = 30.0/64

inferred intended sentences (posterior over alternatives):
  96.9%  ( 62)  the boy did an experiment today
   3.1%  (  2)  the bus did an experiment today
```

### Convenience script

`run_example.sh` wraps the above (activates the env, filters noisy warnings, times the run):

```bash
./run_example.sh                # default sentence, 32 particles, edit distance 2
./run_example.sh 200            # 200 particles
./run_example.sh 64 3           # 64 particles, substitution candidates up to distance 3
```

### From Python

```python
import jax
from src.genjax_port.tokenizer import encode
from src.genjax_port.particle_filter_unified import run_particle_filter_unified

obs = jax.numpy.array(encode("The medics treated the wound to prevent an inflection."))
sentences, log_marginal, min_ess = run_particle_filter_unified(
    jax.random.key(0), obs,
    num_particles=64,   # more particles -> lower-variance posterior, linear cost
    max_dist=2,         # substitution candidate edit-distance bound
    progress=True,      # per-word tqdm bar
)
# sentences: list of `num_particles` decoded intended-sentence strings (the posterior sample)
# log_marginal: estimate of log P(observed); min_ess: smallest per-step effective sample size
```

---

## Knobs you can control

| knob | where | default | effect |
|------|-------|---------|--------|
| **particle count** | `num_particles` / `--particles` | 64 | more = lower-variance posterior, ~linear runtime. Use ≥200 for stable percentages; small P is noisy. |
| **base LM** | `NC_LM` env var | `EleutherAI/pythia-410m` | any Pythia size (`pythia-70m/160m/410m/1.4b/…`). A sharper LM gives a steeper plausibility gradient so edits track real corruptions. See benchmark below. |
| **substitution distance** | `max_dist` / `--max_dist` | 2 | candidate-retrieval bound (not a modeling cap — `SUB_PARAM**d` already down-weights far edits). |
| **substitution rate** | `SUB_PARAM` in `noise.py` | 0.1 | per-character edit cost (Beta(2,11) mode in the original). Higher = more willing to correct. |
| **deletion rate** | `P_DELETE_PRIOR` in `particle_filter.py` | 0.02 | a-priori rate of omitted words; the overall edit-rate knob for deletions. |
| **max deletions / gap** | `MAX_DELETIONS` in `particle_filter.py` | 1 | consecutive omitted words allowed per gap (D=1 ≈ 2× faster than D=2). |
| **deletion proposal width** | `LOOKAHEAD_K` in `particle_filter_lookahead.py` | 6 | candidate omitted tokens scored per gap; K=6 ≈ K=12 in quality, ~1.6× faster. |
| **action prior** | `ACTION_ALPHAS` in `particle_filter.py` | `[3,1,1]` | Dirichlet over (copy, sub, insert); copy-favored. |

The LM forwards are **deduplicated by default** (`cache_dedup`): every-step resampling makes the
particle set ~75–90% redundant, so only the unique buffer rows are run through the LM — a
numerically *exact* ~1.5–2× speedup that grows with sentence length.

---

## Model / LM size tradeoff

Benchmarked on a 5-case suite (one per operation) at P=48 on CPU (Apple Silicon). `metric` is
the key behavior fraction for each case (higher = more of that operation); the last row is total
inference time for the suite (model-load excluded).

| case — metric is the fraction shown | pythia-70m | pythia-160m | pythia-410m |
|---|:--:|:--:|:--:|
| substitution, **obvious** typo (`experimemt`→`experiment`) | 1.00 | 1.00 | 1.00 |
| substitution, **borderline** typo (`recieve`→`receive`) | 1.00 | 0.50 | 0.15 |
| deletion — reconstruct omitted "to" † | 0.92 | 0.17 | 0.98 |
| insertion — remove a doubled word | 0.00 | 0.00 | 0.19 |
| clean control — stay literal (no edit) | 0.90 | 1.00 | 1.00 |
| **total inference (5 cases)** | **41 s** | **58 s** | **140 s** |

† Deletion ESS was very low (~1.6) at P=48, so that row is high-variance — trust it least; use
P≥200 for stable deletion numbers.

**Takeaways:**

- **Runtime** scales roughly with model size: 410m is ~3.4× the 70m cost on this suite, but
  still only seconds per short sentence on CPU. 160m is a modest step up from 70m.
- **A bigger LM is more *discerning*, not simply "more corrections."** Every size fixes the
  obvious typo, but on a *borderline* one — `recieve`, which a strong LM reads as a fairly
  plausible `rec`+`ieve` — the correction rate *drops monotonically* with size (70m 1.00 → 160m
  0.50 → 410m 0.15). The stronger the LM, the more edits track genuine implausibility instead of
  firing on anything slightly odd.
- **Larger LMs catch harder edits and over-edit less.** Only 410m removes a doubled word at all;
  70m over-edits clean text (10% edited) and 160m even produced a garbled deletion output
  (`"We want to have"`). Smaller models are both trigger-happy on borderline typos and unreliable
  on the subtler operations.
- **Recommendation:** **410m is the quality sweet spot** — discerning on borderline typos,
  doesn't over-edit clean text, best at insertion — at ~3.4× the 70m runtime but still
  interactive. Drop to `NC_LM=EleutherAI/pythia-70m` for fast iteration, accepting that it
  over-corrects borderline cases and misses insertions/duplicates.

---

## Architecture / file map

```
run.py                       CLI entry point -> run_particle_filter_unified
particle_filter_unified.py   THE model: word-scan SMC (copy/sub/insert/delete)
particle_filter.py           shared constants (ACTION_ALPHAS, P_DELETE_*, MAX_DELETIONS) +
                             token-level baseline filter (A/B reference)
particle_filter_lookahead.py LOOKAHEAD_K + injectable-LM seam + lookahead-deletion baseline
noise.py                     token-level substitution likelihood + edit helpers (SUB_PARAM)
noise_word.py                word segmentation + SymSpell word-substitution candidates
model.py                     per-step joint log-evidence (copy/sub/insert) -> local proposal
proposal.py                  local-posterior proposal + incremental importance weight
cache_dedup.py               prefix/row dedup wrapper around the LM forwards (default on)
lm_penzai.py                 Pythia (GPT-NeoX) loaded via penzai; batched next-token logits
tokenizer.py                 GPT-NeoX BPE tokenizer wrapper (id<->string, surface forms)
```

### Buffer convention

The intended sentence is a fixed-shape int buffer `[max_intended]`. Position 0 is seeded with
`EOS_ID` as start-of-sequence context, so a buffer with `i_len` filled positions has its
next-token distribution at logits position `i_len - 1`. Padded positions hold `EOS_ID`; causal
attention means they never influence earlier positions. Detokenize only at output.

---

## Known limitations / next work

- **N:1 only.** Substitution and deletion both assume the intended word is a *single* BPE token.
  Multi-token intended words (rarer vocabulary that BPE splits) need an M:N extension.
- **Weak-LM-gain cases.** Corrections where the typo is itself fairly plausible under the LM
  (e.g. `recieve`→`receive`) get only modest mass — governed by LM quality and `SUB_PARAM`.
- **Per-length JIT recompiles.** No cross-input padding/bucketing yet; each new sentence length
  recompiles. Fine for one-at-a-time use, a corpus-scale layer is future work.
