# archive/

Frozen, superseded code kept for reference only. **Not on any import path** and not run by
the test suite. Do not import from here; if you need something, port it forward.

## genjax_port/ — old sampled-alignment rejuvenation stack

Superseded by the unified RB-SMC pair-HMM filter (`src/genjax_port/pairhmm_smc.py`,
`pythia_word_caprop.py`) and its rejuvenation move (`src/genjax_port/pairhmm_rejuv.py`). The old
paradigm *sampled* the edit alignment with trans-dimensional MCMC; the new one Rao-Blackwellizes it
with a forward DP. See `planning/REJUV_KV_REDESIGN_PLAN.md` §0 and the `pairhmm-channel-reframing`
project memo for why it was retired.

- `rejuv_bridge.py` — the old aligned conditional-rejuvenation runner
  (`run_smc_conditional_rejuv_aligned`) + KV-fork spikes. The overflow-safe gate sigmoid
  `custom_sigmoid` it used to host now lives in `src/genjax_port/unigram.py` (next to the surprisal
  it gates on); that is the only piece still referenced by live code.
- `tests/test_rejuv_bridge.py` — exercised the above (incl. a now-superseded prefix-KV scorer spike;
  the validated KV scorer lives in `src/genjax_port/lm_penzai.py`).
- `tests/test_structured_output.py`, `tests/eval_rejuv.py` — old-path harnesses built on
  `run_smc_conditional_rejuv_aligned`. The live structured-output JSON is emitted by
  `pythia_word_caprop` → `viz.py`.
- `run.py` — the old multi-mode CLI orchestrating `rejuv_bridge` / `smc_substitution` /
  `particle_filter_unified`. Superseded as an entry point by `pythia_word_caprop` (driven by
  `run_example_native.sh`). Imported by nothing live.
