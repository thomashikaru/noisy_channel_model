# Prefix-KV-cache spikes (validated 2026-06-16)

De-risking spikes for the prefix-KV-cache rejuvenation scorer — the PRIMARY next task (see the
`rejuv-prefix-kv-cache-spike` memory and `planning/REJUV_GOAL2_CONDITIONAL_PROPOSAL.md`). Both PASSED;
they're kept here (moved out of `/tmp`, which macOS purges) as the concrete reference for the
integration. Run with the `ncgenjax` env:

    NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python planning/kv_cache_spikes/kv_spike.py
    NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python planning/kv_cache_spikes/kv_vmap_spike.py

- **`kv_spike.py`** — drive + FORK penzai's `KVCachingTransformerLM`: incremental cached logits match
  the uncached full forward (~4.5e-3 float gap), and forking the prefix to score two candidates adds
  zero error. Establishes the functional `unbind_variables`/`bind_variables` pattern and the
  `pad_id=-1` gotcha (NOT 0=EOS).
- **`kv_vmap_spike.py`** — the make-or-break: the stateful cache runs UNDER `jax.vmap` with a
  PER-PARTICLE split point via the REWIND trick (`bound.cache_end_index.value = posc`). Two particles
  flipped at different positions each matched their uncached forward (~3e-3). No K-way cache fork
  needed — rewind reuses the one prefix.
