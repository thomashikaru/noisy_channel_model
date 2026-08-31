# Particle-trace JSON schema (the inference step-explorer artifact)

The `--output_json` artifact from `genjax_port.pythia_word_caprop` is a **per-step trace of the SMC
particle cloud**, consumed by the interactive viewer (`genjax_port.viz` → `viz_template.html`). It lets
you scrub the actual inference: the distribution over particle latents at each step, the
alignment-frontier histogram, per-step ESS with resample/rejuvenation markers, and a full per-particle
dump.

**Produced by:** `pairhmm_smc.run(..., trace=[])` appends one snapshot per SMC step (host-side, only
when a `trace` list is passed — the certified math is untouched; `trace=None` is byte-identical to no
trace). `pythia_word_caprop.structured_output(observed, trace, ...)` packages the trace into the JSON
below. `pythia_word_caprop.cli --output_json X` records + writes it.

**Not** the old per-word-surprisal artifact (that recorded MAP word positions, not real SMC steps, with
a single repeated final ESS). This schema supersedes it.

---

## Top level

```jsonc
{
  "observed": "teh cat sat",                 // the input (noisy) sentence
  "config": {
    "lm": "EleutherAI/pythia-70m",
    "particles": 64,                         // P
    "band": 2, "max_dist": 2,
    "rejuv": "gibbs",                        // "off" | "gibbs"
    "lookback": 3,                           // rejuvenation window (words)
    "json_topk": 8,                          // top-K kept in each step's `dist`
    "resample_threshold": 32.0               // = 0.5 * P; ESS below this triggers a resample
  },
  "log_marginal": -23.41,                    // final logZ estimate
  "min_ess": 18.3,                           // min over `ess_series`
  "ess_series": [64.0, 28.8, 50.1, 18.3, …], // ESS at each step (pre-resample) -- the ESS chart
  "resample_steps": [1, 3, 4],               // step `t`s where resampling fired
  "rejuv_steps": [1, 3, 4],                  // step `t`s where a rejuvenation sweep fired
  "steps": [ <step>, … ]                     // the per-step cloud snapshots (see below)
}
```

`ess_series[t] == steps[t].ess`; `resample_steps` / `rejuv_steps` are conveniences derived from
`steps`. The viewer can rebuild them from `steps` alone.

## A step snapshot (`steps[t]`)

One per SMC iteration, recorded **after** that step's extend + any resample + any rejuvenation (the
cloud entering step `t+1`). Weights are `softmax(log_w)` at that point (≈uniform right after a
resample; the proper filtering weights otherwise).

```jsonc
{
  "t": 3,                       // step index (0-based); the final entry has the largest t
  "final": false,               // true ONLY for the extra terminal snapshot (post terminal correction)
  "ess": 18.3,                  // ESS BEFORE this step's resample (the value tested vs resample_threshold)
  "resampled": true,            // did ESS < resample_threshold trigger a resample this step?
  "logZ": -19.02,               // cumulative logZ through this step
  "n_done": 41,                 // particles that have emitted EOS (committed to a full sentence)
  "n_unique": 6,                // number of DISTINCT intended-prefix latents in the cloud

  // EXACT weighted distribution over distinct latents (over ALL P), top json_topk by weight:
  "dist": [
    ["The cat sat", 0.58, 74],  // [decoded intended prefix, summed weight, particle count]
    ["The car sat", 0.22, 28],
    …
  ],
  "dist_residual": [0.09, 12],  // [weight, count] of all latents beyond the top json_topk

  // EXACT alignment-frontier histogram (over ALL P), weighted, sorted by weight desc:
  "frontier": [[2, 0.81], [3, 0.14], [1, 0.05]],
  //            ^k = observed words consumed (argmax of the channel forward carry log_alpha)
  //            ^weight mass of particles at that consumption count -> shows insertion/deletion behavior

  // FULL per-particle dump (the heaviest min(P, 512) particles; the long tail is the residual):
  "particles": [
    {"p": 17, "weight": 0.041, "k": 2, "done": false, "prefix": "the cat sat"},
    …
  ],

  // rejuvenation summary for this step (null when no sweep ran here):
  "rejuv": {
    "words": [1, 4],            // [lo, hi): the word-slot window the sweep revisited
    "ess_after": 39.7,          // ESS after the move's SMCP3 weight was folded in
    "mean_abs_w": 0.0008,       // mean |move_logw| (≈0 for a full-conditional Gibbs move)
    // per swept word slot (word_stats §3.4; present when a trace or word_stats is collected):
    // change_rate = fraction of active particles that moved off their current word this event,
    // stay_prob = mean full-conditional probability of keeping it, n = active particles.
    "sub_words": {"1": {"n": 512, "change_rate": 0.04, "stay_prob": 0.97}, "…": {}}
  }
}
```

### Field provenance (from `pairhmm_smc.run`'s state)

| field | computed from |
|---|---|
| `ess` | `_ess(log_w)` after extend, before resample |
| `dist` / `dist_residual` / `n_unique` | decode `ctx_buf[p][seed_len:ctx_len[p]]` per particle, group by string, sum `softmax(log_w)` |
| `frontier` | `argmax(log_alpha, axis=1)` per particle (the channel pair-HMM consumption count), weighted |
| `particles` | per-particle `{weight, k=frontier, done, prefix}`, capped to the heaviest 512 |
| `rejuv` | the post-resample windowed sweep (R2/R3): window `[lo,hi)`, `ess_after`, `mean|move_logw|`, per-slot `sub_words` (word_stats §3.4) |
| `final` step | one extra snapshot after the **terminal full-consumption correction** — the true posterior |

### Notes / invariants

- `dist`, `dist_residual`, `frontier`, `n_unique`, `n_done` are **exact over all P**. Only `particles`
  is capped (heaviest 512) so the JSON stays bounded at large P; `dist` still accounts for every particle.
- The last `steps` entry (`final: true`) is the terminal-corrected posterior; its `dist` is what
  `decode()` reports as the answer.
- `frontier` k is the alignment latent that is otherwise marginalized/invisible: how many observed
  words the channel pair-HMM has consumed. Spread in k across the cloud = insertion/deletion ambiguity.
- Producing the trace costs ~`(M+slack)` host syncs (decode P prefixes per step); it is **only** built
  when `--output_json` is set, so normal runs pay nothing.
