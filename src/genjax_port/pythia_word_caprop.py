"""Pythia config of the unified pair-HMM RB-SMC filter (Phase 0 / M-A).

This is now a thin model config over :mod:`genjax_port.pairhmm_smc`: it supplies the Pythia-specific
injections (``next_token_logprobs`` as the LM, a char channel over surface forms, SymSpell
candidates, the ``"."`` prime seed) and delegates all inference to the shared filter. The toy bigram
(``tests/test_pairhmm_exact.py``) and Pythia therefore run *identical* inference code -- correctness
proven on the toy by exact enumeration transfers here by construction.

Insertions (spurious observed words) are marginalized inside the channel forward-DP and given reach
by the band; there is no explicit INSERT action (see ``pairhmm_smc._caprop_scores`` for why making
one a peer LM-action biases logZ). Multi-token / KV are Phase D (see ``planning/PAIRHMM_RBSMC_PLAN.md``).

Run:  NC_LM=EleutherAI/pythia-70m python -m genjax_port.pythia_word_caprop --selftest
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from genjax_port import lm_penzai, tokenizer, pairhmm_smc, cache_dedup
from genjax_port.noise_word import (word_sub_candidates, word_sub_candidates_multitoken,
                                     segment_words)
from genjax_port.noise import insertion_loglik
from genjax_port.unigram import unigram_surprisal

EOS_ID = lm_penzai.EOS_ID

LC = 20
CHAR_PAD = 0
ALPHA = 26
CH_COPY = 0.90
CH_INDEL = 0.05
COPY_LP = jnp.log(CH_COPY)
SUB_LP = jnp.log((1.0 - CH_COPY) / ALPHA)
DEL_LP = jnp.log(CH_INDEL)
INS_LP = jnp.log(CH_INDEL)
# Adjacent-transposition (Damerau) edge: a swap of two neighbouring chars costs ONE error event, so
# 'teh'->'the' is distance 1 (one transposition) rather than 2 substitutions. Matched to SUB_LP so a
# swap and a substitution cost the same; this aligns the channel's scoring with SymSpell's candidate
# generation, which already uses Damerau-Levenshtein distance.
TRANSP_LP = SUB_LP

# --- word-action channel: the FORM-only edit costs (planning/WORD_ACTION_CHANNEL_PLAN.md sec 2) -------
# In the word-action redesign the base RATE of editing lives in the per-word Dirichlet action prob
# (p_copy / p_sub), NOT in the character DP. So the demoted pair-HMM scores only the *form* of a
# substitution (which neighbour / how far) with the copy reward REMOVED: matched chars are free
# (COPY_LP=0) and an edited char pays the pure 'which of 26 letters' sharpness SUB_FORM_LP = log(1/26)
# (a transposition is ONE Damerau edit and pays SUB_FORM_LP too -- NOT free). Option (a) of the plan (sec 2): the per-intended-word
# form partition folds into the calibrated p_sub, so the form table need not be surface-normalized.
# SUB_FORM_LP was settled by the prior search (calibration_word_action_prior_search.py): it is
# uncalibratable from the battery (most sub-edits are distance-1 / transpositions) so it is fixed by
# first principles at log(1/26), the principled 'uniform over the alphabet, given an edit occurred'.
SUB_FORM_LP = jnp.log(1.0 / ALPHA)

# Content-neutral "." seed (after the leading <|endoftext|>) marks a sentence boundary without any
# semantics. It must NOT end in a space: candidate words are word-initial tokens (' the'), which
# carry their own single leading space, so a trailing-space prime (the old ". ") tokenizes as
# [".", " "] and collides with that into a DOUBLE space ('.  the'). That malformed context sends the
# correct first word to rank ~31k under the LM and forces the filter to hallucinate a leading word
# to absorb it. "." keeps inter-word spacing consistent with the candidates ('. the').
PRIME = "."

# Word-deletion (missing-word) log-penalty: a missing word -- an intended word with NO observation --
# is how the model adds a word to its reconstruction, so this is the over-editing knob. Too cheap and
# the filter inserts fluent words ('teh cat sat' -> 'The cat CAN BE sat ...'); too steep and it stops
# restoring genuinely-dropped words. Tuned to -9 nats (P~1e-4): curbs over-editing yet still restores
# the 'to' in 'i want go home' (P~0.97). Supersedes config.P_DELETE_PRIOR for this filter; override
# per run via wdel= / --wdel. (Spurious-word insertions are penalized by WINS = insertion_loglik.)
WDEL_DEFAULT = -9.0

# Word-action channel (planning/WORD_ACTION_CHANNEL_PLAN.md): the Dirichlet action prior over
# (copy, sub, insert, delete). The copy concentration was CALIBRATED on the synthetic plausible/implausible
# battery (planning/WORD_ACTION_ALPHA_SWEEP_PLAN.md, 2026-06-21): sweeping alpha_copy with sub=ins=del=1,
# alpha_copy=200 is the knee -- it maximizes literal retention (L=0.91 rejuv=off, 0.99 under rejuv=gibbs)
# and within-pair discrimination at zero over-edit junk, while keeping genuine corrections non-collapsed
# (alpha_copy=500 craters them). Concentrating on copy HELPS editing by killing a spurious-insertion junk
# channel, not by suppressing it. Selected by channel="word_action" (the higher-retention alternative to
# the align default; or an explicit --action_alpha); channel="char_copy" keeps the original bundled char channel (the bit-identical
# exact-enumeration certification anchor). NB the battery is synthetic, NOT the reserved human hold-out.
ACTION_ALPHA_DEFAULT = (200.0, 1.0, 1.0, 1.0)

# Align channel (planning/ALIGN_ACTION_CHANNEL_PLAN.md): the 3-way (align, insert, delete) Dirichlet
# action prior that merges copy+sub into a single "align" action. There is no copy/sub jump -- a copy is
# align at d=0, a near-miss is align at d>=1 -- so the Dirichlet governs only align-vs-insert-vs-delete
# and "how good the match is" lives entirely in the FORM emission: a smooth per-edit distance cost
# ALIGN_SLOPE (= K). channel="align" is now the DEPLOYMENT DEFAULT (promoted 2026-06-21 after the
# calibration below; word_action remains available as the higher-retention alternative).
#
# CALIBRATED 2026-06-21 (planning/ALIGN_OPT_RESULTS.md) on the 40-item plausible/implausible battery to
# jointly maximize correction mass on implausible items (E) and literal retention on plausible items (L):
#   * K = -4.5 (was log(1/26) = -3.26): in the align channel a substitution pays only log(p_align)+K*d
#     (~K*d), with NO per-word sub penalty -- so the old K=-3.26 was ~5 nats CHEAPER per sub than the
#     calibrated word_action operating point (reduction-gate equivalent K=-8.56) and over-edited. K
#     governs ONLY substitutions; sharpening to -4.5 is the knee where real-word-sub corrections peak
#     (SUBW E 0.09->0.82) and sub-driven keep over-edits vanish, before the corrections die (-8.0 reverts
#     to word_action: high L, low E). K is the SUBSTITUTION over-editing knob.
#   * alpha = (200, 2, 2) (was (200,1,1)): the (ins,del)=2 entries are the DELETION/INSERTION knob,
#     independent of K. alpha=2 is a sharp Pareto knee: enough ins/del to fix pythia-70m's passive-voice
#     keep hallucinations ("the boy IS handed...") AND restore dropped function words (DEL-of 0.11->0.70,
#     INS-02a 0->0.95), but not so much that it triggers word-insertion over-edits on clean keeps
#     ("the patient's BODY recovered...", which alpha=4 does). Net vs the old align default, 3-seed mean:
#     E 0.14->0.29, L 0.77->0.93, junk 9/40->~1/40.  (NB battery is synthetic, not the human hold-out.)
ALIGN_ALPHA_DEFAULT = (200.0, 2.0, 2.0)
ALIGN_SLOPE = -4.5   # K: per-edit substitution log-cost in the align FORM table (python float, sweepable)

# Closed-class function words the indel rejuv move (bd_mode="gibbs") may always insert, so a dropped
# function word is restorable even when it is not a local top-J LM bridge (see pairhmm_smc pool part 3).
# Single-token (leading-space) only; the channel treats an inserted intended word as a deletion event.
FUNCWORDS = ("the", "a", "an", "of", "to", "for", "from", "in", "on", "at", "by", "with", "and", "that", "as")
def _funcword_ids():
    ids = []
    for w in FUNCWORDS:
        enc = tokenizer.encode(" " + w)
        if len(enc) == 1:
            ids.append(int(enc[0]))
    return ids


def _char_ids(s):
    s = s.strip().lower()
    ids = [ord(c) for c in s][:LC]
    n = len(ids)
    return ids + [CHAR_PAD] * (LC - n), n


def _channel_dp(observed_ids, intended_ids, n_x, copy_lp, sub_lp, del_lp, ins_lp, transp_lp):
    """Char-level edit-channel logpdf via a forward (sum-product) pair-HMM DP with copy / substitute
    / insert / delete AND adjacent transposition (Damerau). grid[i][j] = log P(observed[:j] | first i
    intended chars). The transposition edge ``grid[i][j] <- grid[i-2][j-2] + transp_lp`` fires when
    the trailing two chars are swapped (intended[i-1]==observed[j-2] and intended[i-2]==observed[j-1])
    -- it carries one extra previous row (grid[i-2]) and the previous intended char in the scan; the
    DP stays fixed-shape and vmap/jit-clean. Case-insensitive (surfaces are lowercased in _char_ids).

    The five edit log-costs are arguments so the same recurrence serves BOTH the bundled base-rate
    channel (:func:`channel_logpdf`, copy_lp=log0.9) AND the base-rate-decoupled FORM channel
    (:func:`channel_form_logpdf`, copy_lp=0) of the word-action redesign."""
    n_o = jnp.sum(observed_ids != CHAR_PAD)
    row0 = jnp.arange(LC + 1, dtype=jnp.float32) * ins_lp
    neg_inf_row = jnp.full(LC + 1, -jnp.inf)                  # grid[-1]: no transposition into row 1
    pad = jnp.zeros((1,), observed_ids.dtype)

    def fill_row(carry, x_char):
        prev2_row, prev_row, prev_x = carry                  # grid[i-2], grid[i-1], intended[i-2]
        cur0 = prev_row[0] + del_lp
        obs_jm2 = jnp.concatenate([pad, observed_ids[:-1]])  # observed[j-2] per output column j
        tp_back = jnp.concatenate([jnp.array([-jnp.inf]), prev2_row[:LC - 1]])  # grid[i-2][j-2]
        can_tr = ((x_char == obs_jm2) & (prev_x == observed_ids)              # last two chars swapped
                  & (observed_ids != CHAR_PAD) & (obs_jm2 != CHAR_PAD) & (prev_x != CHAR_PAD))
        transp = jnp.where(can_tr, tp_back + transp_lp, -jnp.inf)

        def step(left, cols):
            o_char, prev_diag, prev_up, tr = cols
            sub = prev_diag + jnp.where(o_char == x_char, copy_lp, sub_lp)
            dele = prev_up + del_lp
            ins = left + ins_lp
            cell = logsumexp(jnp.stack([sub, dele, ins, tr]))
            return cell, cell

        cols = (observed_ids, prev_row[:-1], prev_row[1:], transp)
        _, rest = jax.lax.scan(step, cur0, cols)
        cur_row = jnp.concatenate([cur0[None], rest])
        return (prev_row, cur_row, x_char), cur_row

    init = (neg_inf_row, row0, jnp.array(CHAR_PAD, observed_ids.dtype))
    _, rows = jax.lax.scan(fill_row, init, intended_ids)
    grid = jnp.concatenate([row0[None], rows])
    return grid[n_x, n_o]


def channel_logpdf(observed_ids, intended_ids, n_x):
    """The BUNDLED base-rate char channel (the original port): matched chars are rewarded COPY_LP and
    edited chars pay SUB_LP, so the score is ``copy^matched . sub^changed . ...`` (base rate * form
    together). This is the OFF / certified path; the word-action ON path uses the FORM variant."""
    return _channel_dp(observed_ids, intended_ids, n_x, COPY_LP, SUB_LP, DEL_LP, INS_LP, TRANSP_LP)


def channel_form_logpdf(observed_ids, intended_ids, n_x):
    """The base-rate-DECOUPLED FORM channel (word-action redesign, plan sec 2 option a): COPY_LP=0
    (matched chars free -- the copy reward is removed; the base rate lives in p_copy now), and each
    edited char pays only the form sharpness SUB_FORM_LP. A transposition is ONE Damerau edit, so it
    costs SUB_FORM_LP too (== one substitution; matching the bundled char channel's TRANSP_LP=SUB_LP and
    the calibrated operating point theta_tr = theta_sub). It is NOT free: a free transposition would make
    co-anagram word pairs (cat/act, form/from, calm/clam) channel-indistinguishable on form. The returned
    score is the pure edit-op form cost ``SUB_FORM_LP^(sub+indel+transp)`` summed over alignments; the
    per-word action cost (log p_copy / log p_sub) is added separately on the emission column."""
    return _channel_dp(observed_ids, intended_ids, n_x, 0.0, SUB_FORM_LP, SUB_FORM_LP, SUB_FORM_LP,
                       SUB_FORM_LP)


def align_form_logpdf(slope):
    """Factory for the ALIGN channel's FORM table (plan ALIGN_ACTION_CHANNEL_PLAN sec 1/3): like
    :func:`channel_form_logpdf` but with the per-edit cost set to the SWEEPABLE slope ``K = slope``
    (matched chars still free, COPY_LP=0) instead of the hardcoded SUB_FORM_LP. A transposition is ONE
    Damerau edit, so it pays the slope K too (== one substitution, NOT free -- see channel_form_logpdf).
    The emission is then a smooth function of (case-insensitive) edit distance, ``K * d``, with no
    copy/sub jump -- the single over-editing knob K decoupled from the action prior. At
    ``slope == SUB_FORM_LP`` it is byte-identical to ``channel_form_logpdf``, so the default align form
    reuses the calibrated word-action form table. ``slope`` is a python float (Phase-4 sweepable)."""
    slope = jnp.float32(slope)
    def _f(observed_ids, intended_ids, n_x):
        return _channel_dp(observed_ids, intended_ids, n_x, 0.0, slope, slope, slope, slope)
    return _f


@functools.lru_cache(maxsize=1)
def _vocab_char_table():
    strs = tokenizer.vocab_strings()
    buf, lens = [], []
    for s in strs:
        ids, n = _char_ids(s)
        buf.append(ids)
        lens.append(n)
    return jnp.array(buf, jnp.int32), jnp.array(lens, jnp.int32)


@functools.lru_cache(maxsize=1)
def _word_token_mask():
    """(emit_vocab,) bool mask: True where the token is a WHOLE word -- word-initial (leading space,
    GPT-NeoX BPE word boundary) and alphabetic (' cat', ' The' -> True; '\\n', '#', '****', bare
    numbers/punctuation -> False; AND mid-word fragments 'xt', 'ing', 'ed' -> False, since a dropped
    intended word is a whole word, not a fragment). Restricts the proposal's top-J LM bridge pool to
    real words so the prior cannot hallucinate non-word / sub-word document-start boilerplate as
    intended/missing words (the sentence-initial '#'/'xtw' that pythia-70m emits after the seed)."""
    return jnp.array([s[:1] == " " and s[1:].isalpha() for s in tokenizer.vocab_strings()],
                     dtype=bool)


def _obs_word_units(observed):
    obs_ids = tokenizer.encode(observed.strip())
    return [unit_str for _ids, unit_str in segment_words(obs_ids)]


def _obs_word_spans(observed):
    """The ACTUAL observed token span of each unit (parallel to :func:`_obs_word_units`). These are
    what ``_candidate_words`` uses to build a faithful COPY, so a unit is reconstructed byte-for-byte
    as observed -- a terminal '.' stays the attached token 15 (not the spaced ' .'=964), an opening
    quote / '$' keeps the leading space it actually had, and case is preserved."""
    obs_ids = tokenizer.encode(observed.strip())
    return [tuple(int(t) for t in ids) for ids, _unit_str in segment_words(obs_ids)]


def _candidate_words(word, obs_span, max_dist, Ke):
    """Candidate intended words for an observed word, each a ``(token-span tuple, surface str)``.

    The COPY comes FIRST so a correctly-spelled word can always be emitted verbatim, and it is now the
    word's **actually observed token span** (``obs_span``) -- NOT a re-encode of ``" " + body``. The
    old re-encode forced a leading space onto every unit, which is right for a mid-sentence word but
    wrong for punctuation: a terminal '.' was observed as the attached token 15 but copied as the
    spaced ' .' (964), so the "faithful" copy detached the period (and lowercased the first word).
    Whether a unit carries a leading space is context-dependent (an opening quote / '$' that followed
    a space does, a sentence-final '.' does not), and only the observed span records it -- hence using
    it verbatim. The one exception: a sentence-initial WORD has no leading space in the observed stream,
    but the "." prime severs it from one, so we restore the word-initial space for the LM (the decoder
    strips it anyway). ``obs_span=None`` (non-Pythia callers, e.g. the toy) falls back to the re-encode.

    Then single-token SymSpell substitution neighbours (case-folded query; ``word_sub_candidates``
    excludes the literal -- it is the COPY) and multi-token neighbours from the wordfreq dictionary
    (D2). Deduped by span, copy-first, capped to Ke. A single-token candidate is ``((tid,), surf)`` and
    keeps ``surface_id == tid`` downstream (the certified single-token path); a multi-token COPY is
    ``((t0,t1,...), surf)`` (e.g. a >=2-token correct word -- rarer words, names, morphology)."""
    body = word.strip()
    sub_body = body.lower()                                       # SymSpell matching stays case-folded
    if obs_span is None:                                          # toy / non-Pythia: re-encode fallback
        lit = tuple(tokenizer.encode(" " + sub_body))
    else:
        lit = tuple(int(t) for t in obs_span)                    # faithful COPY: the observed span itself
        if body and body[0].isalpha() and not tokenizer.surface(lit[0]).startswith(" "):
            lit = tuple(tokenizer.encode(" " + body))            # sentence-initial word: restore the space
    cands = [(lit, body, 0)]                                      # COPY, distance 0 (kept first)
    for tid, d in word_sub_candidates(sub_body, max_dist=max_dist):
        cands.append(((tid,), tokenizer.surface(tid).strip(), d))   # single-token neighbours
    for span, surf, d in word_sub_candidates_multitoken(sub_body, max_dist=max_dist):
        cands.append((span, surf, d))                            # multi-token neighbours (D2)
    cands.sort(key=lambda t: t[2])                               # COPY first, then nearest by distance
    seen, dedup = set(), []
    for span, surf, _d in cands:
        if span and span not in seen:
            seen.add(span)
            dedup.append((span, surf))
        if len(dedup) >= Ke:
            break
    return dedup


@functools.lru_cache(maxsize=8)
def _pythia_model(prime, lm_logprobs_fn=None, use_word_mask=False, dedup=False, align_slope=None):
    """Build the Pythia :class:`pairhmm_smc.PairHMMModel`. Cached per prime so the vocab char table
    + seed are reused across runs. ``lm_logprobs_fn`` defaults to the loaded penzai model.

    ``align_slope`` (default ``None`` -> ``ALIGN_SLOPE``) is the per-edit cost K of the align channel's
    FORM table; it is part of the cache key so a Phase-4 K-sweep gets a distinct cached model per slope
    without disturbing the default-keyed (``None``) word_action / char_copy model.

    ``use_word_mask`` (default OFF) restricts the proposal's top-J LM pool to whole-word tokens
    (:func:`_word_token_mask`). It was added to stop sentence-initial non-word tokens ('#'), but the
    real cause of those was the double-space prime (fixed: PRIME has no trailing space), and the mask
    in fact worsens mid-sentence over-editing -- so it is off by default. Kept as an opt-in knob.

    ``dedup`` (R3 item 1) wraps the filter's per-step forward in :func:`cache_dedup.make_forward_dedup`
    so it runs only on the unique post-resample prefixes (the cloud is ~93% duplicates after resample-
    every-word) and scatters back -- EXACT (bit-identical posterior given the same RNG), removes
    redundant LM forwards. It wraps only ``lm_fn``; the KV ``tail_fn`` (rejuv sweep scorer) is left
    intact and deduped separately (sweep phase)."""
    vocab_char, vocab_clen = _vocab_char_table()
    lm_fn = lm_logprobs_fn or lm_penzai.next_token_logprobs
    if dedup:
        lm_fn = cache_dedup.make_forward_dedup(lm_fn)
    seed_ids = [EOS_ID] + (tokenizer.encode(prime) if prime else [])
    # R3: the rejuvenation sweep scores only the candidate-dependent suffix tail via the KV-cached
    # scorer (prefills the prefix once, shares it across candidates). Default penzai LM only (a custom
    # lm_logprobs_fn has no KV path -> uncached fallback).
    tail_fn = None if lm_logprobs_fn else (
        lambda cb, cl, t, tl: lm_penzai.batch_tail_logprobs(cb, cl, t, tl, use_kv=True))
    return pairhmm_smc.PairHMMModel(
        lm_fn=lm_fn, eos_id=EOS_ID, emit_vocab=vocab_char.shape[0],
        vocab_char=vocab_char, vocab_clen=vocab_clen, channel_logpdf=channel_logpdf,
        channel_form=channel_form_logpdf,   # word-action FORM channel (COPY_LP=0); used iff action_alpha set
        channel_form_align=align_form_logpdf(ALIGN_SLOPE if align_slope is None else align_slope),
        char_ids=_char_ids, candidate_words=_candidate_words, obs_words=_obs_word_units,
        obs_spans=_obs_word_spans,
        decode_ids=lambda t: tokenizer.decode(t).strip(), tail_logprobs=tail_fn,
        seq_token_logprobs=(None if lm_logprobs_fn else lm_penzai.seq_token_logprobs),
        seed_ids=tuple(seed_ids), word_mask=_word_token_mask() if use_word_mask else None)


def run(observed, key, P=64, wdel=None, wins=None, slack=3, band=2,
        max_dist=2, Ke=12, J=8, cwin=1, prime=PRIME, lm_logprobs_fn=None, use_word_mask=False,
        rejuv="off", rejuv_lookback=3, rejuv_Ke=8, rejuv_stats=None, trace=None, dedup=False,
        lm_temp=1.0, ins_rate=0.02, uniform_ins=False, action_alpha=None, channel=None,
        align_slope=None, bd_bridge_j=0, bd_pool_cap=None, bd_p_stay=0.0, bd_mode="gibbs", bd_attempts=1,
        bd_funcwords=True):
    """Channel-aware RB-SMC on Pythia via the shared filter. Returns (state, log_w, logZ, seed_len).

    ``channel`` picks the noise model: ``"word_action"`` (the deployment model -- per-word 4-way Dirichlet
    action latents, concentration ``action_alpha`` defaulting to ``ACTION_ALPHA_DEFAULT``), ``"align"``
    (the 3-way copy+sub-merged action channel, plan ALIGN_ACTION_CHANNEL_PLAN; ``action_alpha`` defaults
    to ``ALIGN_ALPHA_DEFAULT`` and the form per-edit cost K to ``align_slope``/``ALIGN_SLOPE``), or
    ``"char_copy"`` (the deprecated bundled char channel, kept as the exact-enumeration certification
    anchor + opt-out). ``channel=None`` infers it from ``action_alpha`` (back-compat for the retired
    ON/OFF boolean).

    ``wdel`` is the missing-word (over-editing) log-penalty (default ``WDEL_DEFAULT``).

    **Spurious-word (insertion) cost.** By default it is FREQUENCY-AWARE: the cost of explaining an
    observed word as a spurious insertion is ``log(ins_rate) - unigram_surprisal(word)`` -- a per-word
    decomposition into an insertion RATE ``ins_rate`` (how often any spurious word occurs) times the
    out-of-context unigram content distribution (what word it is). This replaces the old flat
    ``-log(vocab)`` floor, under which any below-uniform-frequency word (e.g. "lollipop") was CHEAPER to
    drop as an insertion than to keep as a genuine -- but improbable -- LM sample, so rare correct words
    were laundered away. Out-of-context unigram is the principled content model (a slip is not predicted
    by the discourse). Escape hatches: ``uniform_ins=True`` restores the flat ``-log(vocab)``; an explicit
    scalar ``wins=`` overrides with a uniform value. ``use_word_mask`` opt-in (see _pythia_model).

    ``lm_temp`` (lambda) tempers the LM PRIOR: the posterior is ``P_LM^lm_temp * P_channel`` (applied in
    both the caprop step and the rejuvenation move). ``1.0`` = untempered; ``< 1`` (e.g. 0.5) flattens
    pythia's over-confident word preferences so plausible/grammatical inputs are read more literally,
    curbing the over-editing of clean sentences (it scales up the LM gap an edit must clear by 1/lm_temp).

    ``rejuv="gibbs"`` enables the flag-gated post-resample Gibbs/SMCP3 rejuvenation sweep (R2): a
    windowed (last ``rejuv_lookback`` words) full-conditional resample over a per-slot SymSpell pool
    (``rejuv_Ke`` candidates). ``rejuv_stats`` (dict) collects the cost/degeneracy counters.

    ``dedup=True`` (R3 item 1) dedups the LM forwards over the degenerate post-resample cloud (exact;
    bit-identical posterior given the same RNG): the filter's per-step forward (1a, :func:`_pythia_model`)
    AND the rejuv sweep's tail scorer (1b, via ``rejuv_dedup``). The sweep is the dominant single-sentence
    cost (its prefills scale ~linearly with P), so 1b is the main wall-clock win."""
    from genjax_port import pairhmm_rejuv as RJ
    # Channel selector (plan WORD_ACTION_REJUV_PLAN Phase 3): ``"word_action"`` is the model;
    # ``"char_copy"`` is the deprecated bundled char channel, kept as the exact-enumeration certification
    # anchor (``test_pairhmm_exact``) + opt-out. ``channel=None`` infers it from ``action_alpha``
    # (back-compat); word_action needs a concentration, so default it to ``ACTION_ALPHA_DEFAULT`` (NB the
    # deployed alpha is pending the calibration re-tune, planning/WORD_ACTION_ALPHA_SWEEP_PLAN.md).
    if channel is None:
        channel = "word_action" if action_alpha is not None else "char_copy"
    if channel == "word_action" and action_alpha is None:
        action_alpha = ACTION_ALPHA_DEFAULT
    elif channel == "align" and action_alpha is None:        # 3-way (align,ins,del) default
        action_alpha = ALIGN_ALPHA_DEFAULT
    elif channel == "char_copy":
        action_alpha = None
    if lm_logprobs_fn is None:
        lm_penzai.load_model()
    model = _pythia_model(prime, lm_logprobs_fn, use_word_mask, dedup, align_slope)
    ntok = model.emit_vocab
    obs_words = model.obs_words(observed)
    WDEL = WDEL_DEFAULT if wdel is None else wdel
    if action_alpha is not None:                         # word-action: the insertion RATE comes from theta,
        WINS = jnp.array([-unigram_surprisal(w)          # so WINS carries only the per-word CONTENT cost
                          for w in obs_words], jnp.float32)
    elif wins is not None:                               # explicit uniform scalar override
        WINS = wins
    elif uniform_ins:                                    # legacy flat -log(vocab) over the whole vocab
        WINS = insertion_loglik(ntok)
    else:                                                # frequency-aware: log(rate) - unigram_surprisal,
        WINS = jnp.array([math.log(ins_rate) - unigram_surprisal(w)  # so rare words are dear to drop
                          for w in obs_words], jnp.float32)
    if rejuv in ("gibbs", "gibbs+bd"):    # gibbs+bd also runs the embedded sub-sweep, which needs the KV pre-warm
        Wmax = len(obs_words) + slack
        # The rejuvenation pool is now built INSIDE pairhmm_smc.run from the shared candidate inventory
        # (so its surface ids match the augmented emit_full; R4 multi-token). Here we only pre-build the
        # KV-caching LM EAGERLY (outside the jitted sweep step) so its setup never runs under trace -- an
        # in-trace build leaks a tracer (UnexpectedTracerError). Size it for the multi-token worst case
        # (T_max <= T_UB tokens/word): the sweep's suffix tail can hold (lookback+1) words plus EOS.
        if model.tail_logprobs is not None:
            T_UB = 8
            LCTX = len(model.seed_ids) + Wmax * T_UB + 1
            lm_penzai._kv_setup(LCTX + (rejuv_lookback + 1) * T_UB + 1)
    return pairhmm_smc.run(observed, key, model, P=P, wdel=WDEL, wins=WINS, slack=slack,
                           band=band, max_dist=max_dist, Ke=Ke, J=J, cwin=cwin,
                           proposal="caprop", rejuv=rejuv, rejuv_pool=None,
                           rejuv_lookback=rejuv_lookback, rejuv_stats=rejuv_stats, trace=trace,
                           rejuv_dedup=dedup, lm_temp=lm_temp, action_alpha=action_alpha, channel=channel,
                           bd_bridge_j=bd_bridge_j, bd_pool_cap=bd_pool_cap, bd_p_stay=bd_p_stay,
                           bd_mode=bd_mode, bd_attempts=bd_attempts,
                           bd_funcword_ids=_funcword_ids() if bd_funcwords else None)


def decode(state, log_w, skip=1, key=jax.random.PRNGKey(0), top=3):
    return pairhmm_smc.decode(state, log_w, _pythia_model(PRIME), skip=skip, key=key, top=top)


def structured_output(observed, trace, logZ, P, band, max_dist, rejuv, rejuv_lookback, topk=8):
    """Package the per-step particle-cloud ``trace`` (recorded by ``pairhmm_smc.run(trace=[])``) into
    the step-explorer JSON for viz_template.html. See ``planning/TRACE_SCHEMA.md`` for the contract.

    The trace IS the artifact now: each ``steps[t]`` is a real SMC-step snapshot (the weighted latent
    distribution, the alignment-frontier histogram, the full per-particle dump, ESS, and resample /
    rejuvenation status). The terminal-corrected final posterior is the last step (``final: true``)."""
    ess_series = [s["ess"] for s in trace]
    return {
        "observed": observed,
        "config": {"lm": lm_penzai.MODEL_NAME, "particles": int(P), "band": band,
                   "max_dist": max_dist, "rejuv": rejuv, "lookback": rejuv_lookback,
                   "json_topk": topk, "resample_threshold": 0.5 * P},
        "log_marginal": logZ,
        "min_ess": min(ess_series) if ess_series else None,
        "ess_series": ess_series,
        "resample_steps": [s["t"] for s in trace if s["resampled"]],
        "rejuv_steps": [s["t"] for s in trace if s["rejuv"]],
        "steps": trace,
    }


def _norm(s):
    import re
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def main():
    # Smoke test at the VALIDATED budget (P=128; P=4 decodes pure noise, so it is no sanity check). Two
    # channels. (1) The DEPLOYMENT default -- word_action at the calibrated ACTION_ALPHA_DEFAULT -- on the
    # cases the calibrated prior reliably handles: a SUB typo (word-action's strength) and a clean KEEP
    # (retention ~0.99 at alpha_copy=200). (2) The char_copy CERTIFICATION ANCHOR on DEL/SUB/KEEP. The DEL
    # (missing-word restoration) smoke stays on char_copy on purpose: the calibrated copy concentration
    # deliberately suppresses bare-prior deletion restoration (it needs rejuv + a more capable LM --
    # planning/WORD_ACTION_ALPHA_SWEEP_PLAN.md), so asserting it under word_action would test a behaviour
    # the calibrated prior intentionally avoids. KEEP ('i want to go home') is a clean sentence the weak
    # 70m LM leaves alone -- it confirms the filter does not over-correct an already-correct sentence.
    lm_penzai.load_model()
    wa = dict(channel="word_action", action_alpha=ACTION_ALPHA_DEFAULT)   # the deployment default
    cc = dict(channel="char_copy")                                        # the exact-enumeration anchor
    trials = [
        ("word_action SUB (typo)   ", "teh cat sat on teh mat", "the cat sat on the mat", wa),
        ("word_action KEEP (clean)  ", "i want to go home",      "i want to go home",      wa),
        ("char_copy   DEL (missing) ", "i want go home",         "i want to go home",      cc),
        ("char_copy   SUB (typo)    ", "teh cat sat on teh mat", "the cat sat on the mat", cc),
        ("char_copy   KEEP (clean)  ", "i want to go home",      "i want to go home",      cc),
    ]
    for tag, obs, truth, kw in trials:
        st, lw, _, sl = run(obs, jax.random.PRNGKey(0), P=128, Ke=8, J=8, **kw)
        top = decode(st, lw, skip=sl)[0][0]
        ok = _norm(top) == _norm(truth)
        print(f"{tag}  {'OK' if ok else 'FAIL'}  truth={truth!r}  got={top!r}")


def cli():
    import argparse
    ap = argparse.ArgumentParser(description="Channel-aware pair-HMM noisy-channel SMC on Pythia.")
    ap.add_argument("--sentence", default=None, help="observed (noisy) sentence to correct")
    ap.add_argument("--particles", type=int, default=256)
    ap.add_argument("--band", type=int, default=2)
    ap.add_argument("--max_dist", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--wdel", type=float, default=None,
                    help=f"missing-word (over-editing) log-penalty in nats; more negative => fewer "
                         f"inferred extra words (default {WDEL_DEFAULT})")
    ap.add_argument("--wins", type=float, default=None,
                    help="override the spurious-word cost with a flat scalar in nats (default: the "
                         "frequency-aware per-word cost; see --ins_rate / --uniform_ins)")
    ap.add_argument("--ins_rate", type=float, default=0.02,
                    help="per-position spurious-insertion RATE rho_ins. The cost of explaining an "
                         "observed word as a spurious insertion is log(ins_rate) - unigram_surprisal(word), "
                         "so rare words are dear to drop and common words cheap. Smaller => fewer inferred "
                         "insertions (less word-dropping). Tune on a gold set alongside --wdel / --lm_temp.")
    ap.add_argument("--uniform_ins", action="store_true",
                    help="legacy: use the flat -log(vocab) spurious-word cost (uniform over the vocab) "
                         "instead of the frequency-aware default.")
    ap.add_argument("--lm_temp", type=float, default=1.0,
                    help="LM-prior temperature lambda: posterior is P_LM^lm_temp * P_channel. 1.0 = "
                         "untempered; <1 (e.g. 0.5) flattens pythia's over-confident preferences so "
                         "plausible inputs are read more literally (curbs over-editing of clean text); "
                         ">1 sharpens the prior (more aggressive correction).")
    ap.add_argument("--word_mask", action="store_true",
                    help="restrict the LM bridge pool to whole-word tokens (off by default)")
    ap.add_argument("--channel", choices=("word_action", "align", "char_copy"), default="align",
                    help="noise model: 'align' (DEFAULT; ALIGN_ACTION_CHANNEL_PLAN.md + ALIGN_OPT_RESULTS.md) = "
                         "the 3-way (align,insert,delete) Dirichlet channel that merges copy+sub into one 'align' "
                         "action with a smooth K*d edit-distance form cost, calibrated to ALIGN_ALPHA_DEFAULT + "
                         "ALIGN_SLOPE (K=-4.5, alpha=(200,2,2)); it catches real-word typos word_action misses "
                         "(garage->garbage) -- more correction at a small retention cost. 'word_action' "
                         "(WORD_ACTION_CHANNEL_PLAN.md) = the per-word 4-way (copy,sub,insert,delete) channel "
                         "calibrated to ACTION_ALPHA_DEFAULT (higher literal retention; the prior default). "
                         "'char_copy' = the deprecated bundled char channel, the exact-enumeration cert anchor.")
    ap.add_argument("--action_alpha", default=None,
                    help="override the action Dirichlet prior: 'copy,sub,ins,del' for --channel word_action "
                         f"(default {','.join(str(x) for x in ACTION_ALPHA_DEFAULT)}; implies word_action) or "
                         f"'align,ins,del' for --channel align (default "
                         f"{','.join(str(x) for x in ALIGN_ALPHA_DEFAULT)}).")
    ap.add_argument("--align_slope", type=float, default=None,
                    help="align channel: the per-edit log-cost K of the smooth distance form table "
                         f"(emission = K*edit_distance; default ALIGN_SLOPE={ALIGN_SLOPE:.3f}, calibrated). "
                         "Less negative => cheaper near-misses (more correction); more negative => fewer. "
                         "The single over-editing knob, decoupled from --action_alpha. Only used by --channel align.")
    ap.add_argument("--rejuv", choices=("off", "gibbs", "gibbs+bd"), default="off",
                    help="post-resample Gibbs/SMCP3 rejuvenation sweep (R3): 'gibbs' re-diversifies "
                         "the cloud and cures impoverishment collapses, at ~a few x the runtime "
                         "(KV-cached suffix scorer). 'gibbs+bd' adds the Gibbs indel (birth/death) move "
                         "that can restore dropped / remove spurious words (default bd_mode=gibbs, "
                         "function-word insertion pool on). 'off' is the certified forward-only filter.")
    ap.add_argument("--rejuv_lookback", type=int, default=3,
                    help="rejuvenation window: how many recent words each sweep revisits (default 3)")
    ap.add_argument("--no_dedup", action="store_true",
                    help="disable the EXACT post-resample LM-forward dedup (R3 item 1; on by default). "
                         "Dedup is bit-identical and ~2x faster on rejuv runs (the sweep prefills run on "
                         "the unique buffers only); turn off only to A/B the cost.")
    ap.add_argument("--output_json", default=None,
                    help="write the structured-output JSON here (view with genjax_port.viz)")
    ap.add_argument("--json_topk", type=int, default=8, help="hypotheses kept per step + in posterior")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        main()
        return
    if not args.sentence:
        ap.error("--sentence is required unless --selftest is set")

    import time
    channel = args.channel
    action_alpha = None
    if args.action_alpha is not None:                # an explicit prior implies an action channel
        action_alpha = tuple(float(x) for x in args.action_alpha.split(","))
        if channel == "char_copy":                   # an explicit alpha + the cert channel -> word_action
            channel = "word_action"
    lm_penzai.load_model()
    t0 = time.time()
    trace = [] if args.output_json else None     # record the per-step cloud trace only when writing JSON
    st, lw, logZ, sl = run(args.sentence, jax.random.PRNGKey(args.seed), P=args.particles,
                           band=args.band, max_dist=args.max_dist, wdel=args.wdel, wins=args.wins,
                           use_word_mask=args.word_mask, rejuv=args.rejuv,
                           rejuv_lookback=args.rejuv_lookback, trace=trace, dedup=not args.no_dedup,
                           lm_temp=args.lm_temp, ins_rate=args.ins_rate, uniform_ins=args.uniform_ins,
                           action_alpha=action_alpha, channel=channel, align_slope=args.align_slope)
    top = decode(st, lw, skip=sl, top=args.top)
    ins_desc = "uniform" if (args.uniform_ins or args.wins is not None) else f"rate={args.ins_rate}"
    print(f"observed : {args.sentence!r}")
    print(f"inferred intended (P={args.particles}, band={args.band}, rejuv={args.rejuv}, "
          f"lm_temp={args.lm_temp}, ins={ins_desc}, logZ={logZ:.2f}):")
    for s, p in top:
        print(f"   p={p:.2f}  {s!r}")
    print(f"runtime: {time.time() - t0:.0f}s")

    if args.output_json:
        import json
        out = structured_output(args.sentence, trace, logZ, P=args.particles, band=args.band,
                                max_dist=args.max_dist, rejuv=args.rejuv,
                                rejuv_lookback=args.rejuv_lookback, topk=args.json_topk)
        with open(args.output_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.output_json}")
        print(f"view: PYTHONPATH=src python -m genjax_port.viz {args.output_json}")


if __name__ == "__main__":
    cli()
