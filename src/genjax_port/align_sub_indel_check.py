"""Phase 3 (planning/ALIGN_ACTION_CHANNEL_PLAN.md sec 5): the sub-vs-indel calibration family + the
decisive behavioural check that the ALIGN channel fixes the garage class WITHOUT new over-editing.

The old battery has only ordinary-density, isolated d=1 subs. This family is the signal it lacked: a
substitution into a REAL word competing with a word-restoration / spurious-insertion, across a range of
neighbourhood density and LM-gap. Each item's correction (or literal, for KEEP) is unambiguous and its
LM preference is verified first (per the noisy-channel-test-examples rule).

The discriminating quantity is the LM gap an edit must clear: under word_action a substitution-vs-copy
costs (log p_sub - log p_copy) + SUB_FORM_LP = -8.56 nats at alpha=200, so word_action only corrects
when the LM prefers the neighbour by > 8.56; under align it costs only K = log(1/26) = -3.26, so align
corrects at LM-gap > 3.26. The garage case sits in the (3.26, 8.56) gap -- the LM prefers garbage by
~4.3, so align corrects it and word_action does not (it lets the LM's fluency hallucinate 'garage door').

EDIT items in that (3.26, 8.56) discriminating band are where align should newly succeed; EDIT items
above 8.56 are sanity (both channels correct -- align must not break them); KEEP items are the over-edit
guard -- the literal is a real word with a tempting real neighbour the LM does NOT prefer here, so a
cheaper substitution must NOT over-correct it.

Usage:  NC_LM=EleutherAI/pythia-70m PYTHONPATH=src conda run -n ncgenjax python -u \
          -m genjax_port.align_sub_indel_check [P] [LMonly]
        ('LMonly' = skip the filter runs, just verify the LM gaps; default P=128, rejuv=gibbs.)
"""
import sys
import time

import jax
import jax.numpy as jnp

from genjax_port import lm_penzai, tokenizer
from genjax_port import pythia_word_caprop as W
from genjax_port.pythia_word_caprop import _norm

EOS = lm_penzai.EOS_ID
pr = lambda *a: print(*a, flush=True)

# (id, kind, observed, intended).  For KEEP, intended == observed (the literal is correct); `neighbour`
# is the tempting WRONG edit, reported only to show the LM does not prefer it (the over-edit guard).
# kind in {edit_disc (LM-gap in the 3.26-8.56 align-discriminating band), edit_easy (gap > 8.56, both
# channels should correct), keep}.
FAMILY = [
    ("GARAGE",  "edit_disc", "The garage needs to be tossed out.", "The garbage needs to be tossed out.", "garbage"),
    ("QUIET",   "edit_disc", "She was quiet sure about it.",       "She was quite sure about it.",        "quite"),
    ("DESERT",  "edit_disc", "We had cake for desert tonight.",    "We had cake for dessert tonight.",    "dessert"),
    ("FORM",    "edit_easy", "I heard the news form a friend.",    "I heard the news from a friend.",     "from"),
    ("TRAIL",   "edit_easy", "The court trail lasted three weeks.","The court trial lasted three weeks.", "trial"),
    ("KEEP-DES", "keep",     "The desert was hot and dry.",        "The desert was hot and dry.",         "dessert"),
    ("KEEP-ADV", "keep",     "She gave me sound advice.",          "She gave me sound advice.",           "advise"),
    ("KEEP-QTE", "keep",     "I am quite happy today.",            "I am quite happy today.",             "quiet"),
]


def lm_logprob(text, seed_text="."):
    """Chain-rule log P(tokens, EOS | [EOS]+seed) over the filter's own ``next_token_logprobs``. The text
    is space-prefixed so its first word is word-initial (matching the filter's word-initial candidate
    spans); without it the first word attaches to the seed ('.The'), the malformed-prime artifact."""
    seed = [EOS] + (tokenizer.encode(seed_text) if seed_text else [])
    seq = seed + tokenizer.encode(" " + text.strip()) + [EOS]
    buf = jnp.array([seq + [EOS] * 4], jnp.int32)
    tot = 0.0
    for i in range(len(seed), len(seq)):
        lp = lm_penzai.next_token_logprobs(buf, jnp.array([i], jnp.int32))
        tot += float(lp[0, seq[i]])
    return tot


def _swap_word(sentence, old, new):
    return " ".join(new if _norm(w) == _norm(old) else w for w in sentence.split())


def run_filter(observed, channel, P, rejuv):
    kw = dict(channel=channel)
    if channel == "align":
        kw["action_alpha"] = list(W.ALIGN_ALPHA_DEFAULT)
    else:
        kw["action_alpha"] = list(W.ACTION_ALPHA_DEFAULT)
    st, lw, logZ, sl = W.run(observed, jax.random.PRNGKey(0), P=P, band=2, rejuv=rejuv, dedup=True, **kw)
    return W.decode(st, lw, skip=sl, top=3)


def main():
    P = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 128
    lm_only = "LMonly" in sys.argv
    rejuv = "gibbs"
    lm_penzai.load_model()
    pr(f"LM={lm_penzai.MODEL_NAME}  P={P}  rejuv={rejuv}  lm_only={lm_only}\n")

    # --- LM-gap verification (the discriminating quantity) ---
    pr(f"{'item':9s} {'kind':9s} {'LMgap':>7s}  band     obs -> correction")
    pr("-" * 92)
    for iid, kind, obs, intended, neigh in FAMILY:
        if kind == "keep":
            lit_lm = lm_logprob(obs)
            neigh_sent = _swap_word(obs, _find_kept(obs, neigh), neigh)   # swap the literal -> tempting neighbour
            neigh_lm = lm_logprob(neigh_sent)
            gap = neigh_lm - lit_lm                       # NEGATIVE => LM keeps the literal (good)
            band = "keep-lit" if gap < 0 else "AMBIG"
            pr(f"{iid:9s} {kind:9s} {gap:+7.2f}  {band:8s} keep {obs!r} (vs {neigh!r}: {neigh_lm:.1f} < lit {lit_lm:.1f}?)")
        else:
            gap = lm_logprob(intended) - lm_logprob(obs)  # POSITIVE => LM prefers the correction
            band = "DISCRIM" if 3.26 < gap < 8.56 else ("easy" if gap >= 8.56 else "BELOW-K")
            pr(f"{iid:9s} {kind:9s} {gap:+7.2f}  {band:8s} {obs!r} -> {intended!r}")
    if lm_only:
        return

    # --- behavioural filter runs: word_action (baseline) vs align (fix) ---
    pr(f"\n{'item':9s} {'kind':9s} {'channel':11s} {'verdict':8s}  MAP")
    pr("-" * 92)
    tally = {}
    for iid, kind, obs, intended, neigh in FAMILY:
        want = _norm(intended)
        for ch in ("word_action", "align"):
            t = time.time()
            top = run_filter(obs, ch, P, rejuv)
            maps, mapp = top[0]
            ok = _norm(maps) == want
            verdict = ("CORRECT" if ok else "miss") if kind != "keep" else ("KEPT" if ok else "OVER-EDIT")
            tally.setdefault((kind, ch), [0, 0]); tally[(kind, ch)][0] += 1; tally[(kind, ch)][1] += int(ok)
            pr(f"{iid:9s} {kind:9s} {ch:11s} {verdict:8s}  p={mapp:.2f} {maps!r}  ({time.time()-t:.0f}s)")
    pr("\n=== TALLY (correct/total) ===")
    for (kind, ch), (n, k) in sorted(tally.items()):
        pr(f"  {kind:9s} {ch:11s}  {k}/{n}")


def _find_kept(sentence, neigh):
    """The literal word in `sentence` that `neigh` is a near-edit of (for the KEEP over-edit probe)."""
    from genjax_port.noise_word import _damerau_levenshtein
    best, bd = None, 99
    for w in sentence.split():
        d = _damerau_levenshtein(_norm(w), _norm(neigh), 3)
        if d < bd:
            best, bd = w, d
    return best


if __name__ == "__main__":
    main()
