"""Pythia config of the unified pair-HMM RB-SMC filter (Phase 0 / M-A).

This is now a thin model config over :mod:`genjax_port.pairhmm_smc`: it supplies the Pythia-specific
injections (``next_token_logprobs`` as the LM, a char channel over surface forms, SymSpell
candidates, the ``"."`` prime seed) and delegates all inference to the shared filter. The toy bigram
(``tests/test_pairhmm_exact.py``) and Pythia therefore run *identical* inference code -- correctness
proven on the toy by exact enumeration transfers here by construction.

INSERT / multi-token / KV are Phase 1+ (see ``planning/PAIRHMM_RBSMC_PLAN.md``). The explicit INSERT
move is left OFF here (``insert_action=False``) to preserve the A1 behaviour; A3 turns it on with the
principled (non-heuristic) gate.

Run:  NC_LM=EleutherAI/pythia-70m python -m genjax_port.pythia_word_caprop --selftest
"""

import functools

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from genjax_port import lm_penzai, tokenizer, pairhmm_smc
from genjax_port.noise_word import word_sub_candidates, segment_words
from genjax_port.config import P_DELETE_PRIOR
from genjax_port.noise import insertion_loglik

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

# Content-neutral ". " (period + SPACE) steers the LM out of document-start boilerplate. The leading
# <|endoftext|> seed otherwise asks for the doc-START distribution (titles/"I"/headers); a trailing
# ". " conditions the model as if mid-document, so the next token is an ordinary sentence start. The
# space matters: ". " tokenizes differently from "." and steers markedly better (A/B at P=64). For
# hard cases a full neutral carrier sentence primes even more strongly -- pass it via --prime.
PRIME = ". "


def _char_ids(s):
    s = s.strip().lower()
    ids = [ord(c) for c in s][:LC]
    n = len(ids)
    return ids + [CHAR_PAD] * (LC - n), n


def channel_logpdf(observed_ids, intended_ids, n_x):
    n_o = jnp.sum(observed_ids != CHAR_PAD)
    row0 = jnp.arange(LC + 1, dtype=jnp.float32) * INS_LP

    def fill_row(prev_row, x_char):
        cur0 = prev_row[0] + DEL_LP

        def step(left, cols):
            o_char, prev_diag, prev_up = cols
            sub = prev_diag + jnp.where(o_char == x_char, COPY_LP, SUB_LP)
            dele = prev_up + DEL_LP
            ins = left + INS_LP
            cell = logsumexp(jnp.stack([sub, dele, ins]))
            return cell, cell

        cols = (observed_ids, prev_row[:-1], prev_row[1:])
        _, rest = jax.lax.scan(step, cur0, cols)
        cur_row = jnp.concatenate([cur0[None], rest])
        return cur_row, cur_row

    _, rows = jax.lax.scan(fill_row, row0, intended_ids)
    grid = jnp.concatenate([row0[None], rows])
    return grid[n_x, n_o]


@functools.lru_cache(maxsize=1)
def _vocab_char_table():
    strs = tokenizer.vocab_strings()
    buf, lens = [], []
    for s in strs:
        ids, n = _char_ids(s)
        buf.append(ids)
        lens.append(n)
    return jnp.array(buf, jnp.int32), jnp.array(lens, jnp.int32)


def _obs_word_units(observed):
    obs_ids = tokenizer.encode(observed.strip())
    return [unit_str for _ids, unit_str in segment_words(obs_ids)]


def _candidate_ids(word, max_dist, Ke):
    """Candidate intended word ids for an observed word: the COPY (the observed word's own single
    token, if it is one) FIRST, then SymSpell substitution neighbours. word_sub_candidates excludes
    the literal (it is the copy branch), so without prepending it the filter could never emit a
    correctly-spelled observed word -- it would only enter via the top-J LM and drift to boilerplate.
    Mirrors the toy candidate scan, which keeps distance-0; deduped, copy-first, capped to Ke."""
    body = word.strip().lower()
    ids = []
    lit = tokenizer.encode(" " + body)
    if len(lit) == 1:                       # observed word IS a single word-initial token -> COPY
        ids.append(lit[0])
    ids += [tid for tid, _d in word_sub_candidates(body, max_dist=max_dist)]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
        if len(out) >= Ke:
            break
    return out


@functools.lru_cache(maxsize=4)
def _pythia_model(prime, lm_logprobs_fn=None):
    """Build the Pythia :class:`pairhmm_smc.PairHMMModel`. Cached per prime so the vocab char table
    + seed are reused across runs. ``lm_logprobs_fn`` defaults to the loaded penzai model."""
    vocab_char, vocab_clen = _vocab_char_table()
    lm_fn = lm_logprobs_fn or lm_penzai.next_token_logprobs
    seed_ids = [EOS_ID] + (tokenizer.encode(prime) if prime else [])
    return pairhmm_smc.PairHMMModel(
        lm_fn=lm_fn, eos_id=EOS_ID, emit_vocab=vocab_char.shape[0],
        vocab_char=vocab_char, vocab_clen=vocab_clen, channel_logpdf=channel_logpdf,
        char_ids=_char_ids, candidate_ids=_candidate_ids, obs_words=_obs_word_units,
        decode_ids=lambda t: tokenizer.decode(t).strip(), seed_ids=tuple(seed_ids))


def run(observed, key, P=64, wdel=None, wins=None, slack=3, band=2,
        max_dist=2, Ke=8, J=8, cwin=1, prime=PRIME, lm_logprobs_fn=None):
    """Channel-aware RB-SMC on Pythia via the shared filter. Returns (state, log_w, logZ, seed_len)."""
    if lm_logprobs_fn is None:
        lm_penzai.load_model()
    model = _pythia_model(prime, lm_logprobs_fn)
    ntok = model.emit_vocab
    WDEL = float(jnp.log(P_DELETE_PRIOR)) if wdel is None else wdel
    WINS = insertion_loglik(ntok) if wins is None else wins
    return pairhmm_smc.run(observed, key, model, P=P, wdel=WDEL, wins=WINS, slack=slack,
                           band=band, max_dist=max_dist, Ke=Ke, J=J, cwin=cwin,
                           proposal="caprop", insert_action=False)


def decode(state, log_w, skip=1, key=jax.random.PRNGKey(0), top=3):
    return pairhmm_smc.decode(state, log_w, _pythia_model(PRIME), skip=skip, key=key, top=top)


def _norm(s):
    import re
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def main():
    lm_penzai.load_model()
    trials = [
        ("DEL (missing) ", "i want go home", "i want to go home"),
        ("SUB (typo)    ", "teh cat sat on teh mat", "the cat sat on the mat"),
        ("KEEP (clean)  ", "the dog ran in the park", "the dog ran in the park"),
    ]
    for tag, obs, truth in trials:
        st, lw, _, sl = run(obs, jax.random.PRNGKey(0), P=4, Ke=8, J=8)
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
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        main()
        return
    if not args.sentence:
        ap.error("--sentence is required unless --selftest is set")

    import time
    lm_penzai.load_model()
    t0 = time.time()
    st, lw, logZ, sl = run(args.sentence, jax.random.PRNGKey(args.seed), P=args.particles,
                           band=args.band, max_dist=args.max_dist)
    top = decode(st, lw, skip=sl, top=args.top)
    print(f"observed : {args.sentence!r}")
    print(f"inferred intended (P={args.particles}, band={args.band}, logZ={logZ:.2f}):")
    for s, p in top:
        print(f"   p={p:.2f}  {s!r}")
    print(f"runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cli()
