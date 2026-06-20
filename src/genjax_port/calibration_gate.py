"""P3 calibration gate: verify each battery item is *fit-ready* before it can constrain a
parameter (see planning/CALIBRATION_PLAN.md §4 and planning/CALIBRATION_BATTERY_DRAFT.md).

For every item in the battery CSV we measure the three gates that separate a genuine
model/human discrepancy from a model-capacity or inference artifact:

  G1  LM-noticeability   slp(intended) - slp(observed) > 0  (edit items): the LM must prefer the
      corrected reading over the literal one BEFORE any channel cost, else the correction is
      unreachable for reasons of the prior, not calibration.
  G2  reachability       the production candidate generator (SymSpell, max_dist) actually surfaces
      the intended word (substitution items).
  G3  representability    the fix is not a dropped MULTI-token word (deletion items): to/for/from
      are single-token, so common deletions pass; flags the one deferred capacity limit.

Keep ("don't edit") items are controls: they pass on fluency (finite slp). For the same-word
substitution pairs we also report the plausibility contrast (does the LM resist the tempting edit
in the plausible context?) -- the cleanest behavioural signal for the lambda knob.

slp is the LM log-prob of the word sequence after the production seed (EOS + "." prime), NO trailing
EOS (pure sequence fluency; the length prior lives in WDEL, not here). Reads ONLY the battery CSV --
never the reserved data/ sets.

Run (needs the ncgenjax arm64 env):
  conda run -n ncgenjax python -u -m genjax_port.calibration_gate \
      planning/calibration_battery_v0.csv planning/calibration_battery_v0_gated.csv
"""
import csv
import difflib
import sys
import time

import jax.numpy as jnp
import numpy as np

from genjax_port import lm_penzai, tokenizer, noise_word

EOS = lm_penzai.EOS_ID
_t0 = time.time()


def log(msg):
    print(f"[{time.time() - _t0:6.1f}s] {msg}", flush=True)


def slp_batch(sentences, prime="."):
    """LM log-prob of each sentence (word sequence after EOS + prime), batched. No trailing EOS."""
    seed = [EOS] + (tokenizer.encode(prime) if prime else [])
    tails = [tokenizer.encode(" " + s.strip()) for s in sentences]
    W = max(len(t) for t in tails)
    B, cw = len(sentences), len(seed)
    tot = cw + W
    ctx = np.full((B, tot), EOS, np.int32)
    ctx[:, :cw] = np.array(seed, np.int32)
    clen = np.full(B, cw, np.int32)
    tarr = np.zeros((B, 1, W), np.int32)
    tlen = np.zeros((B, 1), np.int32)
    for i, t in enumerate(tails):
        tarr[i, 0, : len(t)] = t
        tlen[i, 0] = len(t)
    sc = lm_penzai.batch_tail_logprobs(jnp.array(ctx), jnp.array(clen), jnp.array(tarr), jnp.array(tlen))
    return np.asarray(sc[:, 0], dtype=np.float64)


def _body(word):
    """Strip surrounding punctuation/space -> the alphabetic body, lowercased."""
    return "".join(c for c in word if c.isalpha()).lower()


def word_change(observed, intended):
    """Return (kind, from_word, to_word) for the single edit between the two strings.
    kind in {sub, ins(=missing word restored), del(=spurious word removed), none}."""
    ow, iw = observed.split(), intended.split()
    ops = [op for op in difflib.SequenceMatcher(a=ow, b=iw, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return ("none", "", "")
    tag, i1, i2, j1, j2 = ops[0]
    if tag == "replace":
        return ("sub", " ".join(ow[i1:i2]), " ".join(iw[j1:j2]))
    if tag == "insert":  # intended has a word the observed lacks -> a restored (missing) word
        return ("ins", "", " ".join(iw[j1:j2]))
    if tag == "delete":  # observed has a word the intended lacks -> a spurious word
        return ("del", " ".join(ow[i1:i2]), "")
    return ("none", "", "")


def reachable(obs_body, int_body, max_dist=2):
    """Does the production candidate generator surface int_body from obs_body within max_dist?"""
    d = noise_word._damerau_levenshtein(obs_body, int_body, max_dist + 1)
    if not (1 <= d <= max_dist):
        return (False, d, "dist>max_dist")
    single = {tokenizer.surface(tid).strip().lower() for tid, _ in noise_word.word_sub_candidates(obs_body, max_dist)}
    if int_body in single:
        return (True, d, "single-token")
    multi = {s.strip().lower() for _, s, _ in noise_word.word_sub_candidates_multitoken(obs_body, max_dist)}
    if int_body in multi:
        return (True, d, "multi-token")
    return (False, d, "not_retrieved")


def main(in_csv, out_csv):
    rows = list(csv.DictReader(open(in_csv)))
    log(f"loaded {len(rows)} items from {in_csv}")
    log(f"LM = {lm_penzai.MODEL_NAME}; loading model (one-time compile, ~30-60s on 70m)...")

    # Score every distinct sentence once (observed + intended + SUBW tempting counterfactuals).
    sents = {}
    for r in rows:
        sents[r["observed"]] = None
        sents[r["intended"]] = None
    # Keep-item counterfactuals: apply the pair's edit to the plausible context, so we can ask whether
    # the LM is tempted to edit even here (the "don't over-edit" / negative-preference anchor). The
    # edit is family-specific: substitution -> swap the real word; insertion -> remove the (single) word.
    pair_info = {}  # pair_id -> (kind, a, b) from the edit member
    for r in rows:
        if r["expected"] != "edit":
            continue
        k, fw, tw = word_change(r["observed"], r["intended"])
        if r["edit_type"] == "sub" and k == "sub":
            pair_info[r["pair_id"]] = ("swap", fw, tw)
        elif r["edit_type"] == "ins" and k == "del":   # INS family: the edit removes the doubled word fw
            pair_info[r["pair_id"]] = ("remove", fw, "")
    cf = {}  # keep item_id -> counterfactual (tempting-edit) sentence
    for r in rows:
        if r["expected"] != "keep" or r["pair_id"] not in pair_info:
            continue
        kind, a, b = pair_info[r["pair_id"]]
        if not a or (" " + a + " ") not in (" " + r["observed"] + " "):
            continue
        if kind == "swap":
            c = (" " + r["observed"] + " ").replace(" " + a + " ", " " + b + " ", 1).strip()
        else:  # remove the single occurrence -> treat the needed word as spurious (LM should resist)
            c = (" " + r["observed"] + " ").replace(" " + a + " ", " ", 1).strip()
        cf[r["item_id"]] = c
        sents[c] = None

    keys = list(sents)
    log(f"scoring {len(keys)} distinct sentences...")
    vals = slp_batch(keys)
    slp = {k: float(v) for k, v in zip(keys, vals)}
    log("scoring done; applying gates")

    out_fields = list(rows[0].keys()) + [
        "slp_observed", "slp_intended", "slp_gain", "gate_slp",
        "edit_from", "edit_to", "reach_dist", "reach_ok", "miss_ntok",
        "contrast_gain_keep", "gate_pass", "gate_note",
    ]
    out = []
    for r in rows:
        d = dict(r)
        so, si = slp[r["observed"]], slp[r["intended"]]
        gain = si - so
        d["slp_observed"] = round(so, 3)
        d["slp_intended"] = round(si, 3)
        d["slp_gain"] = round(gain, 3)
        d.update(dict.fromkeys(["gate_slp", "edit_from", "edit_to", "reach_dist", "reach_ok",
                                "miss_ntok", "contrast_gain_keep", "gate_note"], ""))
        notes = []
        if r["expected"] == "edit":
            d["gate_slp"] = "pass" if gain > 0 else "FAIL"
            if gain <= 0:
                notes.append(f"LM does not prefer correction (gain {gain:+.2f})")
            kind, fw, tw = word_change(r["observed"], r["intended"])
            d["edit_from"], d["edit_to"] = fw, tw
            ok = True
            if r["edit_type"] == "sub":
                rok, rd, rwhy = reachable(_body(fw), _body(tw))
                d["reach_dist"], d["reach_ok"] = rd, ("yes" if rok else "NO")
                if not rok:
                    notes.append(f"unreachable ({rwhy})")
                    ok = False
            elif r["edit_type"].startswith("del_"):
                nt = len(tokenizer.encode(" " + tw))
                d["miss_ntok"] = nt
                if nt != 1:
                    notes.append(f"multi-token deletion ({nt} tok) -- deferred capacity")
                    ok = False
            d["gate_pass"] = "PASS" if (gain > 0 and ok) else "FAIL"
        else:  # keep / control
            d["gate_slp"] = "n/a"
            d["gate_pass"] = "PASS" if np.isfinite(so) else "FAIL"
            if r["item_id"] in cf:
                cg = slp[cf[r["item_id"]]] - so
                d["contrast_gain_keep"] = round(cg, 3)
                if cg > 0:
                    notes.append(f"LM prefers the tempting edit even here (+{cg:.2f}) -- weak control")
        d["gate_note"] = "; ".join(notes)
        out.append(d)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(out)
    log(f"wrote {out_csv}")

    # ---- report ----
    def rpt(title, items):
        n = len(items)
        p = sum(1 for x in items if x["gate_pass"] == "PASS")
        print(f"  {title:14s}  {p}/{n} pass")

    print("\n================ CALIBRATION GATE REPORT ================")
    edits = [x for x in out if x["expected"] == "edit"]
    keeps = [x for x in out if x["expected"] == "keep"]
    print(f"LM={lm_penzai.MODEL_NAME}  prime='.'  max_dist=2")
    print(f"EDIT items: {sum(1 for x in edits if x['gate_pass']=='PASS')}/{len(edits)} fit-ready")
    print(f"KEEP items: {sum(1 for x in keeps if x['gate_pass']=='PASS')}/{len(keeps)} fluent controls")
    print("\nby family (edit items):")
    for fam in ["SUBW", "SUBN", "DEL_TO", "DEL_FOR", "DEL_FROM", "DEL_OF", "DEL_A", "DEL_THE",
                "INS_DUP", "INS_TO", "INS_FOR", "LADDER"]:
        fi = [x for x in edits if x["family"] == fam]
        if fi:
            rpt(fam, fi)
    print("\nFAILED edit items:")
    for x in edits:
        if x["gate_pass"] != "PASS":
            print(f"  {x['item_id']:16s} gain={x['slp_gain']:+7.2f}  {x['gate_note']}")
    print("\nweak KEEP controls (LM prefers the tempting edit):")
    weak = [x for x in keeps if x["contrast_gain_keep"] != "" and float(x["contrast_gain_keep"]) > 0]
    for x in weak:
        print(f"  {x['item_id']:16s} contrast=+{x['contrast_gain_keep']}")
    if not weak:
        print("  (none)")
    print("========================================================")


if __name__ == "__main__":
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "planning/calibration_battery_v0.csv"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "planning/calibration_battery_v0_gated.csv"
    main(in_csv, out_csv)
