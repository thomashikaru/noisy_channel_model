"""Phase-3 gates (harness plan §4): the worker's harness-input handling and the collect join.

No jax anywhere here -- these cover the login-safe worker paths (read_items, the (context, length)
shard key, the context-aware resume, the unit<->token map, json serialization, the multi-seed words
merge) plus ``collect.py`` end-to-end on a fabricated results tree (pandas; skipped without it).
"""

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "slurm"))       # run_nc_batch.py is a script, not a package
import run_nc_batch as W                       # noqa: E402


def test_read_items_jsonl_and_txt(tmp_path):
    j = tmp_path / "toy.input.jsonl"
    j.write_text('{"sentence_id": 0, "text": "A b.", "context": ""}\n'
                 "# comment\n"
                 '{"sentence_id": 1, "text": "C d.", "context": "Some prime."}\n')
    items = W.read_items(str(j))
    assert [it["idx"] for it in items] == [0, 1]
    assert items[1]["context"] == "Some prime."
    t = tmp_path / "toy.txt"
    t.write_text("A b.\n\n# skip\nC d.\n")
    items = W.read_items(str(t))
    assert [it["text"] for it in items] == ["A b.", "C d."]
    assert all(it["context"] == "" for it in items)
    bad = tmp_path / "bad.input.jsonl"
    bad.write_text('{"sentence_id": 1, "text": "A b.", "context": ""}\n')
    with pytest.raises(ValueError):
        W.read_items(str(bad))                # reordered list would re-map finished results


def test_item_length_key_groups_both_compile_axes():
    a = {"text": "The cat sat.", "context": ""}
    b = {"text": "The cat sat.", "context": "One two three."}
    c = {"text": "The cat sat on mats.", "context": "One two three."}
    assert W._item_length_key(a) == (0, 4)     # 3 words + attached '.'
    assert W._item_length_key(b) == (3, 4)
    assert W._item_length_key(b) != W._item_length_key(c)
    plan = W._shard_plan([a, b, c, dict(a)], sort_by_length=True, min_size=1, max_size=2)
    flat = sorted(i for sh in plan for i in sh)
    assert flat == [0, 1, 2, 3]                # a permutation of the indices, keyed by (ctx, len)


def test_item_status_is_context_aware(tmp_path):
    p = tmp_path / "item_00000.json"
    p.write_text(json.dumps({"observed": "A b.", "context": "P.", "status": "ok"}))
    assert W._item_status(str(p), "A b.", "P.") == "done"
    assert W._item_status(str(p), "A b.", "OTHER.") == "stale"   # edited context -> recompute
    assert W._item_status(str(p), "A b. c.", "P.") == "stale"
    legacy = tmp_path / "item_00001.json"                        # pre-Phase-3 record: no context key
    legacy.write_text(json.dumps({"observed": "A b.", "status": "ok"}))
    assert W._item_status(str(legacy), "A b.", "") == "done"     # old .txt results stay done


def test_unit_map_attaches_punctuation_to_its_token():
    text = "The cat sat, on teh mat."
    units = ["The", "cat", "sat", ",", "on", "teh", "mat", "."]
    m = W._unit_map(text, units, [1] * len(units))
    assert [u["stim_word_idx"] for u in m] == [0, 1, 2, 2, 3, 4, 5, 5]
    assert [u["is_punct"] for u in m] == [0, 0, 0, 1, 0, 0, 0, 1]
    assert [u["unit_idx"] for u in m] == list(range(8))


def test_jsonable_nulls_nonfinite():
    np = pytest.importorskip("numpy")
    x = {"a": np.array([1.0, np.inf, -np.inf, np.nan]), "b": np.int32(3),
         "c": np.bool_(True), "d": [np.float32(2.5)], "e": float("-inf")}
    out = W._jsonable(x)
    assert out == {"a": [1.0, None, None, None], "b": 3, "c": True, "d": [2.5], "e": None}
    json.dumps(out)                            # actually serializable


def _fake_words(plq, p_copy, stay=(1, 4, 1, 3.5)):
    n_events, n_active, n_changed, stay_sum = stay
    units = []
    for i in range(len(plq) - 1):
        units.append({"unit_idx": i, "text": f"w{i}", "stim_word_idx": i, "is_punct": 0,
                      "n_tokens": 1, "surprisal_lm": 2.0 + i,
                      "p_copy": p_copy, "p_sub": 1 - p_copy, "p_ins": 0.0,
                      "p_err": 1 - p_copy, "p_err_positional": 1 - p_copy, "del_before": 0.1,
                      "rejuv": {"n_events": n_events, "n_active": n_active,
                                "n_changed": n_changed, "stay_sum": stay_sum,
                                "change_rate": n_changed / n_active,
                                "stay_prob": stay_sum / n_active}})
    return {"status": "ok", "prime": ".", "lm_temp": 1.0, "convention": {},
            "prefix_logq": list(plq), "surprisal_end_nc": plq[-1] - (-5.0),
            "surprisal_end_lm": 3.0, "del_after_last": 0.2, "units": units}


def test_merge_words_masses_expectations_and_pooling():
    r1 = {"logZ": -5.0, "words": _fake_words([0.0, -1.0, -3.0], p_copy=0.8)}
    r2 = {"logZ": -5.0, "words": _fake_words([0.0, -2.0, -3.0], p_copy=0.6)}
    m = W._merge_words([r1, r2], [0.5, 0.5])
    assert m["status"] == "ok" and m["n_seeds"] == 2
    # prefix masses: mean in MASS space
    want1 = math.log((math.exp(-1.0) + math.exp(-2.0)) / 2)
    assert abs(m["prefix_logq"][1] - want1) < 1e-12
    # surprisal recomputed from the merged masses, not averaged
    assert abs(m["units"][0]["surprisal_nc"] - (m["prefix_logq"][0] - m["prefix_logq"][1])) < 1e-12
    # posterior expectations: evidence-weighted (equal logZ -> plain mean)
    assert abs(m["units"][0]["p_copy"] - 0.7) < 1e-12
    # rejuv: POOLED counts, rates recomputed
    assert m["units"][0]["rejuv"]["n_active"] == 8
    assert abs(m["units"][0]["rejuv"]["change_rate"] - 2 / 8) < 1e-12
    # S_end from merged masses and merged logZ
    assert abs(m["surprisal_end_nc"] - (m["prefix_logq"][-1] - (-5.0))) < 1e-12
    # a words-less seed is skipped, not fatal
    assert W._merge_words([{"logZ": -5.0, "words": {"status": "error"}}], [1.0]) is None


def test_collect_end_to_end(tmp_path):
    pd = pytest.importorskip("pandas")
    sys.path.insert(0, str(REPO / "experiments"))
    import collect as C

    stim_dir = tmp_path / "stimuli"; stim_dir.mkdir()
    (stim_dir / "toy.stimuli.csv").write_text(
        "dataset,subset,item_id,condition,stim_uid,model_input,context,sentence_id,"
        "plausibility,is_grammatical,contrast,intended_uids,n_intended,critical_word_idx\n"
        "toy,,1,a,toy//1/a,A b.,,0,plausible,1,dative,toy//1/a,1,0\n"
        "toy,,1,b,toy//1/b,A b.,,0,implausible,1,dative,toy//1/a,1,0\n"   # shares sentence_id 0
        "toy,,2,a,toy//2/a,C d.,,1,plausible,1,dative,toy//2/a,1,0\n")
    res = tmp_path / "results_nc" / "toy.input" / "lm-x__ch-align" / "results"
    res.mkdir(parents=True)
    rec = {"idx": 0, "observed": "A b.", "context": "", "status": "ok", "map": "A b.",
           "hypotheses": [{"sentence": "A b.", "prob": 0.9}, {"sentence": "A c.", "prob": 0.1}],
           "logZ": -4.0, "logZ_stats": {"std": 0.1, "spread": 0.2}, "p_literal": 0.9,
           "n_seeds": 2, "runtime_s": 1.0, "git": {"sha": "f" * 40},
           "words": _fake_words([0.0, -1.0, -3.0], p_copy=0.9)}
    (res / "item_00000.json").write_text(json.dumps(rec))
    (res / "item_00000_s0.json").write_text(json.dumps(rec))       # one per-seed record too
    # sentence_id 1 has no record -> missing
    out = tmp_path / "outputs"
    C.main(["toy", "--results-root", str(tmp_path / "results_nc"),
            "--stimuli", str(stim_dir), "--out", str(out)])
    d = out / "lm-x__ch-align" / "toy"
    sent = pd.read_csv(d / "sentences.csv.gz", keep_default_na=False)
    assert len(sent) == 3                                           # one row per STIMULUS row
    assert list(sent["status"]) == ["ok", "ok", "missing"]
    assert set(sent[sent.status == "ok"]["map_edit_type"]) == {"none"}
    post = pd.read_csv(d / "posterior.csv.gz")
    assert len(post) == 2 and post.iloc[1]["hyp_edit_type"] == "sub"
    words = pd.read_csv(d / "words.csv.gz")
    assert set(words["seed"].astype(str)) == {"merged", "0"}
    assert len(words) == 4                                          # 2 units x (merged + seed 0)
    assert (out / "status.md").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
