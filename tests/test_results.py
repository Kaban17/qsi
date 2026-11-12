import json

from qsi.results import atomic_write_text, read_metrics, update_metrics


def test_tiny_fixtures(tiny_model, tiny_tokenizer):
    ids = tiny_tokenizer.apply_chat_template(
        [{"role": "user", "content": "ab"}], return_tensors="pt", return_dict=True
    )["input_ids"]
    assert ids[0].tolist()[0] == tiny_tokenizer.bos_token_id and ids.shape[1] == 5
    out = tiny_model(ids, output_hidden_states=True)
    assert len(out.hidden_states) == 4 and out.hidden_states[0].shape[-1] == 32


def test_update_metrics_merges(tmp_path):
    update_metrics(tmp_path, {"a": 1, "sae": {"6": {"fvu": 0.1}}})
    m = update_metrics(tmp_path, {"sae": {"12": {"fvu": 0.2}}})
    assert m == {"a": 1, "sae": {"6": {"fvu": 0.1}, "12": {"fvu": 0.2}}}
    assert read_metrics(tmp_path) == m


def test_read_metrics_missing(tmp_path):
    assert read_metrics(tmp_path / "nope") == {}


def test_update_metrics_nonfinite(tmp_path):
    m = update_metrics(tmp_path, {"perplexity": float("inf"), "refusal": {"auroc": float("nan")}})
    assert m["perplexity"] is None and m["refusal"]["auroc"] is None
    assert m["nonfinite"] == {"perplexity": "inf", "refusal.auroc": "nan"}
    text = (tmp_path / "metrics.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == m


def test_nonfinite_inside_list(tmp_path):
    m = update_metrics(tmp_path, {"cos": [1.0, float("nan")]})
    assert m["cos"] == [1.0, None] and m["nonfinite"] == {"cos.1": "nan"}


def test_nonfinite_cleared_when_value_becomes_finite(tmp_path):
    update_metrics(tmp_path, {"perplexity": float("inf")})
    m = update_metrics(tmp_path, {"perplexity": 3.0})
    assert m["perplexity"] == 3.0 and "nonfinite" not in m


def test_atomic_write_leaves_no_tmp(tmp_path):
    atomic_write_text(tmp_path / "x.json", "{}")
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_update_creates_dir(tmp_path):
    update_metrics(tmp_path / "fp16", {"a": 1})
    assert read_metrics(tmp_path / "fp16") == {"a": 1}
