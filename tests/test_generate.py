import json

import pytest
import torch

from qsi.data import Prompt
from qsi.generate import format_chat, generate_responses, generate_texts


def test_format_chat(tiny_tokenizer):
    assert format_chat(tiny_tokenizer, "ab") == "<bos><ab>"


def manual_generate(model, tokenizer, text, n):
    ids = tokenizer(format_chat(tokenizer, text), return_tensors="pt", add_special_tokens=False).input_ids
    out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def test_generate_texts_returns_only_new_tokens(tiny_model, tiny_tokenizer):
    texts = ["ab", "abcde", "a"]
    got = generate_texts(tiny_model, tiny_tokenizer, texts, max_new_tokens=5, batch_size=1)
    assert got == [manual_generate(tiny_model, tiny_tokenizer, t, 5) for t in texts]


def test_generate_texts_independent_of_batching(tiny_model, tiny_tokenizer):
    texts = ["ab", "abcde", "a", "hello there", "xyz"]
    a = generate_texts(tiny_model, tiny_tokenizer, texts, max_new_tokens=5, batch_size=1)
    b = generate_texts(tiny_model, tiny_tokenizer, texts, max_new_tokens=5, batch_size=2)
    c = generate_texts(tiny_model, tiny_tokenizer, texts, max_new_tokens=5, batch_size=5)
    assert len(a) == 5 and a == b == c


def test_generate_texts_empty_input(tiny_model, tiny_tokenizer):
    assert generate_texts(tiny_model, tiny_tokenizer, []) == []


def test_oom_halves_batch(tiny_model, tiny_tokenizer, monkeypatch):
    real = tiny_model.generate
    sizes = []

    def flaky(**kwargs):
        sizes.append(kwargs["input_ids"].shape[0])
        if kwargs["input_ids"].shape[0] > 1:
            raise torch.cuda.OutOfMemoryError("fake")
        return real(**kwargs)

    expected = generate_texts(tiny_model, tiny_tokenizer, ["ab", "abc", "a", "b"], max_new_tokens=3, batch_size=1)
    monkeypatch.setattr(tiny_model, "generate", flaky)
    got = generate_texts(tiny_model, tiny_tokenizer, ["ab", "abc", "a", "b"], max_new_tokens=3, batch_size=4)
    assert got == expected
    assert sizes[:3] == [4, 2, 1] and sizes.count(1) == 4


def test_oom_at_batch_one_propagates(tiny_model, tiny_tokenizer, monkeypatch):
    def always(**kwargs):
        raise torch.cuda.OutOfMemoryError("fake")

    monkeypatch.setattr(tiny_model, "generate", always)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        generate_texts(tiny_model, tiny_tokenizer, ["ab", "abc"], max_new_tokens=3, batch_size=2)


def test_generate_responses_writes_and_skips(tiny_model, tiny_tokenizer, tmp_path):
    ps = [Prompt("harmful_eval-000", "ab", "harmful_eval"), Prompt("harmless_eval-000", "ba", "harmless_eval")]
    out = tmp_path / "cfg" / "generations.jsonl"
    assert generate_responses(tiny_model, tiny_tokenizer, ps, out, max_new_tokens=3) is True
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == [p.id for p in ps]
    assert set(rows[0]) == {"id", "split", "prompt", "response"}
    assert rows[1]["split"] == "harmless_eval" and rows[1]["prompt"] == "ba"
    assert rows[0]["response"] == manual_generate(tiny_model, tiny_tokenizer, "ab", 3)
    assert generate_responses(None, None, ps, out) is False  # model is not touched


def test_leftover_tmp_is_not_done(tiny_model, tiny_tokenizer, tmp_path):
    out = tmp_path / "generations.jsonl"
    (tmp_path / "generations.jsonl.tmp").write_text("garbage")
    assert generate_responses(tiny_model, tiny_tokenizer, [Prompt("x-000", "ab", "x")], out, max_new_tokens=2) is True
    assert len(out.read_text().splitlines()) == 1
    assert not (tmp_path / "generations.jsonl.tmp").exists()


def test_empty_response_is_written(tiny_model, tiny_tokenizer, tmp_path, monkeypatch):
    monkeypatch.setattr("qsi.generate.generate_texts", lambda *a, **k: ["", "  "])
    out = tmp_path / "g.jsonl"
    generate_responses(tiny_model, tiny_tokenizer, [Prompt("a-000", "a", "a"), Prompt("a-001", "b", "a")], out)
    assert [json.loads(line)["response"] for line in out.read_text().splitlines()] == ["", "  "]


def test_non_ascii_roundtrip(tiny_model, tiny_tokenizer, tmp_path, monkeypatch):
    monkeypatch.setattr("qsi.generate.generate_texts", lambda *a, **k: ["привет\n\"x\""])
    out = tmp_path / "g.jsonl"
    generate_responses(tiny_model, tiny_tokenizer, [Prompt("a-000", "a", "a")], out)
    lines = out.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["response"] == "привет\n\"x\""
