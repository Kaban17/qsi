import json
import math

import pytest
import torch

from qsi.pipeline import run_stage0, truncate_splits
from qsi.results import read_metrics


@pytest.fixture
def wikitext():
    torch.manual_seed(0)
    return torch.randint(4, 48, (3, 12))


def exploding_loader(name):
    raise AssertionError("model must not be loaded")


def test_stage0_writes_generations_and_perplexity(tmp_path, tiny_loader, tiny_splits, wikitext):
    run_stage0("fp16", tmp_path, tiny_splits, wikitext, loader=tiny_loader)
    rows = [json.loads(line) for line in (tmp_path / "fp16" / "generations.jsonl").read_text().splitlines()]
    assert [r["split"] for r in rows] == ["harmful_eval"] * 4 + ["harmless_eval"] * 4
    assert math.isfinite(read_metrics(tmp_path / "fp16")["perplexity"])
    assert tiny_loader.calls == ["fp16"]


def test_stage0_skips_when_done(tmp_path, tiny_loader, tiny_splits, wikitext):
    run_stage0("rtn-4", tmp_path, tiny_splits, wikitext, loader=tiny_loader)
    before = (tmp_path / "rtn-4" / "generations.jsonl").read_text()
    run_stage0("rtn-4", tmp_path, tiny_splits, wikitext, loader=exploding_loader)
    assert (tmp_path / "rtn-4" / "generations.jsonl").read_text() == before


def test_stage0_resumes_missing_perplexity(tmp_path, tiny_loader, tiny_splits, wikitext):
    run_stage0("fp16", tmp_path, tiny_splits, wikitext, loader=tiny_loader)
    (tmp_path / "fp16" / "metrics.json").unlink()
    before = (tmp_path / "fp16" / "generations.jsonl").stat().st_mtime_ns
    run_stage0("fp16", tmp_path, tiny_splits, wikitext, loader=tiny_loader)
    assert "perplexity" in read_metrics(tmp_path / "fp16")
    assert (tmp_path / "fp16" / "generations.jsonl").stat().st_mtime_ns == before


def test_stage0_nonfinite_perplexity_is_recorded_and_not_recomputed(tmp_path, tiny_loader, tiny_splits, wikitext, monkeypatch):
    monkeypatch.setattr("qsi.pipeline.perplexity", lambda *a, **k: float("inf"))
    run_stage0("rtn-2", tmp_path, tiny_splits, wikitext, loader=tiny_loader)
    m = read_metrics(tmp_path / "rtn-2")
    assert m["perplexity"] is None and m["nonfinite"] == {"perplexity": "inf"}
    run_stage0("rtn-2", tmp_path, tiny_splits, wikitext, loader=exploding_loader)


def test_stage0_rejects_unknown_config(tmp_path, tiny_splits, wikitext):
    with pytest.raises(ValueError):
        run_stage0("gguf", tmp_path, tiny_splits, wikitext, loader=exploding_loader)


def test_truncate_splits(tiny_splits):
    t = truncate_splits(tiny_splits, 2)
    assert all(len(v) == 2 for v in t.values()) and t["harmful_eval"] == tiny_splits["harmful_eval"][:2]
    assert truncate_splits(tiny_splits, None) == tiny_splits
