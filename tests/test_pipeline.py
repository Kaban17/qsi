import json
import math

import pytest
import torch

from qsi.pipeline import run_stage0, run_stage1, run_stage2, truncate_splits
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


def test_stage1_fp16_then_quant(tmp_path, tiny_loader, tiny_splits):
    run_stage1("fp16", tmp_path, tiny_splits, loader=tiny_loader)
    fp = read_metrics(tmp_path / "fp16")["refusal"]
    assert set(fp) == {"layer", "baseline_refusal_rate", "ablated_refusal_rate_per_layer", "cos_selected", "cos_per_layer", "auroc"}
    assert fp["cos_selected"] == pytest.approx(1.0) and len(fp["ablated_refusal_rate_per_layer"]) == 3
    run_stage1("rtn-4", tmp_path, tiny_splits, loader=tiny_loader)
    m = read_metrics(tmp_path / "rtn-4")["refusal"]
    assert set(m) == {"layer", "cos_selected", "cos_per_layer", "auroc"}
    assert m["layer"] == fp["layer"] and len(m["cos_per_layer"]) == 3
    # the tiny loader returns the same weights for both names, so directions must coincide
    assert m["cos_selected"] == pytest.approx(1.0, abs=1e-4) and m["auroc"] == pytest.approx(fp["auroc"])
    d = torch.load(tmp_path / "rtn-4" / "directions.pt")
    assert d.shape == (3, 32) and d.dtype == torch.float32


def test_stage1_requires_fp16(tmp_path, tiny_splits):
    with pytest.raises(RuntimeError, match="fp16 first"):
        run_stage1("rtn-4", tmp_path, tiny_splits, loader=exploding_loader)


def test_stage1_skips_when_done(tmp_path, tiny_loader, tiny_splits):
    run_stage1("fp16", tmp_path, tiny_splits, loader=tiny_loader)
    run_stage1("fp16", tmp_path, tiny_splits, loader=exploding_loader)


def test_stage1_nan_activations_do_not_crash(tmp_path, tiny_loader, tiny_splits, tiny_model):
    run_stage1("fp16", tmp_path, tiny_splits, loader=tiny_loader)
    for p in tiny_model.parameters():
        p.data.fill_(float("nan"))
    run_stage1("rtn-2", tmp_path, tiny_splits, loader=tiny_loader)
    m = read_metrics(tmp_path / "rtn-2")
    assert m["refusal"]["auroc"] is None and m["refusal"]["cos_selected"] is None
    assert m["nonfinite"]["refusal.auroc"] == "nan"


@pytest.fixture
def sae_env(monkeypatch):
    """Two SAE layers that exist in the tiny model, and a random SAE in place of Gemma Scope."""
    from qsi.sae import JumpReLUSAE

    monkeypatch.setattr("qsi.sae.SAE_LAYERS", (0, 2))

    def sae_loader(layer):
        g = torch.Generator().manual_seed(layer)
        w = torch.randn(32, 16, generator=g) / 4
        return JumpReLUSAE(w, torch.zeros(16), w.T.clone(), torch.zeros(32), torch.zeros(16))

    return sae_loader


def test_stage2_fp16_then_quant(tmp_path, tiny_loader, tiny_splits, wikitext, sae_env):
    run_stage1("fp16", tmp_path, tiny_splits, loader=tiny_loader)
    run_stage2("fp16", tmp_path, tiny_splits, wikitext, loader=tiny_loader, sae_loader=sae_env)
    fp = read_metrics(tmp_path / "fp16")["sae"]
    assert set(fp) == {"transfer_ok", "refusal_sae_layer", "0", "2"} and set(fp["0"]) == {"fvu", "l0"}
    assert fp["refusal_sae_layer"] in (0, 2) and isinstance(fp["transfer_ok"], bool)
    assert (tmp_path / "fp16" / "resid_layer0.pt").exists() and (tmp_path / "fp16" / "resid_layer2.pt").exists()
    assert torch.load(tmp_path / "fp16" / "refusal_features.pt").shape == (16,)  # top_k capped at SAE width

    run_stage2("rtn-4", tmp_path, tiny_splits, wikitext, loader=tiny_loader, sae_loader=sae_env)
    m = read_metrics(tmp_path / "rtn-4")["sae"]
    assert set(m) == {"0", "2"}
    assert set(m["0"]) >= {"fvu", "l0", "n_eligible", "median_corr", "frac_corr_gt_0.9", "n_dead", "n_new"}
    assert "refusal_feats" in m[str(fp["refusal_sae_layer"])]
    other = "2" if fp["refusal_sae_layer"] == 0 else "0"
    assert "refusal_feats" not in m[other]
    # same weights under both names: features must match exactly
    assert m["0"]["median_corr"] == pytest.approx(1.0, abs=1e-3) and m["0"]["n_dead"] == 0
    assert m["0"]["fvu"] == pytest.approx(fp["0"]["fvu"], rel=1e-3)
    assert not (tmp_path / "rtn-4" / "resid_layer0.pt").exists()


def test_stage2_requires_fp16(tmp_path, tiny_loader, tiny_splits, wikitext, sae_env):
    run_stage1("fp16", tmp_path, tiny_splits, loader=tiny_loader)
    with pytest.raises(RuntimeError, match="stage 2 for fp16 first"):
        run_stage2("rtn-4", tmp_path, tiny_splits, wikitext, loader=exploding_loader, sae_loader=sae_env)


def test_stage2_fp16_requires_stage1(tmp_path, tiny_splits, wikitext, sae_env):
    with pytest.raises(RuntimeError, match="stage 1 for fp16 first"):
        run_stage2("fp16", tmp_path, tiny_splits, wikitext, loader=exploding_loader, sae_loader=sae_env)


def test_stage2_detects_mismatched_token_sets(tmp_path, tiny_loader, tiny_splits, wikitext, sae_env):
    run_stage1("fp16", tmp_path, tiny_splits, loader=tiny_loader)
    run_stage2("fp16", tmp_path, tiny_splits, wikitext, loader=tiny_loader, sae_loader=sae_env)
    with pytest.raises(RuntimeError, match="different set of tokens"):
        run_stage2("rtn-4", tmp_path, truncate_splits(tiny_splits, 2), wikitext, loader=tiny_loader, sae_loader=sae_env)


def test_stage2_skips_when_done(tmp_path, tiny_loader, tiny_splits, wikitext, sae_env):
    run_stage1("fp16", tmp_path, tiny_splits, loader=tiny_loader)
    run_stage2("fp16", tmp_path, tiny_splits, wikitext, loader=tiny_loader, sae_loader=sae_env)
    run_stage2("fp16", tmp_path, tiny_splits, wikitext, loader=exploding_loader, sae_loader=sae_env)
