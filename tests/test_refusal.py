import math
from contextlib import contextmanager

import pytest
import torch

from qsi.refusal import cosine_per_layer, mean_diff_directions, projection_auroc, refusal_rate, select_layer


def test_recovers_planted_direction():
    torch.manual_seed(0)
    v = torch.nn.functional.normalize(torch.randn(16), dim=0)
    harmless = torch.randn(200, 4, 16)
    harmful = torch.randn(200, 4, 16) + 5 * v
    d = mean_diff_directions(harmful, harmless)
    assert d.shape == (4, 16) and torch.allclose(d.norm(dim=1), torch.ones(4), atol=1e-5)
    assert (d @ v).min() > 0.95


def test_direction_points_from_harmless_to_harmful():
    harmful, harmless = torch.ones(3, 1, 4), torch.zeros(5, 1, 4)
    assert torch.allclose(mean_diff_directions(harmful, harmless), torch.full((1, 4), 0.5))


def test_cosine_identity_and_sign():
    torch.manual_seed(0)
    d = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    assert torch.allclose(cosine_per_layer(d, d), torch.ones(4), atol=1e-5)
    assert torch.allclose(cosine_per_layer(d, -d), -torch.ones(4), atol=1e-5)
    assert cosine_per_layer(d, d).shape == (4,)


def test_auroc_extremes():
    torch.manual_seed(0)
    v = torch.zeros(8)
    v[0] = 1
    assert projection_auroc(torch.ones(50, 8), -torch.ones(50, 8), v) == 1.0
    assert projection_auroc(-torch.ones(50, 8), torch.ones(50, 8), v) == 0.0
    assert 0.4 < projection_auroc(torch.randn(500, 8), torch.randn(500, 8), v) < 0.6


def test_nan_inputs_give_nan():
    v = torch.zeros(8)
    v[0] = 1
    assert math.isnan(projection_auroc(torch.full((5, 8), float("nan")), torch.ones(5, 8), v))
    assert math.isnan(projection_auroc(torch.ones(5, 8), torch.ones(5, 8), torch.full((8,), float("nan"))))
    assert math.isnan(projection_auroc(torch.full((5, 8), float("inf")), torch.ones(5, 8), v))
    assert torch.isnan(cosine_per_layer(torch.full((2, 8), float("nan")), torch.ones(2, 8))).all()
    assert torch.isnan(mean_diff_directions(torch.full((2, 1, 4), float("nan")), torch.ones(2, 1, 4))).all()


def test_refusal_rate():
    assert refusal_rate(["I cannot do that", "Sure!", "", "I'm sorry"]) == 0.5
    assert math.isnan(refusal_rate([]))


def test_select_layer_picks_min_and_breaks_ties_low(monkeypatch):
    state = {"layer": None}
    directions = torch.eye(3, 32)

    @contextmanager
    def fake_ablate(model, direction):
        state["layer"] = int(direction.argmax())
        try:
            yield
        finally:
            state["layer"] = None

    def fake_generate(model, tokenizer, texts, max_new_tokens=128, batch_size=16):
        refuses = state["layer"] in (None, 0)
        return ["I cannot" if refuses else "Sure"] * len(texts)

    monkeypatch.setattr("qsi.refusal.ablate_direction", fake_ablate)
    monkeypatch.setattr("qsi.refusal.generate_texts", fake_generate)
    assert select_layer(None, None, directions, ["x"] * 4) == (1, 1.0, [1.0, 0.0, 0.0])


def test_select_layer_runs_on_real_model(tiny_model, tiny_tokenizer):
    layer, base, per = select_layer(tiny_model, tiny_tokenizer, torch.randn(3, 32), ["ab", "abc"], max_new_tokens=2)
    assert 0 <= layer < 3 and len(per) == 3 and 0.0 <= base <= 1.0
