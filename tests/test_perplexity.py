import math

import pytest
import torch

from qsi.perplexity import perplexity


def test_perplexity_matches_manual(tiny_model):
    torch.manual_seed(1)
    chunks = torch.randint(0, 64, (3, 10))
    logits = tiny_model(chunks).logits[:, :-1]
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), chunks[:, 1:].reshape(-1))
    assert perplexity(tiny_model, chunks, batch_size=2) == pytest.approx(nll.exp().item(), rel=1e-4)


def test_perplexity_independent_of_batch_size(tiny_model):
    chunks = torch.randint(0, 64, (5, 8))
    assert perplexity(tiny_model, chunks, batch_size=1) == pytest.approx(perplexity(tiny_model, chunks, batch_size=5), rel=1e-5)


def test_perplexity_nonfinite_does_not_raise(tiny_model):
    for p in tiny_model.parameters():
        p.data.fill_(float("nan"))
    assert math.isnan(perplexity(tiny_model, torch.randint(0, 64, (2, 6))))
