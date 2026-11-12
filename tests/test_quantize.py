import pytest
import torch

from qsi.quantize import quantize_model_rtn, quantize_tensor_rtn


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_levels_per_row(bits):
    torch.manual_seed(0)
    q = quantize_tensor_rtn(torch.randn(8, 256), bits)
    assert all(row.unique().numel() <= 2**bits - 1 for row in q)


def test_error_decreases_with_bits():
    torch.manual_seed(0)
    w = torch.randn(16, 128)
    errs = [(quantize_tensor_rtn(w, b) - w).pow(2).mean().item() for b in (2, 3, 4, 6, 8)]
    assert errs == sorted(errs, reverse=True) and errs[0] > errs[-1]


def test_16_bits_is_near_identity():
    torch.manual_seed(0)
    w = torch.randn(4, 64)
    assert torch.allclose(quantize_tensor_rtn(w, 16), w, atol=1e-3)


def test_zero_row_and_dtype():
    torch.manual_seed(0)
    w = torch.randn(3, 8, dtype=torch.float16)
    w[1] = 0
    q = quantize_tensor_rtn(w, 4)
    assert q.dtype == torch.float16 and torch.isfinite(q).all() and (q[1] == 0).all()


def test_row_max_preserved():
    torch.manual_seed(0)
    w = torch.randn(4, 32)
    assert torch.allclose(quantize_tensor_rtn(w, 4).abs().amax(1), w.abs().amax(1))


def test_input_not_modified():
    w = torch.randn(4, 32)
    before = w.clone()
    quantize_tensor_rtn(w, 2)
    assert torch.equal(w, before)


@pytest.mark.parametrize("bits", [1, 0, -3])
def test_rejects_too_few_bits(bits):
    with pytest.raises(ValueError):
        quantize_tensor_rtn(torch.randn(2, 2), bits)


def test_model_only_block_linears(tiny_model):
    emb = tiny_model.get_input_embeddings().weight.clone()
    norm = tiny_model.model.norm.weight.clone()
    before = tiny_model.model.layers[0].mlp.up_proj.weight.clone()
    n = quantize_model_rtn(tiny_model.model.layers, 3)
    assert n == 3 * 7  # q, k, v, o + gate, up, down per block
    assert torch.equal(tiny_model.get_input_embeddings().weight, emb)
    assert torch.equal(tiny_model.model.norm.weight, norm)
    after = tiny_model.model.layers[0].mlp.up_proj.weight
    assert not torch.equal(after, before)
    assert all(row.unique().numel() <= 7 for row in after)
