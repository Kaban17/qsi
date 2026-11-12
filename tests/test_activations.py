import pytest
import torch

from qsi.activations import ablate_direction, all_token_resid, last_token_resid
from qsi.generate import format_chat


def block_outputs(model, ids):
    """Reference: outputs of every block for one unpadded sequence, via plain hooks."""
    outs = []
    hooks = [
        b.register_forward_hook(lambda m, a, o: outs.append(o[0] if isinstance(o, tuple) else o))
        for b in model.model.layers
    ]
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in hooks:
            h.remove()
    return outs


def n_hooks(model):
    return sum(len(m._forward_hooks) for m in model.modules())


def test_last_token_matches_block_outputs(tiny_model, tiny_tokenizer):
    texts = [format_chat(tiny_tokenizer, t) for t in ["ab", "abcdef"]]
    r = last_token_resid(tiny_model, tiny_tokenizer, texts, batch_size=1)
    assert r.shape == (2, 3, 32) and r.dtype == torch.float32 and r.device.type == "cpu"
    ids = tiny_tokenizer(texts[1], return_tensors="pt", add_special_tokens=False).input_ids
    ref = block_outputs(tiny_model, ids)
    for layer in range(3):
        assert torch.allclose(r[1, layer], ref[layer][0, -1], atol=1e-5)
    hs = tiny_model(ids, output_hidden_states=True).hidden_states
    assert torch.allclose(r[1, 0], hs[1][0, -1], atol=1e-5)  # layer l == hidden_states[l + 1]


def test_last_token_ignores_padding(tiny_model, tiny_tokenizer):
    texts = [format_chat(tiny_tokenizer, t) for t in ["ab", "abcdef", "hello there"]]
    a = last_token_resid(tiny_model, tiny_tokenizer, texts, batch_size=1)
    b = last_token_resid(tiny_model, tiny_tokenizer, texts, batch_size=3)
    assert torch.allclose(a, b, atol=1e-4)


def test_last_token_no_double_bos(tiny_model, tiny_tokenizer):
    text = format_chat(tiny_tokenizer, "ab")
    r = last_token_resid(tiny_model, tiny_tokenizer, [text])
    ids = torch.tensor([[1] + [tiny_tokenizer.convert_tokens_to_ids(c) for c in "<ab>"]])
    assert torch.allclose(r[0, 2], block_outputs(tiny_model, ids)[2][0, -1], atol=1e-5)


def test_capture_leaves_no_hooks(tiny_model, tiny_tokenizer):
    last_token_resid(tiny_model, tiny_tokenizer, [format_chat(tiny_tokenizer, "ab")])
    assert n_hooks(tiny_model) == 0


def test_all_token_excludes_bos_and_padding(tiny_model, tiny_tokenizer):
    enc = tiny_tokenizer(["<bos><ab>", "<bos><abcd>"], return_tensors="pt", padding=True, add_special_tokens=False)
    out = all_token_resid(tiny_model, enc.input_ids, enc.attention_mask, layers=[0, 2], batch_size=1)
    assert set(out) == {0, 2}
    assert out[0].shape == (enc.attention_mask.sum().item() - 2, 32) and out[0].dtype == torch.float16
    # row order, then position order: the first 4 rows are tokens "<ab>" of sequence 0
    ids0 = tiny_tokenizer("<bos><ab>", return_tensors="pt", add_special_tokens=False).input_ids
    ref = block_outputs(tiny_model, ids0)[2][0, 1:]
    assert torch.allclose(out[2][:4].float(), ref, atol=1e-2)
    both = all_token_resid(tiny_model, enc.input_ids, enc.attention_mask, layers=[0, 2], batch_size=2)
    assert torch.allclose(out[2].float(), both[2].float(), atol=1e-2)


def test_ablation_zeroes_projection(tiny_model, tiny_tokenizer):
    torch.manual_seed(0)
    d = torch.randn(32)
    texts = [format_chat(tiny_tokenizer, "abc")]
    plain = last_token_resid(tiny_model, tiny_tokenizer, texts)
    assert (plain @ (d / d.norm())).abs().max() > 1e-3
    with ablate_direction(tiny_model, d):
        r = last_token_resid(tiny_model, tiny_tokenizer, texts)
    assert (r @ (d / d.norm())).abs().max() < 1e-4


def test_ablation_changes_generation_inputs_everywhere(tiny_model, tiny_tokenizer):
    """Every position, not just the last, is ablated."""
    torch.manual_seed(0)
    d = torch.randn(32)
    ids = tiny_tokenizer("<bos><abcd>", return_tensors="pt", add_special_tokens=False).input_ids
    with ablate_direction(tiny_model, d):
        outs = block_outputs(tiny_model, ids)
    assert all((o @ (d / d.norm())).abs().max() < 1e-4 for o in outs)


def test_hooks_removed_even_on_error(tiny_model, tiny_tokenizer):
    texts = [format_chat(tiny_tokenizer, "abc")]
    before = last_token_resid(tiny_model, tiny_tokenizer, texts)
    with pytest.raises(RuntimeError):
        with ablate_direction(tiny_model, torch.randn(32)):
            raise RuntimeError
    assert n_hooks(tiny_model) == 0
    assert torch.equal(before, last_token_resid(tiny_model, tiny_tokenizer, texts))


def test_ablation_handles_half_precision_direction(tiny_model, tiny_tokenizer):
    d = torch.randn(32).half()
    with ablate_direction(tiny_model, d):
        r = last_token_resid(tiny_model, tiny_tokenizer, [format_chat(tiny_tokenizer, "abc")])
    assert torch.isfinite(r).all()
