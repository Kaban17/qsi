import pytest
import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 <>.,!?'"
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    config = Gemma2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attn_implementation="eager",
    )
    return Gemma2ForCausalLM(config).eval()


@pytest.fixture
def tiny_tokenizer():
    vocab = {tok: i for i, tok in enumerate(SPECIALS + list(CHARS))}
    assert len(vocab) <= 64
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tok.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        padding_side="left",
    )
    fast.chat_template = "{{ bos_token }}{% for m in messages %}<{{ m['content'] }}>{% endfor %}"
    return fast


@pytest.fixture
def tiny_loader(tiny_model, tiny_tokenizer):
    """Stands in for qsi.models.load_model; records which configurations were loaded."""
    calls = []

    def loader(name):
        calls.append(name)
        return tiny_model, tiny_tokenizer

    loader.calls = calls
    return loader


@pytest.fixture
def tiny_splits():
    from qsi.data import SPLIT_SIZES, Prompt

    words = ["abc", "hello", "xy", "test me"]
    return {
        split: [Prompt(f"{split}-{i:03d}", f"{w} {split[:8]} {i}", split) for i, w in enumerate(words)]
        for split in SPLIT_SIZES
    }
