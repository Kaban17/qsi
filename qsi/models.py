"""Configuration names and model loading."""

import gc
import re
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from qsi.quantize import quantize_model_rtn

MODEL_ID = "google/gemma-2-2b-it"
CONFIG_NAMES = ["fp16", "bnb-8bit", "bnb-nf4", "rtn-8", "rtn-6", "rtn-4", "rtn-3", "rtn-2"]


@dataclass(frozen=True)
class ConfigSpec:
    kind: Literal["fp16", "bnb", "rtn"]
    bits: int | None


def parse_config(name: str) -> ConfigSpec:
    if name == "fp16":
        return ConfigSpec("fp16", None)
    if name == "bnb-8bit":
        return ConfigSpec("bnb", 8)
    if name == "bnb-nf4":
        return ConfigSpec("bnb", 4)
    match = re.fullmatch(r"rtn-(\d+)", name)
    if match and int(match.group(1)) >= 2:
        return ConfigSpec("rtn", int(match.group(1)))
    raise ValueError(f"unknown configuration {name!r}; expected one of {CONFIG_NAMES} or rtn-<bits>")


def get_blocks(model) -> nn.ModuleList:
    """The transformer blocks of a Hugging Face Gemma causal LM."""
    return model.model.layers


def load_model(name: str, model_id: str = MODEL_ID):
    """Load `model_id` in the given configuration. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = parse_config(name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if spec.kind == "bnb":
        from transformers import BitsAndBytesConfig

        if spec.bits == 8:
            quant = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        # float16 on purpose: without it the unquantized parts load as bfloat16, which a T4 only
        # emulates, and bitsandbytes then casts (and logs a warning) on every matmul.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        if spec.kind == "rtn":
            quantize_model_rtn(get_blocks(model), spec.bits)
    return model.eval(), tokenizer


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
