"""Prompt sets (harmful / harmless) and WikiText chunks."""

import random
import re
from dataclasses import dataclass

from torch import Tensor

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"

SPLIT_SIZES = {
    "harmful_dir": 128,
    "harmful_val": 32,
    "harmful_eval": 100,
    "harmless_dir": 128,
    "harmless_eval": 100,
}


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    split: str


def normalize(text: str) -> str:
    """Canonical form used only to detect the same prompt across datasets."""
    return re.sub(r"\s+", " ", text).strip().lower().rstrip(".!?;:, ")


def _dedupe(texts: list[str], exclude: set[str]) -> list[str]:
    seen = set(exclude)
    out = []
    for text in texts:
        key = normalize(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _prompts(split: str, texts: list[str]) -> list[Prompt]:
    return [Prompt(f"{split}-{i:03d}", text, split) for i, text in enumerate(texts)]


def make_splits(advbench: list[str], jbb: list[str], alpaca: list[str], seed: int = 0) -> dict[str, list[Prompt]]:
    """Build disjoint prompt sets.

    JailbreakBench is the harmful evaluation set as is. AdvBench (minus anything that
    also appears in JailbreakBench) supplies the harmful prompts used to find the
    refusal direction and to pick its layer. Alpaca supplies the harmless prompts.
    """
    n = SPLIT_SIZES
    if len(jbb) != n["harmful_eval"] or len({normalize(t) for t in jbb}) != len(jbb):
        raise ValueError(f"JailbreakBench: expected {n['harmful_eval']} distinct prompts, got {len(jbb)}")
    rng = random.Random(seed)

    harmful = _dedupe(advbench, exclude={normalize(t) for t in jbb})
    need = n["harmful_dir"] + n["harmful_val"]
    if len(harmful) < need:
        raise ValueError(f"AdvBench: need {need} prompts after deduplication, got {len(harmful)}")
    rng.shuffle(harmful)

    harmless = _dedupe(alpaca, exclude=set())
    need = n["harmless_dir"] + n["harmless_eval"]
    if len(harmless) < need:
        raise ValueError(f"Alpaca: need {need} prompts after deduplication, got {len(harmless)}")
    rng.shuffle(harmless)

    a, b = n["harmful_dir"], n["harmful_dir"] + n["harmful_val"]
    c, d = n["harmless_dir"], n["harmless_dir"] + n["harmless_eval"]
    return {
        "harmful_dir": _prompts("harmful_dir", harmful[:a]),
        "harmful_val": _prompts("harmful_val", harmful[a:b]),
        "harmful_eval": _prompts("harmful_eval", list(jbb)),
        "harmless_dir": _prompts("harmless_dir", harmless[:c]),
        "harmless_eval": _prompts("harmless_eval", harmless[c:d]),
    }


def load_splits(seed: int = 0) -> dict[str, list[Prompt]]:
    """Download the source datasets and build the splits."""
    import pandas as pd
    from datasets import load_dataset

    advbench = pd.read_csv(ADVBENCH_URL)["goal"].tolist()
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")["Goal"]
    alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca = [row["instruction"] for row in alpaca_ds if row["input"] == ""]
    return make_splits(list(advbench), list(jbb), alpaca, seed=seed)


def load_wikitext_chunks(tokenizer, n_chunks: int = 40, chunk_len: int = 512) -> Tensor:
    """The first `n_chunks` consecutive `chunk_len`-token windows of the WikiText-2 test set."""
    from datasets import load_dataset

    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    ids = tokenizer("\n\n".join(rows), add_special_tokens=False, return_tensors="pt").input_ids[0]
    if len(ids) < n_chunks * chunk_len:
        raise ValueError(f"WikiText: need {n_chunks * chunk_len} tokens, got {len(ids)}")
    return ids[: n_chunks * chunk_len].reshape(n_chunks, chunk_len)
