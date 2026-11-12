"""The refusal direction (Arditi et al., 2024): difference of mean activations on harmful vs harmless prompts."""

import math

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from qsi.activations import ablate_direction
from qsi.generate import generate_texts
from qsi.judge import is_refusal_substring


def mean_diff_directions(harmful: Tensor, harmless: Tensor) -> Tensor:
    """Unit direction per layer, pointing from harmless to harmful. Inputs [n, L, d] and [m, L, d]; output [L, d]."""
    diff = harmful.float().mean(dim=0) - harmless.float().mean(dim=0)
    return diff / diff.norm(dim=1, keepdim=True)


def cosine_per_layer(a: Tensor, b: Tensor) -> Tensor:
    """Cosine similarity between matching rows of two [L, d] direction sets."""
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1)


def projection_auroc(harmful: Tensor, harmless: Tensor, direction: Tensor) -> float:
    """How well the projection onto `direction` separates harmful [n, d] from harmless [m, d] activations.

    1.0 means every harmful prompt projects higher than every harmless one. Returns nan
    if any projection is not finite (a collapsed model), instead of raising.
    """
    scores = torch.cat([harmful.float() @ direction.float(), harmless.float() @ direction.float()])
    if not torch.isfinite(scores).all():
        return math.nan
    labels = [1] * len(harmful) + [0] * len(harmless)
    return float(roc_auc_score(labels, scores.numpy()))


def refusal_rate(texts: list[str]) -> float:
    """Share of texts containing a stock refusal phrase."""
    if not texts:
        return math.nan
    return sum(is_refusal_substring(t) for t in texts) / len(texts)


def select_layer(
    model, tokenizer, directions: Tensor, val_texts: list[str], max_new_tokens: int = 32
) -> tuple[int, float, list[float]]:
    """Find the layer whose direction, when ablated, suppresses refusals the most.

    Returns (best layer, refusal rate without ablation, refusal rate with each layer's
    direction ablated). Ties go to the lower layer.
    """
    baseline = refusal_rate(generate_texts(model, tokenizer, val_texts, max_new_tokens=max_new_tokens))
    per_layer = []
    for direction in directions:
        with ablate_direction(model, direction):
            replies = generate_texts(model, tokenizer, val_texts, max_new_tokens=max_new_tokens)
        per_layer.append(refusal_rate(replies))
    return per_layer.index(min(per_layer)), baseline, per_layer
