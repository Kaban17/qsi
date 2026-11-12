"""Perplexity of a causal LM on fixed-length token chunks."""

import math

import torch
from torch import Tensor


@torch.no_grad()
def perplexity(model, chunks: Tensor, batch_size: int = 4) -> float:
    """exp(mean negative log-likelihood) over every predicted token of `chunks` [n, length].

    A broken model yields inf or nan rather than an exception.
    """
    total = 0.0
    count = 0
    for i in range(0, len(chunks), batch_size):
        ids = chunks[i : i + batch_size].to(model.device)
        logits = model(input_ids=ids).logits[:, :-1].float()
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum"
        )
        total += nll.double().item()
        count += ids[:, 1:].numel()
    mean = total / count
    try:
        return math.exp(mean)
    except OverflowError:
        return math.inf
