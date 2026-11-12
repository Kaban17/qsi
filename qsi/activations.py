"""Reading the residual stream and removing a direction from it.

"Layer l" always means the output of transformer block l.
"""

from contextlib import contextmanager

import torch
from torch import Tensor

from qsi.models import get_blocks


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def _capture_blocks(model, layers: list[int]):
    """Collect the outputs of the given blocks during forward passes into a dict."""
    store: dict[int, Tensor] = {}
    blocks = get_blocks(model)
    handles = []

    def make_hook(layer):
        def hook(module, args, output):
            store[layer] = _hidden(output).detach()

        return hook

    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def last_token_resid(model, tokenizer, texts: list[str], batch_size: int = 16) -> Tensor:
    """Residual stream at the last token of each text, for every layer: [n, L, d] float32 on CPU.

    `texts` must already be chat-formatted (they contain BOS).
    """
    layers = list(range(len(get_blocks(model))))
    out = []
    with _capture_blocks(model, layers) as store:
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            # Left padding puts every sequence's last real token at position -1.
            out.append(torch.stack([store[l][:, -1].float().cpu() for l in layers], dim=1))
    return torch.cat(out)


@torch.no_grad()
def all_token_resid(
    model, input_ids: Tensor, attention_mask: Tensor, layers: list[int], batch_size: int = 8
) -> dict[int, Tensor]:
    """Residual stream at every real token except each row's first (BOS): {layer: [n_tokens, d]} float16 on CPU.

    Tokens are ordered by row, then by position.
    """
    out: dict[int, list[Tensor]] = {l: [] for l in layers}
    with _capture_blocks(model, layers) as store:
        for i in range(0, len(input_ids), batch_size):
            ids = input_ids[i : i + batch_size].to(model.device)
            mask = attention_mask[i : i + batch_size].to(model.device)
            model(input_ids=ids, attention_mask=mask)
            keep = mask.bool()
            first = mask.float().argmax(dim=1)  # index of the first real token in each row
            keep[torch.arange(len(ids), device=keep.device), first] = False
            for l in layers:
                out[l].append(store[l][keep].to(torch.float16).cpu())
    return {l: torch.cat(parts) for l, parts in out.items()}


@contextmanager
def ablate_direction(model, direction: Tensor):
    """Inside the context, project `direction` out of the residual stream everywhere.

    Hooks sit on the embedding output and on every block output, so nothing downstream
    can read the direction and no block can write it back.
    """
    unit = direction.detach().float()
    unit = unit / unit.norm()

    def project_out(x: Tensor) -> Tensor:
        u = unit.to(device=x.device, dtype=x.dtype)
        return x - (x @ u).unsqueeze(-1) * u

    def hook(module, args, output):
        if isinstance(output, tuple):
            return (project_out(output[0]), *output[1:])
        return project_out(output)

    handles = []
    try:
        handles.append(model.get_input_embeddings().register_forward_hook(hook))
        for block in get_blocks(model):
            handles.append(block.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
