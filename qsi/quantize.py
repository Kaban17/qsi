"""Round-to-nearest weight quantization ("fake quantization": values change, dtype does not)."""

import torch
from torch import Tensor, nn


def quantize_tensor_rtn(w: Tensor, bits: int) -> Tensor:
    """Symmetric per-row round-to-nearest quantization of a 2D weight matrix.

    Each row (output channel) gets its own scale, so its largest-magnitude weight
    is represented exactly and the rest snap to one of 2^bits - 1 levels.
    """
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    qmax = 2 ** (bits - 1) - 1
    w32 = w.float()
    scale = w32.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(w32 / scale), -qmax, qmax)
    return (q * scale).to(w.dtype)


@torch.no_grad()
def quantize_model_rtn(blocks: nn.ModuleList, bits: int) -> int:
    """Quantize the weight of every nn.Linear inside `blocks` in place. Returns how many."""
    count = 0
    for module in blocks.modules():
        if isinstance(module, nn.Linear):
            module.weight.copy_(quantize_tensor_rtn(module.weight, bits))
            count += 1
    return count
