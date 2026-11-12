"""Gemma Scope JumpReLU sparse autoencoders and feature-preservation metrics."""

import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_LAYERS = (6, 12, 20)


class JumpReLUSAE(nn.Module):
    """features = relu(x @ W_enc + b_enc), zeroed where the pre-activation is below a per-feature threshold."""

    def __init__(self, W_enc: Tensor, b_enc: Tensor, W_dec: Tensor, b_dec: Tensor, threshold: Tensor):
        super().__init__()
        self.W_enc = nn.Parameter(W_enc.float(), requires_grad=False)  # [d, k]
        self.b_enc = nn.Parameter(b_enc.float(), requires_grad=False)  # [k]
        self.W_dec = nn.Parameter(W_dec.float(), requires_grad=False)  # [k, d]
        self.b_dec = nn.Parameter(b_dec.float(), requires_grad=False)  # [d]
        self.threshold = nn.Parameter(threshold.float(), requires_grad=False)  # [k]

    @classmethod
    def from_npz(cls, path: Path) -> "JumpReLUSAE":
        params = np.load(path)
        return cls(*(torch.from_numpy(params[k]) for k in ("W_enc", "b_enc", "W_dec", "b_dec", "threshold")))

    @property
    def n_features(self) -> int:
        return self.W_enc.shape[1]

    def encode(self, x: Tensor) -> Tensor:
        pre = x @ self.W_enc + self.b_enc
        return torch.relu(pre) * (pre > self.threshold)

    def decode(self, features: Tensor) -> Tensor:
        return features @ self.W_dec + self.b_dec


def pick_sae_path(repo_files: list[str], layer: int, width: str = "16k", target_l0: int = 100) -> str:
    """Among a layer's SAEs of the given width, the one whose average L0 is closest to the target (ties: lower)."""
    pattern = re.compile(rf"layer_{layer}/width_{re.escape(width)}/average_l0_(\d+)/params\.npz")
    candidates = [(int(m.group(1)), f) for f in repo_files if (m := pattern.fullmatch(f))]
    if not candidates:
        raise ValueError(f"no SAE for layer {layer}, width {width} in the repository listing")
    return min(candidates, key=lambda c: (abs(c[0] - target_l0), c[0]))[1]


def load_gemma_scope_sae(layer: int) -> JumpReLUSAE:
    from huggingface_hub import hf_hub_download, list_repo_files

    path = pick_sae_path(list_repo_files(SAE_REPO), layer)
    sae = JumpReLUSAE.from_npz(hf_hub_download(SAE_REPO, path))
    return sae.to("cuda" if torch.cuda.is_available() else "cpu")


def fvu(x: Tensor, x_hat: Tensor) -> float:
    """Fraction of variance unexplained: 0 is perfect reconstruction, 1 is as bad as predicting the mean."""
    x, x_hat = x.double(), x_hat.double()
    return ((x - x_hat).pow(2).sum() / (x - x.mean(dim=0)).pow(2).sum()).item()


def _device(sae: JumpReLUSAE) -> torch.device:
    return sae.W_enc.device


@torch.no_grad()
def reconstruction_stats(sae: JumpReLUSAE, x: Tensor, batch_size: int = 4096) -> tuple[float, float]:
    """(FVU, mean L0) of the SAE on activations x [n, d], computed in batches."""
    mean = x.double().mean(dim=0)
    total_var = (x.double() - mean).pow(2).sum().item()
    sq_err = 0.0
    active = 0
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].float().to(_device(sae))
        feats = sae.encode(xb)
        sq_err += (xb - sae.decode(feats)).double().pow(2).sum().item()
        active += (feats > 0).sum().item()
    fvu_value = sq_err / total_var if total_var > 0 or math.isnan(total_var) else math.nan
    return fvu_value, active / len(x)


class PairStats:
    """Streaming comparison of the same features on two models (a = reference, b = quantized)."""

    def __init__(self, k: int):
        z = lambda: torch.zeros(k, dtype=torch.float64)  # noqa: E731
        self.n = 0
        self.sum_a, self.sum_b, self.sum_aa, self.sum_bb, self.sum_ab = z(), z(), z(), z(), z()
        self.count_a = torch.zeros(k, dtype=torch.long)
        self.count_b = torch.zeros(k, dtype=torch.long)

    def update(self, fa: Tensor, fb: Tensor) -> None:
        """Add a batch of feature activations [n, k] for the same tokens on both models."""
        a, b = fa.double(), fb.double()  # reduce on the inputs' device; only [k]-sized sums move to CPU
        self.n += len(a)
        self.sum_a += a.sum(0).cpu()
        self.sum_b += b.sum(0).cpu()
        self.sum_aa += (a * a).sum(0).cpu()
        self.sum_bb += (b * b).sum(0).cpu()
        self.sum_ab += (a * b).sum(0).cpu()
        self.count_a += (a > 0).sum(0).cpu()
        self.count_b += (b > 0).sum(0).cpu()

    def summary(self, min_count: int = 10, feature_idx: Tensor | None = None) -> dict:
        """Feature preservation among features that fired at least `min_count` times on the reference.

        A feature is "dead" if it fired on the reference and never on the quantized model,
        "new" if it never fired on the reference but fires on the quantized model.
        """
        idx = torch.arange(len(self.sum_a)) if feature_idx is None else feature_idx.cpu().long()
        n = max(self.n, 1)
        mean_a, mean_b = self.sum_a / n, self.sum_b / n
        var_a = self.sum_aa / n - mean_a**2
        var_b = self.sum_bb / n - mean_b**2
        cov = self.sum_ab / n - mean_a * mean_b
        corr = cov / torch.sqrt(var_a * var_b)
        corr = torch.where(var_b <= 1e-12, torch.zeros_like(corr), corr)  # constant on b: nothing preserved

        eligible = (self.count_a >= min_count)[idx]
        corr = corr[idx][eligible]
        count_a, count_b = self.count_a[idx], self.count_b[idx]
        n_eligible = int(eligible.sum())
        return {
            "n_eligible": n_eligible,
            "median_corr": corr.median().item() if n_eligible else math.nan,
            "frac_corr_gt_0.9": (corr > 0.9).double().mean().item() if n_eligible else math.nan,
            "n_dead": int(((count_a >= min_count) & (count_b == 0)).sum()),
            "n_new": int(((count_a == 0) & (count_b >= min_count)).sum()),
        }


@torch.no_grad()
def refusal_feature_idx(sae: JumpReLUSAE, harmful_last: Tensor, harmless_last: Tensor, top_k: int = 50) -> Tensor:
    """The features whose mean activation is most higher on harmful than on harmless prompts."""
    dev = _device(sae)
    diff = sae.encode(harmful_last.float().to(dev)).mean(0) - sae.encode(harmless_last.float().to(dev)).mean(0)
    return diff.topk(min(top_k, sae.n_features)).indices.cpu()


def nearest_sae_layer(layer: int, sae_layers=SAE_LAYERS) -> int:
    return min(sae_layers, key=lambda l: (abs(l - layer), l))


def sae_transfer_ok(fvus: dict[int, float], threshold: float = 0.5) -> bool:
    """False only if the SAE fails (FVU above the threshold, or not a number) on every layer."""
    return any(f <= threshold for f in fvus.values())
