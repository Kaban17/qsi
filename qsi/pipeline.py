"""Per-configuration stages. Each loads one model, writes its files, and unloads it.

Every stage skips work whose output already exists, so an interrupted Colab
session can simply be re-run.
"""

from pathlib import Path

import torch
from torch import Tensor

from qsi import sae as sae_mod
from qsi.activations import all_token_resid, last_token_resid
from qsi.data import Prompt
from qsi.generate import format_chat, generate_responses
from qsi.models import free_memory, load_model, parse_config
from qsi.perplexity import perplexity
from qsi.refusal import cosine_per_layer, mean_diff_directions, projection_auroc, select_layer
from qsi.results import read_metrics, update_metrics

Splits = dict[str, list[Prompt]]


def truncate_splits(splits: Splits, limit: int | None) -> Splits:
    """The first `limit` prompts of every split (for smoke runs)."""
    return {name: prompts[:limit] for name, prompts in splits.items()}


def run_stage0(name: str, results_dir: Path, splits: Splits, wikitext: Tensor, loader=load_model) -> None:
    """Generate replies to the evaluation prompts and measure perplexity."""
    parse_config(name)
    config_dir = Path(results_dir) / name
    gen_path = config_dir / "generations.jsonl"
    need_gen = not gen_path.exists()
    need_ppl = "perplexity" not in read_metrics(config_dir)
    if not (need_gen or need_ppl):
        return
    model, tokenizer = loader(name)
    try:
        if need_gen:
            generate_responses(model, tokenizer, splits["harmful_eval"] + splits["harmless_eval"], gen_path)
        if need_ppl:
            update_metrics(config_dir, {"perplexity": perplexity(model, wikitext)})
    finally:
        del model
        free_memory()


def _last_token(model, tokenizer, prompts: list[Prompt]) -> Tensor:
    return last_token_resid(model, tokenizer, [format_chat(tokenizer, p.text) for p in prompts])


def run_stage1(name: str, results_dir: Path, splits: Splits, loader=load_model) -> None:
    """Refusal direction: find it (and its layer) on fp16, compare every other configuration against it."""
    parse_config(name)
    results_dir = Path(results_dir)
    config_dir = results_dir / name
    if "refusal" in read_metrics(config_dir):
        return
    is_reference = name == "fp16"
    if not is_reference:
        ref_path = results_dir / "fp16" / "directions.pt"
        ref_metrics = read_metrics(results_dir / "fp16").get("refusal")
        if ref_metrics is None or not ref_path.exists():
            raise RuntimeError(f"run stage 1 for fp16 first: {name} is compared against its refusal direction")

    model, tokenizer = loader(name)
    try:
        directions = mean_diff_directions(
            _last_token(model, tokenizer, splits["harmful_dir"]),
            _last_token(model, tokenizer, splits["harmless_dir"]),
        )
        harmful_eval = _last_token(model, tokenizer, splits["harmful_eval"])
        harmless_eval = _last_token(model, tokenizer, splits["harmless_eval"])
        if is_reference:
            layer, baseline, per_layer = select_layer(
                model, tokenizer, directions, [p.text for p in splits["harmful_val"]]
            )
            reference = directions
            extra = {"baseline_refusal_rate": baseline, "ablated_refusal_rate_per_layer": per_layer}
        else:
            layer = ref_metrics["layer"]
            reference = torch.load(ref_path)
            extra = {}
    finally:
        del model
        free_memory()

    cos = cosine_per_layer(directions, reference).tolist()
    config_dir.mkdir(parents=True, exist_ok=True)
    torch.save(directions.float(), config_dir / "directions.pt")
    update_metrics(config_dir, {"refusal": {
        "layer": layer,
        **extra,
        "cos_selected": cos[layer],
        "cos_per_layer": cos,
        "auroc": projection_auroc(harmful_eval[:, layer], harmless_eval[:, layer], reference[layer]),
    }})


def _token_resid(model, tokenizer, splits: Splits, wikitext: Tensor, layers: list[int]) -> dict[int, Tensor]:
    """Activations on every token of the evaluation prompts and of the WikiText chunks (BOS prepended)."""
    prompts = [format_chat(tokenizer, p.text) for p in splits["harmful_eval"] + splits["harmless_eval"]]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    from_prompts = all_token_resid(model, enc.input_ids, enc.attention_mask, layers)
    bos = torch.full((len(wikitext), 1), tokenizer.bos_token_id, dtype=wikitext.dtype)
    wiki_ids = torch.cat([bos, wikitext], dim=1)
    from_wiki = all_token_resid(model, wiki_ids, torch.ones_like(wiki_ids), layers)
    return {l: torch.cat([from_prompts[l], from_wiki[l]]) for l in layers}


@torch.no_grad()
def _pair_stats(sae, ref: Tensor, own: Tensor, batch_size: int = 4096) -> "sae_mod.PairStats":
    stats = sae_mod.PairStats(sae.n_features)
    device = sae.W_enc.device
    for i in range(0, len(ref), batch_size):
        stats.update(
            sae.encode(ref[i : i + batch_size].float().to(device)),
            sae.encode(own[i : i + batch_size].float().to(device)),
        )
    return stats


def run_stage2(
    name: str, results_dir: Path, splits: Splits, wikitext: Tensor, loader=load_model, sae_loader=None
) -> None:
    """SAE features: reconstruction quality everywhere, and per-feature agreement with fp16."""
    parse_config(name)
    sae_loader = sae_loader or sae_mod.load_gemma_scope_sae
    layers = list(sae_mod.SAE_LAYERS)
    results_dir = Path(results_dir)
    config_dir, ref_dir = results_dir / name, results_dir / "fp16"
    if "sae" in read_metrics(config_dir):
        return
    is_reference = name == "fp16"
    ref_refusal = read_metrics(ref_dir).get("refusal")
    if ref_refusal is None:
        raise RuntimeError("run stage 1 for fp16 first: stage 2 needs the refusal layer")
    refusal_sae_layer = sae_mod.nearest_sae_layer(ref_refusal["layer"], layers)
    feats_path = ref_dir / "refusal_features.pt"
    resid_paths = {l: ref_dir / f"resid_layer{l}.pt" for l in layers}
    if not is_reference and not (feats_path.exists() and all(p.exists() for p in resid_paths.values())):
        raise RuntimeError(f"run stage 2 for fp16 first: {name} is compared against its activations")

    model, tokenizer = loader(name)
    try:
        resid = _token_resid(model, tokenizer, splits, wikitext, layers)
        if is_reference:
            harmful_last = _last_token(model, tokenizer, splits["harmful_eval"])[:, refusal_sae_layer]
            harmless_last = _last_token(model, tokenizer, splits["harmless_eval"])[:, refusal_sae_layer]
    finally:
        del model
        free_memory()

    metrics: dict = {}
    for layer in layers:
        sae = sae_loader(layer)
        fvu, l0 = sae_mod.reconstruction_stats(sae, resid[layer])
        metrics[str(layer)] = {"fvu": fvu, "l0": l0}
        if is_reference:
            if layer == refusal_sae_layer:
                ref_dir.mkdir(parents=True, exist_ok=True)
                torch.save(sae_mod.refusal_feature_idx(sae, harmful_last, harmless_last), feats_path)
            torch.save(resid[layer], resid_paths[layer])
        else:
            ref = torch.load(resid_paths[layer])
            if ref.shape != resid[layer].shape:
                raise RuntimeError(
                    f"fp16 activations were computed on a different set of tokens "
                    f"({tuple(ref.shape)} vs {tuple(resid[layer].shape)}); re-run stage 2 for fp16 with the same prompts"
                )
            stats = _pair_stats(sae, ref, resid[layer])
            metrics[str(layer)].update(stats.summary())
            if layer == refusal_sae_layer:
                metrics[str(layer)]["refusal_feats"] = stats.summary(feature_idx=torch.load(feats_path))
        del sae
        free_memory()

    if is_reference:
        metrics["transfer_ok"] = sae_mod.sae_transfer_ok({l: metrics[str(l)]["fvu"] for l in layers})
        metrics["refusal_sae_layer"] = refusal_sae_layer
    update_metrics(config_dir, {"sae": metrics})
