"""Per-configuration stages. Each loads one model, writes its files, and unloads it.

Every stage skips work whose output already exists, so an interrupted Colab
session can simply be re-run.
"""

from pathlib import Path

import torch
from torch import Tensor

from qsi.activations import last_token_resid
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
