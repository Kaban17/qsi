"""Per-configuration stages. Each loads one model, writes its files, and unloads it.

Every stage skips work whose output already exists, so an interrupted Colab
session can simply be re-run.
"""

from pathlib import Path

from torch import Tensor

from qsi.data import Prompt
from qsi.generate import generate_responses
from qsi.models import free_memory, load_model, parse_config
from qsi.perplexity import perplexity
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
