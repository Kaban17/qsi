"""Greedy batched generation."""

import json
from pathlib import Path

import torch

from qsi.data import Prompt
from qsi.results import atomic_write_text


def format_chat(tokenizer, text: str) -> str:
    """Wrap a user message in the model's chat template, ending where the reply starts."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_texts(model, tokenizer, texts: list[str], max_new_tokens: int = 128, batch_size: int = 16) -> list[str]:
    """Greedy replies to each user message. Returns only the newly generated text.

    If the GPU runs out of memory the batch size is halved and the batch retried.
    """
    out: list[str] = []
    i = 0
    while i < len(texts):
        batch = [format_chat(tokenizer, t) for t in texts[i : i + batch_size]]
        # The chat template already contains BOS, so the tokenizer must not add another.
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        try:
            ids = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size //= 2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        new_tokens = ids[:, enc.input_ids.shape[1] :]
        out.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        i += len(batch)
    return out


def generate_responses(
    model, tokenizer, prompts: list[Prompt], out_path: Path, max_new_tokens: int = 128, batch_size: int = 16
) -> bool:
    """Generate a reply to every prompt and write them as JSONL.

    Returns False without doing anything if `out_path` already exists.
    """
    out_path = Path(out_path)
    if out_path.exists():
        return False
    responses = generate_texts(model, tokenizer, [p.text for p in prompts], max_new_tokens, batch_size)
    lines = [
        json.dumps({"id": p.id, "split": p.split, "prompt": p.text, "response": r}, ensure_ascii=False)
        for p, r in zip(prompts, responses, strict=True)
    ]
    atomic_write_text(out_path, "".join(line + "\n" for line in lines))
    return True
