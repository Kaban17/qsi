"""Reading and writing per-configuration result files."""

import json
import math
import os
from pathlib import Path

METRICS_FILE = "metrics.json"


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file so an interrupted run never leaves a partial result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_metrics(config_dir: Path) -> dict:
    path = Path(config_dir) / METRICS_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def _sanitize(value, path: str, nonfinite: dict):
    """Replace NaN/inf with None, recording the original value under its dotted path."""
    if isinstance(value, dict):
        return {k: _sanitize(v, f"{path}.{k}" if path else str(k), nonfinite) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v, f"{path}.{i}" if path else str(i), nonfinite) for i, v in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite[path] = str(value)
        return None
    return value


def _merge(base: dict, new: dict) -> dict:
    out = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _lookup(tree, path: str):
    for part in path.split("."):
        if isinstance(tree, dict) and part in tree:
            tree = tree[part]
        elif isinstance(tree, list) and part.isdigit() and int(part) < len(tree):
            tree = tree[int(part)]
        else:
            return ...
    return tree


def update_metrics(config_dir: Path, values: dict) -> dict:
    """Recursively merge `values` into metrics.json and return the merged result.

    Non-finite floats are stored as null; their original value is kept as a string
    in the top-level "nonfinite" dict under the dotted path of the key.
    """
    config_dir = Path(config_dir)
    existing = read_metrics(config_dir)
    old_nonfinite = existing.pop("nonfinite", {})
    new_nonfinite: dict = {}
    merged = _merge(existing, _sanitize(values, "", new_nonfinite))
    nonfinite = {p: v for p, v in old_nonfinite.items() if _lookup(merged, p) is None}
    nonfinite.update(new_nonfinite)
    if nonfinite:
        merged["nonfinite"] = nonfinite
    atomic_write_text(config_dir / METRICS_FILE, json.dumps(merged, indent=2, allow_nan=False) + "\n")
    return merged
