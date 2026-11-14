"""Report figures. Everything here reads metrics.json files only; no model, no GPU."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qsi.models import CONFIG_NAMES
from qsi.results import read_metrics

# Roughly from least to most aggressive quantization.
PLOT_ORDER = ["fp16", "bnb-8bit", "rtn-8", "rtn-6", "bnb-nf4", "rtn-4", "rtn-3", "rtn-2"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def load_all_metrics(results_dir: Path) -> dict[str, dict]:
    """Metrics of every configuration that has results, in CONFIG_NAMES order."""
    results_dir = Path(results_dir)
    return {n: m for n in CONFIG_NAMES if (m := read_metrics(results_dir / n))}


def _get(tree: dict, *path):
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree


def summary_table(metrics: dict[str, dict], sae_layer: int = 12) -> dict[str, dict[str, float | None]]:
    """One row of headline numbers per configuration; None where a value is missing or not finite."""
    base_ppl = _get(metrics.get("fp16", {}), "perplexity")
    table = {}
    for name, m in metrics.items():
        ppl = m.get("perplexity")
        table[name] = {
            "perplexity_ratio": ppl / base_ppl if ppl is not None and base_ppl else None,
            "harmful_refusal": _get(m, "safety", "harmful_refusal"),
            "harmful_compliance": _get(m, "safety", "harmful_compliance"),
            "harmful_gibberish": _get(m, "safety", "harmful_gibberish"),
            "harmless_refusal": _get(m, "safety", "harmless_refusal"),
            "cos_selected": _get(m, "refusal", "cos_selected"),
            "auroc": _get(m, "refusal", "auroc"),
            "fvu": _get(m, "sae", str(sae_layer), "fvu"),
            "median_corr": _get(m, "sae", str(sae_layer), "median_corr"),
        }
    return table


def _names(metrics: dict) -> list[str]:
    return [n for n in PLOT_ORDER if n in metrics]


def _series(values: list) -> np.ndarray:
    """None becomes NaN, which matplotlib draws as a gap."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def _xticks(ax, names: list[str]) -> None:
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.6, len(names) - 0.4)


def _line(ax, names, values, color, label) -> None:
    ax.plot(range(len(names)), _series(values), color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)


def _legend(ax, **kwargs) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, **kwargs)


def _stacked_harmful(ax, metrics, names) -> None:
    parts = [("harmful_refusal", "отказ", BLUE), ("harmful_compliance", "выполнение", ORANGE),
             ("harmful_gibberish", "мусор", AQUA)]
    bottom = np.zeros(len(names))
    for key, label, color in parts:
        values = np.nan_to_num(_series([_get(metrics[n], "safety", key) for n in names]))
        ax.bar(range(len(names)), values, bottom=bottom, width=0.62, color=color, edgecolor=SURFACE,
               linewidth=1.5, label=label)
        for x, (v, b) in enumerate(zip(values, bottom)):
            if v >= 0.08:
                ax.text(x, b + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8, color="#ffffff")
        bottom += values
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def _save(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_summary(metrics: dict[str, dict], out_path: Path, sae_layer: int = 12) -> None:
    """Four stacked panels on a shared configuration axis: what breaks, and when."""
    names = _names(metrics)
    table = summary_table(metrics, sae_layer)
    col = lambda key: [table[n][key] for n in names]  # noqa: E731
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 11), sharex=True, constrained_layout=True)

    _style(axes[0], "Перплексия на WikiText-2 (отношение к fp16, лог. шкала)")
    _line(axes[0], names, col("perplexity_ratio"), BLUE, None)
    if np.isfinite(_series(col("perplexity_ratio"))).any():
        axes[0].set_yscale("log")
        axes[0].minorticks_off()

    _style(axes[1], "Ответы на вредные запросы (судья)")
    _stacked_harmful(axes[1], metrics, names)
    _legend(axes[1], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)

    _style(axes[2], "Направление отказа: сходство с fp16")
    _line(axes[2], names, col("cos_selected"), BLUE, "косинус с направлением fp16")
    _line(axes[2], names, col("auroc"), ORANGE, "AUROC разделения вредных и безобидных")
    _legend(axes[2], loc="lower left")

    fvus = _series(col("fvu"))
    off_scale = bool((fvus[np.isfinite(fvus)] > 1).any())
    # A broken model drives FVU far above 1; left unclipped it flattens every other point on the panel.
    _style(axes[3], f"SAE-фичи, слой {sae_layer}" + (" (FVU выше 1 — за пределами шкалы)" if off_scale else ""))
    axes[3].set_ylim(-0.03, 1.05)
    _line(axes[3], names, col("fvu"), BLUE, "необъяснённая дисперсия (FVU)")
    _line(axes[3], names, col("median_corr"), ORANGE, "медианная корреляция фич с fp16")
    _legend(axes[3], loc="center left")

    _xticks(axes[3], names)
    _save(fig, out_path)


def plot_safety(metrics: dict[str, dict], out_path: Path) -> None:
    names = _names(metrics)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True, constrained_layout=True)
    _style(axes[0], "Ответы на вредные запросы (судья)")
    _stacked_harmful(axes[0], metrics, names)
    _legend(axes[0], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)

    _style(axes[1], "Доля отказов: судья и подстрочная метрика")
    _line(axes[1], names, [_get(metrics[n], "safety", "harmful_refusal") for n in names], BLUE,
          "вредные, судья")
    _line(axes[1], names, [_get(metrics[n], "safety", "harmful_refusal_substring") for n in names], ORANGE,
          "вредные, подстроки")
    _line(axes[1], names, [_get(metrics[n], "safety", "harmless_refusal") for n in names], AQUA,
          "безобидные, судья (ложные отказы)")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _legend(axes[1], loc="center left")
    _xticks(axes[1], names)
    _save(fig, out_path)


def plot_refusal(metrics: dict[str, dict], out_path: Path) -> None:
    names = _names(metrics)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10), constrained_layout=True)

    _style(axes[0], "Направление отказа: сходство с fp16 на выбранном слое")
    _line(axes[0], names, [_get(metrics[n], "refusal", "cos_selected") for n in names], BLUE, "косинус")
    _line(axes[0], names, [_get(metrics[n], "refusal", "auroc") for n in names], ORANGE, "AUROC")
    _legend(axes[0], loc="lower left")
    _xticks(axes[0], names)

    rows = [(n, _get(metrics[n], "refusal", "cos_per_layer")) for n in names]
    rows = [(n, r) for n, r in rows if r]
    ax = axes[1]
    _style(ax, "Косинус с направлением fp16 по слоям")
    ax.grid(False)
    if rows:
        grid = np.array([_series(r) for _, r in rows])
        image = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(len(rows)), [n for n, _ in rows])
        ax.set_xlabel("слой", fontsize=9, color=INK_SECONDARY)
        fig.colorbar(image, ax=ax, pad=0.01).outline.set_visible(False)

    ax = axes[2]
    _style(ax, "fp16: доля отказов при абляции направления каждого слоя")
    per_layer = _get(metrics.get("fp16", {}), "refusal", "ablated_refusal_rate_per_layer")
    baseline = _get(metrics.get("fp16", {}), "refusal", "baseline_refusal_rate")
    if per_layer:
        ax.plot(range(len(per_layer)), _series(per_layer), color=BLUE, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, label="с абляцией")
        if baseline is not None:
            ax.axhline(baseline, color=MUTED, linewidth=1.2, linestyle="--", label="без абляции")
        chosen = _get(metrics["fp16"], "refusal", "layer")
        if chosen is not None and per_layer[chosen] is not None:
            ax.annotate(f"выбран слой {chosen}", (chosen, per_layer[chosen]), textcoords="offset points",
                        xytext=(8, 12), fontsize=9, color=INK_SECONDARY)
        ax.set_xlabel("слой", fontsize=9, color=INK_SECONDARY)
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        _legend(ax, loc="best")
    _save(fig, out_path)


def plot_sae(metrics: dict[str, dict], out_path: Path) -> None:
    names = _names(metrics)
    layers = sorted({int(k) for m in metrics.values() for k in m.get("sae", {}) if k.isdigit()})
    colors = [BLUE, ORANGE, AQUA]
    panels = [("fvu", "Необъяснённая дисперсия реконструкции (FVU)"),
              ("median_corr", "Медианная корреляция активаций фич с fp16"),
              ("n_dead", "Число умерших фич")]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        _style(ax, title)
        for layer, color in zip(layers, colors):
            _line(ax, names, [_get(metrics[n], "sae", str(layer), key) for n in names], color, f"слой {layer}")
        _legend(ax, loc="best")
    axes[0].set_title("Необъяснённая дисперсия реконструкции (FVU, лог. шкала)", loc="left", fontsize=11, color=INK, pad=8)
    axes[0].set_yscale("log")
    axes[0].set_yticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
    axes[0].minorticks_off()
    _xticks(axes[2], names)
    _save(fig, out_path)
