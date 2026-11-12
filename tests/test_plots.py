import matplotlib

matplotlib.use("Agg")

import pytest

from qsi.plots import PLOT_ORDER, load_all_metrics, plot_refusal, plot_sae, plot_safety, plot_summary, summary_table
from qsi.results import update_metrics


def full(ppl, refusal, cos, auroc, fvu, corr):
    return {
        "perplexity": ppl,
        "safety": {"harmful_refusal": refusal, "harmful_compliance": 1 - refusal - 0.1, "harmful_gibberish": 0.1,
                   "harmless_refusal": 0.02, "harmful_refusal_substring": refusal, "substring_agreement": 0.9},
        "refusal": {"layer": 1, "cos_selected": cos, "cos_per_layer": [cos, cos, cos], "auroc": auroc},
        "sae": {str(l): {"fvu": fvu, "l0": 80.0, "median_corr": corr, "frac_corr_gt_0.9": 0.8, "n_dead": 3, "n_new": 1,
                         "n_eligible": 100} for l in (6, 12, 20)},
    }


@pytest.fixture
def results(tmp_path):
    fp = full(10.0, 0.9, 1.0, 0.99, 0.2, None)
    fp["refusal"].update({"baseline_refusal_rate": 0.9, "ablated_refusal_rate_per_layer": [0.9, 0.1, 0.5]})
    for layer in ("6", "12", "20"):
        fp["sae"][layer] = {"fvu": 0.2, "l0": 80.0}
    update_metrics(tmp_path / "fp16", fp)
    update_metrics(tmp_path / "rtn-4", full(15.0, 0.7, 0.8, 0.95, 0.3, 0.85))
    # a collapsed configuration: perplexity is null, and stage 2 never ran
    broken = full(float("inf"), 0.0, float("nan"), float("nan"), 0.0, 0.0)
    del broken["sae"]
    update_metrics(tmp_path / "rtn-2", broken)
    (tmp_path / "not-a-config").mkdir()
    return tmp_path


def test_plot_order_covers_all_configs():
    from qsi.models import CONFIG_NAMES

    assert sorted(PLOT_ORDER) == sorted(CONFIG_NAMES)
    assert PLOT_ORDER == ["fp16", "bnb-8bit", "rtn-8", "rtn-6", "bnb-nf4", "rtn-4", "rtn-3", "rtn-2"]


def test_load_all_metrics_order_and_missing(results):
    assert list(load_all_metrics(results)) == ["fp16", "rtn-4", "rtn-2"]


def test_summary_table(results):
    t = summary_table(load_all_metrics(results))
    assert t["fp16"]["perplexity_ratio"] == 1.0 and t["rtn-4"]["perplexity_ratio"] == 1.5
    assert t["rtn-4"]["harmful_refusal"] == 0.7 and t["rtn-4"]["harmful_gibberish"] == 0.1
    assert t["rtn-4"]["cos_selected"] == 0.8 and t["rtn-4"]["auroc"] == 0.95
    assert t["rtn-4"]["fvu"] == 0.3 and t["rtn-4"]["median_corr"] == 0.85
    assert t["fp16"]["median_corr"] is None
    assert t["rtn-2"]["perplexity_ratio"] is None and t["rtn-2"]["cos_selected"] is None
    assert t["rtn-2"]["fvu"] is None and t["rtn-2"]["harmful_refusal"] == 0.0
    assert set(t["fp16"]) == {"perplexity_ratio", "harmful_refusal", "harmful_compliance", "harmful_gibberish",
                              "harmless_refusal", "cos_selected", "auroc", "fvu", "median_corr"}


def test_summary_table_without_fp16(tmp_path):
    update_metrics(tmp_path / "rtn-4", full(15.0, 0.7, 0.8, 0.95, 0.3, 0.85))
    assert summary_table(load_all_metrics(tmp_path))["rtn-4"]["perplexity_ratio"] is None


@pytest.mark.parametrize("plot", [plot_summary, plot_safety, plot_refusal, plot_sae])
def test_plots_write_png_despite_gaps(results, tmp_path, plot):
    out = tmp_path / "figs" / f"{plot.__name__}.png"
    plot(load_all_metrics(results), out)
    assert out.stat().st_size > 5000 and out.read_bytes()[:4] == b"\x89PNG"


@pytest.mark.parametrize("plot", [plot_summary, plot_safety, plot_refusal, plot_sae])
def test_plots_with_only_stage0(tmp_path, plot):
    update_metrics(tmp_path / "fp16", {"perplexity": 10.0})
    out = tmp_path / "x.png"
    plot(load_all_metrics(tmp_path), out)
    assert out.exists()
