import math

import numpy as np
import pytest
import torch

from qsi.sae import (
    SAE_LAYERS,
    SAE_REPO,
    JumpReLUSAE,
    PairStats,
    fvu,
    nearest_sae_layer,
    pick_sae_path,
    reconstruction_stats,
    refusal_feature_idx,
    sae_transfer_ok,
)


def make_sae(d=4, k=6, threshold=0.5):
    """Feature i < d reads and writes input coordinate i; the remaining features never fire."""
    w_enc = torch.zeros(d, k)
    w_enc[:, :d] = torch.eye(d)
    return JumpReLUSAE(w_enc, torch.zeros(k), w_enc.T.clone(), torch.zeros(d), torch.full((k,), threshold))


def test_constants():
    assert SAE_REPO == "google/gemma-scope-2b-pt-res" and SAE_LAYERS == (6, 12, 20)


def test_encode_threshold():
    f = make_sae().encode(torch.tensor([[1.0, 0.4, -2.0, 0.6]]))
    assert f[0, :4].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.6]) and (f[0, 4:] == 0).all()


def test_encode_uses_bias_before_threshold():
    sae = make_sae()
    sae.b_enc.data[1] = 0.2
    assert sae.encode(torch.tensor([[0.0, 0.4, 0.0, 0.0]]))[0, 1].item() == pytest.approx(0.6)


def test_decode_roundtrip_and_fvu():
    sae = make_sae()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    assert fvu(x, sae.decode(sae.encode(x))) == pytest.approx(0.0, abs=1e-6)
    assert fvu(x, x.mean(0, keepdim=True).expand_as(x)) == pytest.approx(1.0)


def test_decode_adds_bias():
    sae = make_sae()
    sae.b_dec.data[:] = 1.0
    assert sae.decode(torch.zeros(1, 6)).tolist() == [[1.0] * 4]


def test_from_npz(tmp_path):
    sae = make_sae()
    sae.b_enc.data[2] = 0.3
    np.savez(tmp_path / "params.npz", **{k: v.detach().numpy() for k, v in sae.state_dict().items()})
    loaded = JumpReLUSAE.from_npz(tmp_path / "params.npz")
    x = torch.randn(5, 4)
    assert loaded.W_enc.shape == (4, 6) and loaded.W_dec.shape == (6, 4) and loaded.W_enc.dtype == torch.float32
    assert torch.equal(loaded.encode(x), sae.encode(x))


def test_reconstruction_stats_batched():
    torch.manual_seed(0)
    sae = make_sae()
    x = torch.rand(50, 4) + 1.0
    x[:, 3] = 0.1  # below threshold: never reconstructed
    f, l0 = reconstruction_stats(sae, x.half(), batch_size=7)
    full = sae.decode(sae.encode(x.half().float()))
    assert f == pytest.approx(fvu(x.half().float(), full), rel=1e-4) and l0 == pytest.approx(3.0)


def test_reconstruction_stats_nan_input():
    f, l0 = reconstruction_stats(make_sae(), torch.full((4, 4), float("nan")))
    assert math.isnan(f)


def test_pick_sae_path():
    files = [f"layer_12/width_16k/average_l0_{n}/params.npz" for n in (22, 82, 118, 176)]
    files += ["layer_12/width_65k/average_l0_100/params.npz", "layer_6/width_16k/average_l0_70/params.npz", "README.md",
              "layer_1/width_16k/average_l0_100/params.npz", "layer_120/width_16k/average_l0_100/params.npz"]
    assert pick_sae_path(files, 12) == "layer_12/width_16k/average_l0_82/params.npz"  # tie -> lower L0
    assert pick_sae_path(files, 12, target_l0=170) == "layer_12/width_16k/average_l0_176/params.npz"
    assert pick_sae_path(files, 12, width="65k") == "layer_12/width_65k/average_l0_100/params.npz"
    with pytest.raises(ValueError):
        pick_sae_path(files, 20)


def test_pairstats_identical_and_batched():
    torch.manual_seed(0)
    a = torch.relu(torch.randn(500, 8))
    s1 = PairStats(8)
    s1.update(a, a)
    s2 = PairStats(8)
    s2.update(a[:200], a[:200])
    s2.update(a[200:], a[200:])
    r = s1.summary()
    assert r["median_corr"] == pytest.approx(1.0) and r["frac_corr_gt_0.9"] == 1.0
    assert r["n_dead"] == r["n_new"] == 0 and r["n_eligible"] == 8
    assert s2.summary() == pytest.approx(r)


def test_pairstats_matches_numpy_corrcoef():
    torch.manual_seed(0)
    a = torch.relu(torch.randn(400, 3))
    b = torch.relu(a + 0.5 * torch.randn(400, 3))
    s = PairStats(3)
    s.update(a, b)
    expected = sorted(np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(3))[1]
    assert s.summary()["median_corr"] == pytest.approx(expected, rel=1e-5)


def test_pairstats_dead_new_and_filter():
    torch.manual_seed(0)
    a = torch.relu(torch.randn(500, 4))
    b = a.clone()
    b[:, 0] = 0  # feature 0 died
    a[:, 1] = 0  # feature 1 is new
    b[:, 2] = torch.relu(torch.randn(500))  # feature 2 decorrelated
    s = PairStats(4)
    s.update(a, b)
    r = s.summary()
    assert r["n_dead"] == 1 and r["n_new"] == 1 and r["n_eligible"] == 3
    assert r["frac_corr_gt_0.9"] == pytest.approx(1 / 3)
    only3 = s.summary(feature_idx=torch.tensor([3]))
    assert only3 == {"n_eligible": 1, "median_corr": pytest.approx(1.0), "frac_corr_gt_0.9": 1.0, "n_dead": 0, "n_new": 0}
    assert s.summary(feature_idx=torch.tensor([0, 1])) ["n_dead"] == 1


def test_pairstats_min_count():
    a = torch.zeros(100, 2)
    a[:9, 0] = 1.0  # fires 9 times: not eligible
    a[:10, 1] = 1.0  # fires 10 times: eligible
    s = PairStats(2)
    s.update(a, a)
    assert s.summary(min_count=10)["n_eligible"] == 1


def test_pairstats_no_eligible_features():
    s = PairStats(2)
    s.update(torch.zeros(5, 2), torch.zeros(5, 2))
    r = s.summary()
    assert r["n_eligible"] == 0 and math.isnan(r["median_corr"]) and math.isnan(r["frac_corr_gt_0.9"])


def test_refusal_feature_idx():
    sae = make_sae()
    harmless = torch.ones(10, 4)
    harmful = torch.ones(10, 4)
    harmful[:, 0] += 3.0
    harmful[:, 3] += 2.0
    idx = refusal_feature_idx(sae, harmful, harmless, top_k=2)
    assert idx.tolist() == [0, 3]


def test_nearest_sae_layer():
    assert [nearest_sae_layer(l) for l in (0, 9, 10, 16, 25)] == [6, 6, 12, 12, 20]
    assert nearest_sae_layer(1, sae_layers=(0, 2)) == 0


def test_transfer_ok():
    assert sae_transfer_ok({6: 0.6, 12: 0.4, 20: 0.9})
    assert not sae_transfer_ok({6: 0.6, 12: 0.51, 20: 0.9})
    assert not sae_transfer_ok({6: float("nan"), 12: float("nan")})
