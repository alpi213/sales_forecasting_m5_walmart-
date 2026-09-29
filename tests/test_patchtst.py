import numpy as np
import pytest

torch = pytest.importorskip("torch")

from m5.models.patchtst import PatchTST, PatchTSTConfig, tweedie_deviance
from m5.models.torch_train import WindowDataset, release_days


def test_forward_shape_and_positivity():
    cfg = PatchTSTConfig(lookback=64, horizon=28, patch_len=16, stride=8, d_model=32, n_heads=4, n_layers=1, d_ff=64, n_series=10)
    model = PatchTST(cfg).eval()
    x = torch.rand(5, 64) * 10
    out = model(x, torch.arange(5))
    assert out.shape == (5, 28) and (out > 0).all()
    with pytest.raises(ValueError):
        model(torch.rand(5, 63))


def test_tweedie_deviance_zero_at_truth():
    y = torch.tensor([0.0, 1.0, 5.0])
    assert tweedie_deviance(y, y.clamp_min(1e-6)) < 1e-4
    assert tweedie_deviance(y, torch.tensor([2.0, 2.0, 2.0])) > 0


def test_window_dataset_respects_release():
    y = np.zeros((2, 100))
    y[0, 10:] = 1
    y[1, 60:] = 1
    rel = release_days(y)
    ds = WindowDataset(y, rel, lookback=20, horizon=5, t_max=100, stride=1)
    idx = ds.index
    assert (idx[idx[:, 0] == 0, 1] >= 30).all() and (idx[idx[:, 0] == 1, 1] >= 80).all()
    xb, yb, i = ds.gather(torch.tensor([0, 1]))
    assert xb.shape == (2, 20) and yb.shape == (2, 5) and i.tolist() == [0, 0]
    t0 = int(idx[0, 1])
    assert torch.equal(xb[0], torch.tensor(y[0, t0 - 20 : t0], dtype=torch.float32))
    assert torch.equal(yb[0], torch.tensor(y[0, t0 : t0 + 5], dtype=torch.float32))


def test_one_training_step_reduces_loss():
    torch.manual_seed(0)
    cfg = PatchTSTConfig(lookback=32, horizon=4, patch_len=8, stride=4, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    model = PatchTST(cfg)
    x = torch.rand(64, 32) * 5
    y = x[:, -4:].clone()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = tweedie_deviance(y, model(x))
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]
