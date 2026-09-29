import numpy as np
import pandas as pd
import pytest

from m5.evaluation.splits import rolling_origin_folds
from m5.evaluation.wrmsse import LEVELS, scale_factor, wrmsse


def _meta(n):
    return pd.DataFrame({
        "item_id": [f"I{i}" for i in range(n)],
        "dept_id": ["D0" if i % 2 else "D1" for i in range(n)],
        "cat_id": ["C0"] * n,
        "store_id": ["S0" if i < n // 2 else "S1" for i in range(n)],
        "state_id": ["CA"] * n,
    })


def test_folds_do_not_overlap():
    folds = rolling_origin_folds(last_day=1913, horizon=28, n_folds=3, gap=0)
    assert folds[0].valid_end == 1913 and folds[0].valid_start == 1886 and folds[0].train_end == 1885
    assert folds[1].valid_end == 1885 and folds[2].valid_end == 1857
    for f in folds:
        assert f.train_end < f.valid_start


def test_scale_factor_ignores_leading_zeros():
    y = np.array([[0, 0, 0, 2, 4, 2]], dtype=float)  # diffs after first non-zero: 2, -2 -> mean sq = 4
    assert np.isclose(scale_factor(y)[0], 4.0)


def test_perfect_forecast_is_zero_and_levels_complete():
    rng = np.random.default_rng(0)
    n, t, h = 8, 100, 28
    train = rng.poisson(3, size=(n, t)).astype(float)
    actual = rng.poisson(3, size=(n, h)).astype(float)
    res = wrmsse(train, actual, actual.copy(), _meta(n), np.ones(n))
    assert res.total == 0.0 and len(res.per_level) == len(LEVELS)


def test_naive_seasonal_scores_near_one():
    rng = np.random.default_rng(1)
    n, t, h = 50, 400, 28
    train = np.cumsum(rng.normal(size=(n, t)), axis=1) + 100  # random walks: last value is the best naive forecast
    actual = train[:, -1:] + np.cumsum(rng.normal(size=(n, h)), axis=1)
    forecast = np.repeat(train[:, -1:], h, axis=1)
    res = wrmsse(train, actual, forecast, _meta(n), rng.uniform(1, 5, n))
    assert 1.0 < res.total < 6.0  # random-walk error grows with horizon; must be O(1), not 0 nor huge
    assert res.per_level["item_idxstore_id"] == pytest.approx(res.per_level["item_id"])  # 1 store per item here
