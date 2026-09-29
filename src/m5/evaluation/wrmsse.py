"""Weighted Root Mean Squared Scaled Error, the official M5 accuracy metric.

For each of the 12 aggregation levels (total, state, store, category, department, ... , item-store):
  * aggregate the training history, actuals and forecasts by summing the series in each group;
  * scale_g   = mean over t of (y_t - y_{t-1})^2 on the training history, starting at the
                first non-zero observation of the aggregated series;
  * RMSSE_g   = sqrt( mean over the horizon of (actual - forecast)^2 / scale_g );
  * weight_g  = dollar sales (units * price) of the group over the last 28 training days,
                normalised to sum to 1 within the level;
  * level score = sum_g weight_g * RMSSE_g.
WRMSSE = mean of the 12 level scores. Lower is better; the naive seasonal forecast scores ~1.

Inputs are wide arrays (n_series x T) plus a metadata frame with the id columns, so the same
function scores LightGBM and PatchTST forecasts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVELS: list[list[str]] = [
    [],
    ["state_id"],
    ["store_id"],
    ["cat_id"],
    ["dept_id"],
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    ["item_id"],
    ["item_id", "state_id"],
    ["item_id", "store_id"],
]


@dataclass
class WRMSSEResult:
    total: float
    per_level: dict[str, float]

    def __str__(self) -> str:
        lines = [f"WRMSSE = {self.total:.4f}"]
        lines += [f"  {k:<22s} {v:.4f}" for k, v in self.per_level.items()]
        return "\n".join(lines)


def _group_sum(values: np.ndarray, meta: pd.DataFrame, keys: list[str]) -> np.ndarray:
    if not keys:
        return values.sum(axis=0, keepdims=True)
    codes = pd.MultiIndex.from_frame(meta[keys]).factorize()[0] if len(keys) > 1 else meta[keys[0]].factorize()[0]
    n_groups = codes.max() + 1
    out = np.zeros((n_groups, values.shape[1]), dtype=np.float64)
    np.add.at(out, codes, values)
    return out


def _group_sum_1d(values: np.ndarray, meta: pd.DataFrame, keys: list[str]) -> np.ndarray:
    return _group_sum(values[:, None], meta, keys)[:, 0]


def scale_factor(train: np.ndarray) -> np.ndarray:
    """Per-series mean squared one-step difference from the first non-zero observation."""
    n, t = train.shape
    first_nz = np.argmax(train != 0, axis=1)
    has_sales = (train != 0).any(axis=1)
    diffs = np.diff(train, axis=1) ** 2
    idx = np.arange(1, t)[None, :]
    mask = idx > first_nz[:, None]  # differences that start after the first non-zero
    counts = mask.sum(axis=1)
    scale = np.where(counts > 0, (diffs * mask).sum(axis=1) / np.maximum(counts, 1), np.nan)
    scale[~has_sales] = np.nan
    return scale


def wrmsse(
    train: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
    meta: pd.DataFrame,
    dollar_last28: np.ndarray,
) -> WRMSSEResult:
    """
    train         : (n_series, T_train) observed history used for scaling
    actual        : (n_series, H) truth over the horizon
    forecast      : (n_series, H)
    meta          : DataFrame with columns item_id, dept_id, cat_id, store_id, state_id (row-aligned)
    dollar_last28 : (n_series,) dollar sales over the last 28 training days (weights)
    """
    train = np.asarray(train, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    assert train.shape[0] == actual.shape[0] == forecast.shape[0] == len(meta) == len(dollar_last28)
    assert actual.shape == forecast.shape

    per_level = {}
    for keys in LEVELS:
        tr = _group_sum(train, meta, keys)
        ac = _group_sum(actual, meta, keys)
        fc = _group_sum(forecast, meta, keys)
        wt = _group_sum_1d(np.asarray(dollar_last28, dtype=np.float64), meta, keys)
        scale = scale_factor(tr)
        mse = ((ac - fc) ** 2).mean(axis=1)
        rmsse = np.sqrt(mse / scale)
        valid = np.isfinite(rmsse) & (wt > 0)
        w = wt[valid] / wt[valid].sum()
        name = "total" if not keys else "x".join(keys)
        per_level[name] = float((w * rmsse[valid]).sum())
    total = float(np.mean(list(per_level.values())))
    return WRMSSEResult(total=total, per_level=per_level)


def dollar_sales_last_days(
    sales_wide: np.ndarray, meta: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame,
    last_day: int, n_days: int = 28,
) -> np.ndarray:
    """Units x price over the last `n_days` observed days, per series."""
    days = np.arange(last_day - n_days + 1, last_day + 1)
    cal = calendar.set_index("day_idx").loc[days, "wm_yr_wk"]
    price_lookup = prices.set_index(["store_id", "item_id", "wm_yr_wk"])["sell_price"]
    dollars = np.zeros(len(meta))
    units = sales_wide[:, days - 1]
    for j, wk in enumerate(cal.to_numpy()):
        key = pd.MultiIndex.from_arrays([meta["store_id"], meta["item_id"], np.full(len(meta), wk)])
        p = price_lookup.reindex(key).to_numpy()
        dollars += units[:, j] * np.nan_to_num(p, nan=0.0)
    return dollars
