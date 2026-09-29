"""Feature engineering in SQL with window functions.

Design
------
* Grain: one row per (series id, day). Rows for the 28 future days exist with sales = NULL.
* Direct multi-horizon setup: every sales-derived feature is shifted by at least `horizon`
  days (lags >= horizon, rolling windows ending `horizon` days back). A single model then
  produces all 28 forecast days with no recursion and no leakage: for the furthest target
  day, the most recent input is the last observed day.
* Price features are not shifted: M5 provides the price list for the forecast weeks, and in a
  retailer setting next month's prices are a decision, not an unknown.
* The SQL uses only LAG, AVG, MAX, MIN and COUNT window functions with ROWS frames, so the same
  text runs on DuckDB (production) and sqlite3 (tests).

The query is generated from the config (`lags`, `rolling_windows`, ...), so adding a feature is
a config change, and `feature_columns()` is the single place the model reads its input list from.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from m5.data.load import execute, run_sql

log = logging.getLogger(__name__)

CATEGORICAL = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type_1"]
KEY_COLS = ["id", "day_idx", "sales"]


def _shifted_frame(horizon: int, window: int) -> str:
    return f"ROWS BETWEEN {horizon + window - 1} PRECEDING AND {horizon} PRECEDING"


def build_feature_sql(cfg: dict[str, Any], min_day: int) -> str:
    h = int(cfg["data"]["horizon"])
    fc = cfg["features"]
    w = "PARTITION BY id ORDER BY day_idx"
    pw = int(fc["price_rolling_window"])

    lag_cols = [f"LAG(sales, {lag}) OVER ({w}) AS lag_{lag}" for lag in fc["lags"]]
    rmean_cols = [
        f"AVG(sales) OVER ({w} {_shifted_frame(h, win)}) AS rmean_{win}" for win in fc["rolling_windows"]
    ]
    # variance pieces; sqrt is taken in pandas (sqlite has no STDDEV / SQRT by default)
    rvar_cols = []
    for win in fc["rolling_std_windows"]:
        frame = _shifted_frame(h, win)
        rvar_cols.append(f"AVG(sales * sales) OVER ({w} {frame}) AS _m2_{win}")
        if win not in fc["rolling_windows"]:
            rvar_cols.append(f"AVG(sales) OVER ({w} {frame}) AS _m1_{win}")
    rmax_col = f"MAX(sales) OVER ({w} {_shifted_frame(h, 28)}) AS rmax_28"
    zero_share = (
        f"AVG(CASE WHEN sales = 0 THEN 1.0 ELSE 0.0 END) OVER ({w} {_shifted_frame(h, 28)}) AS zero_share_28"
    )

    price_cols = [
        "sell_price",
        "sell_price * 1.0 / MAX(sell_price) OVER (PARTITION BY id) AS price_rel_max",
        f"sell_price * 1.0 / AVG(sell_price) OVER ({w} ROWS BETWEEN {pw - 1} PRECEDING AND CURRENT ROW) AS price_rel_roll",
        f"sell_price * 1.0 / LAG(sell_price, 7) OVER ({w}) - 1.0 AS price_chg_7",
        "sell_price * 1.0 / AVG(sell_price) OVER (PARTITION BY dept_id, store_id, wm_yr_wk) AS price_rel_dept",
    ]

    release_col = "MIN(CASE WHEN sales > 0 THEN day_idx END) OVER (PARTITION BY id) AS first_sale_day"

    where = [f"b.day_idx >= {int(min_day)}"]
    sql = f"""
WITH base AS (
    SELECT
        s.id, s.item_id, s.dept_id, s.cat_id, s.store_id, s.state_id,
        s.day_idx, s.sales,
        c.wm_yr_wk, c.wday, c.month, c.year, c.dom, c.week_of_year, c.n_events, c.event_type_1,
        CASE s.state_id WHEN 'CA' THEN c.snap_CA WHEN 'TX' THEN c.snap_TX ELSE c.snap_WI END AS snap,
        p.sell_price
    FROM sales_long s
    JOIN calendar c ON c.day_idx = s.day_idx
    LEFT JOIN prices p
        ON p.store_id = s.store_id AND p.item_id = s.item_id AND p.wm_yr_wk = c.wm_yr_wk
),
feats AS (
    SELECT
        id, item_id, dept_id, cat_id, store_id, state_id, day_idx, sales,
        wday, month, year, dom, week_of_year, n_events, event_type_1, snap,
        {", ".join(lag_cols)},
        {", ".join(rmean_cols)},
        {", ".join(rvar_cols)},
        {rmax_col},
        {zero_share},
        {", ".join(price_cols)},
        {release_col}
    FROM base
)
SELECT b.* , b.day_idx - b.first_sale_day AS days_since_release
FROM feats b
WHERE {" AND ".join(where)}
"""
    return sql


def feature_columns(cfg: dict[str, Any]) -> list[str]:
    fc = cfg["features"]
    cols = ["wday", "month", "year", "dom", "week_of_year", "n_events", "snap"]
    cols += [f"lag_{lag}" for lag in fc["lags"]]
    cols += [f"rmean_{w}" for w in fc["rolling_windows"]]
    cols += [f"rstd_{w}" for w in fc["rolling_std_windows"]]
    cols += ["rmax_28", "zero_share_28"]
    cols += ["sell_price", "price_rel_max", "price_rel_roll", "price_chg_7", "price_rel_dept"]
    cols += ["days_since_release"]
    cols += CATEGORICAL
    return cols


def build_features_table(con: Any, cfg: dict[str, Any], last_day: int) -> None:
    """Materialise the `features` table for days >= last_day - min_train_days."""
    min_day = max(1, last_day - int(cfg["data"]["min_train_days"]) - max(cfg["features"]["lags"]))
    sql = build_feature_sql(cfg, min_day)
    execute(con, "DROP TABLE IF EXISTS features")
    execute(con, f"CREATE TABLE features AS {sql}")
    n = run_sql(con, "SELECT COUNT(*) AS n FROM features")["n"].iloc[0]
    log.info("features table: %d rows from day %d", n, min_day)


def _finalise(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Post-process a features query result: std from moments, dtypes, categoricals."""
    for win in cfg["features"]["rolling_std_windows"]:
        m1 = df[f"rmean_{win}"] if f"rmean_{win}" in df else df[f"_m1_{win}"]
        var = (df[f"_m2_{win}"] - m1 * m1).clip(lower=0)
        df[f"rstd_{win}"] = np.sqrt(var)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_m")])
    df["event_type_1"] = df["event_type_1"].fillna("none")
    for c in [c for c in df.columns if c.startswith(("lag_", "rmean_", "rstd_", "price", "sell_price", "rmax", "zero"))]:
        df[c] = df[c].astype(np.float32)
    return df


def load_features(
    con: Any,
    cfg: dict[str, Any],
    day_from: int,
    day_to: int,
    require_price: bool = True,
) -> pd.DataFrame:
    conds = [f"day_idx BETWEEN {int(day_from)} AND {int(day_to)}"]
    if require_price:
        conds.append("sell_price IS NOT NULL")
    if cfg["data"]["drop_before_release"]:
        conds.append("first_sale_day IS NOT NULL AND day_idx >= first_sale_day")
    df = run_sql(con, f"SELECT * FROM features WHERE {' AND '.join(conds)}")
    return _finalise(df, cfg)


# --------------------------------------------------------------------------- categorical encoding
class CategoryEncoder:
    """Stable string -> int codes, persisted so training and inference agree."""

    def __init__(self, mapping: dict[str, dict[str, int]] | None = None):
        self.mapping = mapping or {}

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "CategoryEncoder":
        for c in cols:
            values = sorted(df[c].astype(str).unique())
            self.mapping[c] = {v: i for i, v in enumerate(values)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c, m in self.mapping.items():
            df[c] = df[c].astype(str).map(m).fillna(-1).astype(np.int32)  # -1 = unseen
        return df

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.mapping, f)

    @classmethod
    def load(cls, path: str | Path) -> "CategoryEncoder":
        with open(path) as f:
            return cls(json.load(f))
