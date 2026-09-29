"""Weekly panel for price-elasticity estimation.

Prices in M5 are weekly (one price per store-item-wm_yr_wk), so the natural unit of analysis
is the item-store-week. Aggregating to weeks also removes the day-of-week seasonality that is
irrelevant for the price question.

Panel columns
  item_id, dept_id, cat_id, store_id, state_id, wm_yr_wk, week_idx, units, price,
  snap_days, event_days, year, week_of_year,
  log_q = log1p(units), log_p = log(price),
  lag_log_q_{1,2,4}                (within item-store, previous kept weeks)
Only complete weeks (7 observed days) after the item's first sale are kept, and item-stores with
fewer than `min_weeks` weeks are dropped.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from m5.data.load import run_sql

log = logging.getLogger(__name__)

PANEL_SQL = """
SELECT
    s.item_id, s.dept_id, s.cat_id, s.store_id, s.state_id, c.wm_yr_wk,
    SUM(s.sales) AS units,
    MAX(p.sell_price) AS price,
    SUM(CASE s.state_id WHEN 'CA' THEN c.snap_CA WHEN 'TX' THEN c.snap_TX ELSE c.snap_WI END) AS snap_days,
    SUM(CASE WHEN c.n_events > 0 THEN 1 ELSE 0 END) AS event_days,
    MIN(c.year) AS year,
    MIN(c.week_of_year) AS week_of_year,
    COUNT(*) AS n_days
FROM sales_long s
JOIN calendar c ON c.day_idx = s.day_idx
JOIN prices p ON p.store_id = s.store_id AND p.item_id = s.item_id AND p.wm_yr_wk = c.wm_yr_wk
WHERE s.sales IS NOT NULL AND s.day_idx <= {last_day}
GROUP BY s.item_id, s.dept_id, s.cat_id, s.store_id, s.state_id, c.wm_yr_wk
HAVING COUNT(*) = 7
"""


def build_weekly_panel(con: Any, last_day: int, min_weeks: int) -> pd.DataFrame:
    df = run_sql(con, PANEL_SQL.format(last_day=int(last_day)))
    df = df.sort_values(["store_id", "item_id", "wm_yr_wk"]).reset_index(drop=True)
    df["week_idx"] = df["wm_yr_wk"].rank(method="dense").astype(int)
    df["series"] = df["store_id"] + "|" + df["item_id"]

    # keep weeks from the first week with sales onwards
    first = df[df["units"] > 0].groupby("series")["week_idx"].min().rename("first_week")
    df = df.join(first, on="series")
    df = df[df["week_idx"] >= df["first_week"]].drop(columns="first_week")

    df["log_q"] = np.log1p(df["units"].astype(float))
    df["log_p"] = np.log(df["price"].astype(float))
    g = df.groupby("series")
    for lag in (1, 2, 4):
        df[f"lag_log_q_{lag}"] = g["log_q"].shift(lag)
    df = df.dropna(subset=["lag_log_q_4"])

    counts = df.groupby("series")["week_idx"].transform("size")
    df = df[counts >= min_weeks].reset_index(drop=True)
    log.info("weekly panel: %d rows, %d item-stores, %d categories", len(df), df["series"].nunique(), df["cat_id"].nunique())
    return df
