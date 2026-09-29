"""Data validation for the three raw M5 files.

Validation runs before anything is loaded into DuckDB. Every check raises `DataValidationError`
with a message precise enough to act on; the pipeline never silently continues on bad input.
"""
from __future__ import annotations

import re

import pandas as pd

SALES_ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLS = [
    "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI",
]
PRICE_COLS = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
_D_RE = re.compile(r"^d_(\d+)$")


class DataValidationError(ValueError):
    pass


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: missing columns {missing}")


def day_columns(sales: pd.DataFrame) -> list[str]:
    cols = [c for c in sales.columns if _D_RE.match(c)]
    idx = sorted(int(_D_RE.match(c).group(1)) for c in cols)
    if not idx:
        raise DataValidationError("sales: no d_* columns found")
    if idx != list(range(1, len(idx) + 1)):
        raise DataValidationError("sales: d_* columns are not consecutive from d_1")
    return [f"d_{i}" for i in idx]


def validate_sales(sales: pd.DataFrame) -> None:
    _require_columns(sales, SALES_ID_COLS, "sales")
    if sales["id"].duplicated().any():
        raise DataValidationError("sales: duplicated id values")
    dcols = day_columns(sales)
    values = sales[dcols]
    if values.isna().any().any():
        raise DataValidationError("sales: NaN in daily sales")
    if (values < 0).any().any():
        raise DataValidationError("sales: negative daily sales")
    for a, b in [("dept_id", "cat_id"), ("store_id", "state_id")]:
        if (sales.groupby(a)[b].nunique() > 1).any():
            raise DataValidationError(f"sales: {a} maps to more than one {b}")


def validate_calendar(cal: pd.DataFrame, n_days_min: int) -> None:
    _require_columns(cal, CALENDAR_COLS, "calendar")
    d_idx = cal["d"].str.extract(r"^d_(\d+)$")[0].astype(float)
    if d_idx.isna().any() or (d_idx.to_numpy() != range(1, len(cal) + 1)).any():
        raise DataValidationError("calendar: d must run d_1..d_N without gaps, in order")
    if len(cal) < n_days_min:
        raise DataValidationError(f"calendar has {len(cal)} days but sales have {n_days_min}")
    dates = pd.to_datetime(cal["date"])
    if not (dates.diff().dropna() == pd.Timedelta(days=1)).all():
        raise DataValidationError("calendar: dates are not consecutive days")
    if not cal["wday"].between(1, 7).all():
        raise DataValidationError("calendar: wday outside 1..7")
    if not cal["wm_yr_wk"].is_monotonic_increasing:
        raise DataValidationError("calendar: wm_yr_wk must be non-decreasing")
    for c in ["snap_CA", "snap_TX", "snap_WI"]:
        if not cal[c].isin([0, 1]).all():
            raise DataValidationError(f"calendar: {c} must be 0/1")


def validate_prices(prices: pd.DataFrame, sales: pd.DataFrame) -> None:
    _require_columns(prices, PRICE_COLS, "prices")
    if prices[["store_id", "item_id", "wm_yr_wk"]].duplicated().any():
        raise DataValidationError("prices: duplicated (store_id, item_id, wm_yr_wk)")
    if (prices["sell_price"] <= 0).any() or prices["sell_price"].isna().any():
        raise DataValidationError("prices: sell_price must be positive and non-null")
    known = set(zip(sales["store_id"], sales["item_id"]))
    unknown = set(zip(prices["store_id"], prices["item_id"])) - known
    if unknown:
        raise DataValidationError(f"prices: {len(unknown)} (store,item) pairs not present in sales")


def validate_all(sales: pd.DataFrame, cal: pd.DataFrame, prices: pd.DataFrame) -> None:
    validate_sales(sales)
    validate_calendar(cal, n_days_min=len(day_columns(sales)))
    validate_prices(prices, sales)
