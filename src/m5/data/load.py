"""Load the raw M5 CSVs into a DuckDB database in long (tidy) format.

Tables created
--------------
sales_long(id, item_id, dept_id, cat_id, store_id, state_id, day_idx, sales)
    One row per series per day. `day_idx` is the integer from `d_<n>`. Rows for the
    `horizon` days after the last observed day are appended with sales = NULL so that the
    feature SQL can build the inputs for the forecast window in the same query.
calendar(day_idx, date, wm_yr_wk, wday, month, year, dom, week_of_year, event_type_1,
         event_type_2, n_events, snap_CA, snap_TX, snap_WI)
prices(store_id, item_id, wm_yr_wk, sell_price)

The wide-to-long melt is done in chunks of series to keep peak memory bounded
(the full dataset is ~59M rows).

`run_sql` is a tiny adapter so the same SQL can be executed against DuckDB (production) or
sqlite3 (unit tests without the duckdb dependency); the feature SQL is written in the subset of
window-function SQL both engines share.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from m5.data.validate import day_columns, validate_all

log = logging.getLogger(__name__)

SALES_LONG_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "day_idx", "sales"]


# --------------------------------------------------------------------------- engine adapter
def connect(db_path: str | Path):
    import duckdb  # imported lazily so tests can run without it

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def is_duckdb(con: Any) -> bool:
    return type(con).__module__.startswith("duckdb")


def run_sql(con: Any, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute a SELECT and return a DataFrame on either DuckDB or sqlite3."""
    if is_duckdb(con):
        return (con.execute(sql, params) if params else con.execute(sql)).df()
    return pd.read_sql_query(sql, con, params=params or {})


def execute(con: Any, sql: str) -> None:
    if is_duckdb(con):
        con.execute(sql)
    else:
        con.execute(sql)
        con.commit()


def write_table(con: Any, name: str, df: pd.DataFrame, append: bool = False) -> None:
    if is_duckdb(con):
        con.register("_tmp_df", df)
        if append:
            con.execute(f"INSERT INTO {name} SELECT * FROM _tmp_df")
        else:
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp_df")
        con.unregister("_tmp_df")
    else:
        df.to_sql(name, con, if_exists="append" if append else "replace", index=False)


# --------------------------------------------------------------------------- transforms
def prepare_calendar(cal: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "day_idx": cal["d"].str.slice(2).astype(int),
            "date": pd.to_datetime(cal["date"]).dt.strftime("%Y-%m-%d"),
            "wm_yr_wk": cal["wm_yr_wk"].astype(int),
            "wday": cal["wday"].astype(int),
            "month": cal["month"].astype(int),
            "year": cal["year"].astype(int),
            "dom": pd.to_datetime(cal["date"]).dt.day.astype(int),
            "week_of_year": pd.to_datetime(cal["date"]).dt.isocalendar().week.astype(int),
            "event_type_1": cal["event_type_1"].where(cal["event_type_1"].notna(), None),
            "event_type_2": cal["event_type_2"].where(cal["event_type_2"].notna(), None),
            "n_events": cal["event_name_1"].notna().astype(int) + cal["event_name_2"].notna().astype(int),
            "snap_CA": cal["snap_CA"].astype(int),
            "snap_TX": cal["snap_TX"].astype(int),
            "snap_WI": cal["snap_WI"].astype(int),
        }
    )
    return out


def melt_sales_chunk(chunk: pd.DataFrame, dcols: list[str]) -> pd.DataFrame:
    long = chunk.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=dcols,
        var_name="d",
        value_name="sales",
    )
    long["day_idx"] = long["d"].str.slice(2).astype(np.int32)
    long["sales"] = long["sales"].astype(np.float32)
    return long[SALES_LONG_COLS]


def future_rows(sales: pd.DataFrame, last_day: int, horizon: int) -> pd.DataFrame:
    meta = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]
    fut = meta.loc[meta.index.repeat(horizon)].reset_index(drop=True)
    fut["day_idx"] = np.tile(np.arange(last_day + 1, last_day + horizon + 1, dtype=np.int32), len(meta))
    fut["sales"] = np.nan
    return fut[SALES_LONG_COLS]


def load_raw(raw_dir: str | Path, sales_file: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    sales = pd.read_csv(raw_dir / sales_file)
    cal = pd.read_csv(raw_dir / "calendar.csv")
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    validate_all(sales, cal, prices)
    return sales, cal, prices


def build_database(
    con: Any,
    sales: pd.DataFrame,
    cal: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
    chunk_series: int = 5000,
) -> int:
    """Populate the database. Returns the last observed day index."""
    dcols = day_columns(sales)
    last_day = len(dcols)
    if last_day + horizon > len(cal):
        raise ValueError(
            f"calendar has {len(cal)} days, need {last_day + horizon} to cover the forecast horizon"
        )
    log.info("sales: %d series x %d days; horizon %d", len(sales), last_day, horizon)

    write_table(con, "calendar", prepare_calendar(cal))
    write_table(con, "prices", prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]].copy())

    first = True
    for start in range(0, len(sales), chunk_series):
        chunk = sales.iloc[start : start + chunk_series]
        write_table(con, "sales_long", melt_sales_chunk(chunk, dcols), append=not first)
        first = False
        log.info("melted series %d-%d", start, start + len(chunk))
    write_table(con, "sales_long", future_rows(sales, last_day, horizon), append=True)
    write_table(
        con,
        "meta",
        pd.DataFrame({"key": ["last_day", "horizon"], "value": [last_day, horizon]}),
    )
    if is_duckdb(con):
        # pandas encodes missing floats as NaN; make the future rows real SQL NULLs so that
        # `sales IS NULL` and the window aggregates behave.
        execute(con, "UPDATE sales_long SET sales = NULL WHERE isnan(sales)")
        execute(con, "CREATE INDEX IF NOT EXISTS sales_long_idx ON sales_long(id, day_idx)")
    return last_day


def read_meta(con: Any) -> dict[str, int]:
    df = run_sql(con, "SELECT key, value FROM meta")
    return {k: int(v) for k, v in zip(df["key"], df["value"])}


def load_wide(con: Any, day_to: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (meta, sales) with sales as a dense (n_series, day_to) float array, row-aligned with meta.

    Only integer/float columns are pulled for the big table (a dense series index instead of the
    string id), which keeps the transfer for the full dataset at a few hundred MB.
    """
    meta = run_sql(
        con,
        "SELECT id, item_id, dept_id, cat_id, store_id, state_id FROM sales_long WHERE day_idx = 1 ORDER BY id",
    )
    df = run_sql(
        con,
        f"SELECT DENSE_RANK() OVER (ORDER BY id) - 1 AS sidx, day_idx, sales "
        f"FROM sales_long WHERE day_idx <= {int(day_to)}",
    )
    n = len(meta)
    wide = np.full((n, day_to), np.nan, dtype=np.float64)
    wide[df["sidx"].to_numpy(dtype=np.int64), df["day_idx"].to_numpy(dtype=np.int64) - 1] = df["sales"].to_numpy(dtype=np.float64)
    if np.isnan(wide).any():
        raise ValueError("sales matrix has gaps: every series must have every day up to day_to")
    return meta, wide


def load_prices(con: Any) -> pd.DataFrame:
    return run_sql(con, "SELECT store_id, item_id, wm_yr_wk, sell_price FROM prices")


def load_calendar(con: Any) -> pd.DataFrame:
    return run_sql(con, "SELECT * FROM calendar ORDER BY day_idx")
