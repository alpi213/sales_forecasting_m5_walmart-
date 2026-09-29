"""Generate a small dataset with the exact M5 schema.

Purpose: run the whole pipeline end-to-end (tests, CI, smoke runs) without the 400 MB Kaggle
download, and check that the elasticity estimators recover a *known* elasticity.

Schema produced (identical to the Kaggle files):
  sales_train_validation.csv : id,item_id,dept_id,cat_id,store_id,state_id,d_1..d_T
  calendar.csv               : date,wm_yr_wk,weekday,wday,month,year,d,event_name_1,event_type_1,
                               event_name_2,event_type_2,snap_CA,snap_TX,snap_WI
  sell_prices.csv            : store_id,item_id,wm_yr_wk,sell_price

Demand model (per item-store, day t):
  log lambda_t = log(base) + beta_cat * log(price_t / ref_price) + weekday effect
                 + event effect + snap effect + slow AR(1) noise
  sales_t ~ Poisson(lambda_t), forced to 0 before the item's release day.
The category-level elasticities `TRUE_ELASTICITY` are what the DML / Bayesian tests check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STATES = {"CA": ["CA_1", "CA_2"], "TX": ["TX_1"], "WI": ["WI_1"]}
CATS = {"FOODS": ["FOODS_1", "FOODS_2"], "HOBBIES": ["HOBBIES_1"], "HOUSEHOLD": ["HOUSEHOLD_1"]}
TRUE_ELASTICITY = {"FOODS": -1.5, "HOBBIES": -0.8, "HOUSEHOLD": -1.1}
WEEKDAY_EFFECT = np.array([0.25, 0.30, -0.10, -0.15, -0.15, -0.05, 0.10])  # Sat..Fri (M5 wday=1 is Saturday)
EVENTS = ["SuperBowl", "ValentinesDay", "Easter", "Mother's day", "Thanksgiving", "Christmas"]


def make_calendar(start: str, n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    # M5 weeks start on Saturday; wm_yr_wk is a Walmart week id (yyyww-like). We only need it
    # to be constant within a Sat..Fri week and monotone across weeks.
    wday = ((dates.dayofweek + 2) % 7) + 1  # Saturday -> 1 ... Friday -> 7
    week_idx = np.cumsum(wday == 1)
    week_idx = week_idx - week_idx[0]
    wm_yr_wk = 11101 + week_idx
    cal = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": wm_yr_wk,
            "weekday": dates.day_name(),
            "wday": wday,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i}" for i in range(1, n_days + 1)],
            "event_name_1": None,
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
        }
    )
    event_days = rng.choice(n_days, size=max(1, n_days // 45), replace=False)
    for i, day in enumerate(event_days):
        cal.loc[day, "event_name_1"] = EVENTS[i % len(EVENTS)]
        cal.loc[day, "event_type_1"] = "Cultural" if i % 2 else "National"
    for st in STATES:
        # SNAP days: first 10 days of the month, as in the real data
        cal[f"snap_{st}"] = (dates.day <= 10).astype(int)
    return cal


def generate(
    out_dir: str | Path,
    n_items_per_dept: int = 12,
    n_days: int = 500,
    start: str = "2011-01-29",
    seed: int = 0,
    extra_calendar_days: int = 56,
) -> dict[str, pd.DataFrame]:
    """Like the Kaggle files, the calendar and price list extend `extra_calendar_days` past the
    last observed sales day so the forecast window has prices and calendar features."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = make_calendar(start, n_days + extra_calendar_days, rng)
    n_weeks = int(cal["wm_yr_wk"].nunique())
    weeks = np.sort(cal["wm_yr_wk"].unique())
    day_week_pos = np.searchsorted(weeks, cal["wm_yr_wk"].to_numpy())

    stores = [(st, s) for st, ss in STATES.items() for s in ss]
    items = [
        (cat, dept, f"{dept}_{i:03d}") for cat, depts in CATS.items() for dept in depts for i in range(1, n_items_per_dept + 1)
    ]

    sales_rows, price_rows = [], []
    event_flag = cal["event_name_1"].notna().to_numpy()[:n_days]
    snap_by_state = {st: cal[f"snap_{st}"].to_numpy()[:n_days] for st in STATES}
    wday0 = cal["wday"].to_numpy()[:n_days] - 1
    day_week_pos_obs = day_week_pos[:n_days]

    for cat, dept, item in items:
        ref_price = float(np.exp(rng.normal(np.log(4.0), 0.6)))
        base_item = float(np.exp(rng.normal(np.log(2.0), 0.9)))
        beta = TRUE_ELASTICITY[cat]
        for st, store in stores:
            base = base_item * float(np.exp(rng.normal(0, 0.3)))
            release_week = int(rng.integers(0, max(1, n_weeks // 4)))
            # weekly price path: sticky, with occasional promotions and permanent changes
            price_w = np.full(n_weeks, ref_price)
            level = ref_price
            for w in range(n_weeks):
                u = rng.random()
                if u < 0.05:
                    level *= float(np.exp(rng.normal(0, 0.10)))  # permanent change
                promo = 0.8 if rng.random() < 0.12 else 1.0  # promotion week
                price_w[w] = round(level * promo, 2)
            price_d = price_w[day_week_pos_obs]
            ar = np.zeros(n_days)
            for t in range(1, n_days):
                ar[t] = 0.95 * ar[t - 1] + rng.normal(0, 0.05)
            log_lam = (
                np.log(base)
                + beta * np.log(price_d / ref_price)
                + WEEKDAY_EFFECT[wday0]
                + 0.35 * event_flag
                + (0.15 * snap_by_state[st] if cat == "FOODS" else 0.0)
                + ar
            )
            sales = rng.poisson(np.exp(log_lam))
            sales[day_week_pos_obs < release_week] = 0
            sales_rows.append(
                [f"{item}_{store}_validation", item, dept, cat, store, st, *sales.tolist()]
            )
            for w in range(release_week, n_weeks):
                price_rows.append([store, item, int(weeks[w]), float(price_w[w])])

    sales = pd.DataFrame(
        sales_rows,
        columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        + [f"d_{i}" for i in range(1, n_days + 1)],
    )
    prices = pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    sales.to_csv(out_dir / "sales_train_validation.csv", index=False)
    cal.to_csv(out_dir / "calendar.csv", index=False)
    prices.to_csv(out_dir / "sell_prices.csv", index=False)
    return {"sales": sales, "calendar": cal, "prices": prices}
