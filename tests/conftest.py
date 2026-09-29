"""Shared fixtures: a small synthetic dataset loaded into an in-memory sqlite database.

sqlite is used so the tests do not depend on duckdb; the SQL in the package is written in the
window-function subset both engines share (see features.py).
"""
import sqlite3
from pathlib import Path

import pytest

from m5.config import load_config
from m5.data.features import build_features_table
from m5.data.load import build_database
from m5.data.synthetic import generate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    c = load_config(ROOT / "configs" / "default.yaml")
    c["data"]["min_train_days"] = 200
    c["features"]["lags"] = [28, 35, 42, 56]
    c["features"]["rolling_windows"] = [7, 14, 28]
    c["features"]["rolling_std_windows"] = [7, 28]
    return c


@pytest.fixture(scope="session")
def raw(tmp_path_factory):
    out = tmp_path_factory.mktemp("raw")
    return generate(out, n_items_per_dept=6, n_days=420, seed=1)


@pytest.fixture(scope="session")
def db(cfg, raw):
    con = sqlite3.connect(":memory:")
    last_day = build_database(con, raw["sales"], raw["calendar"], raw["prices"], horizon=cfg["data"]["horizon"])
    build_features_table(con, cfg, last_day)
    return con, last_day
