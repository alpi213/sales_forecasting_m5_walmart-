import numpy as np
import pandas as pd

from m5.data.features import CATEGORICAL, CategoryEncoder, feature_columns, load_features
from m5.data.load import run_sql


def _series(raw, id_):
    row = raw["sales"].set_index("id").loc[id_]
    return row[[c for c in row.index if c.startswith("d_")]].to_numpy(dtype=float)


def test_lag_and_rolling_match_pandas(cfg, raw, db):
    con, last_day = db
    feats = load_features(con, cfg, 1, last_day, require_price=False)
    id_ = feats["id"].iloc[0]
    y = _series(raw, id_)
    f = feats[feats["id"] == id_].set_index("day_idx")
    for t in [200, 300, last_day]:
        assert f.loc[t, "lag_28"] == y[t - 1 - 28]
        assert f.loc[t, "lag_56"] == y[t - 1 - 56]
        # rmean_7 = mean of sales at lags 28..34 (window ends `horizon` days back)
        assert np.isclose(f.loc[t, "rmean_7"], y[t - 1 - 34 : t - 1 - 27].mean())
        assert np.isclose(f.loc[t, "rmean_28"], y[t - 1 - 55 : t - 1 - 27].mean())
        assert np.isclose(f.loc[t, "rstd_7"], y[t - 1 - 34 : t - 1 - 27].std(), atol=1e-5)


def test_no_sales_leakage_in_future_rows(cfg, db):
    con, last_day = db
    fut = load_features(con, cfg, last_day + 1, last_day + cfg["data"]["horizon"], require_price=False)
    assert fut["sales"].isna().all()
    sales_feats = [c for c in feature_columns(cfg) if c.startswith(("lag_", "rmean_", "rstd_", "rmax", "zero"))]
    assert fut[sales_feats].notna().all().all(), "future rows must have complete shifted features"
    assert len(fut) == run_sql(con, "SELECT COUNT(DISTINCT id) AS n FROM sales_long")["n"].iloc[0] * cfg["data"]["horizon"]


def test_price_features_bounds(cfg, db):
    con, last_day = db
    f = load_features(con, cfg, last_day - 100, last_day)
    assert f["sell_price"].notna().all()
    assert (f["price_rel_max"] <= 1 + 1e-6).all() and (f["price_rel_max"] > 0).all()
    assert f["price_rel_dept"].between(0.01, 100).all()
    assert (f["days_since_release"] >= 0).all()
    assert f["snap"].isin([0, 1]).all()


def test_feature_columns_exist(cfg, db):
    con, last_day = db
    f = load_features(con, cfg, last_day - 10, last_day)
    missing = set(feature_columns(cfg)) - set(f.columns)
    assert not missing, missing


def test_category_encoder_roundtrip(cfg, db, tmp_path):
    con, last_day = db
    f = load_features(con, cfg, last_day - 10, last_day)
    enc = CategoryEncoder().fit(f, CATEGORICAL)
    enc.save(tmp_path / "enc.json")
    enc2 = CategoryEncoder.load(tmp_path / "enc.json")
    a, b = enc.transform(f), enc2.transform(f)
    pd.testing.assert_frame_equal(a, b)
    unseen = f.head(3).copy()
    unseen["store_id"] = "ZZ_9"
    assert (enc.transform(unseen)["store_id"] == -1).all()
