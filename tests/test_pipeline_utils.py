"""Pieces of the model pipelines that do not need lightgbm/torch: wide loader, prediction
alignment, fold scoring with a seasonal-naive forecast."""
import numpy as np
import pandas as pd

from m5.data.load import load_wide
from m5.evaluation.splits import rolling_origin_folds
from m5.models.lgbm import predictions_to_wide, score_fold, to_submission


def test_load_wide_matches_raw(raw, db):
    con, last_day = db
    meta, wide = load_wide(con, last_day)
    assert wide.shape == (len(raw["sales"]), last_day)
    src = raw["sales"].set_index("id").loc[meta["id"]]
    np.testing.assert_array_equal(wide, src[[f"d_{i}" for i in range(1, last_day + 1)]].to_numpy(dtype=float))


def test_predictions_to_wide_alignment():
    meta = pd.DataFrame({"id": ["b", "a", "c"]})
    pred = pd.DataFrame({"id": ["a", "a", "c", "zzz"], "day_idx": [101, 103, 128, 105], "yhat": [1.0, 2.0, 3.0, 9.0]})
    w = predictions_to_wide(pred, meta, day_from=101, horizon=28)
    assert w.shape == (3, 28) and w[1, 0] == 1.0 and w[1, 2] == 2.0 and w[2, 27] == 3.0 and w.sum() == 6.0
    sub = to_submission(pred, meta["id"], 101, 28)
    assert list(sub.columns[:2]) == ["id", "F1"] and sub.shape == (3, 29)


def test_score_fold_seasonal_naive(cfg, db):
    """A seasonal-naive forecast (sales 28 days earlier) is a sane baseline: WRMSSE ~ 0.7-2."""
    con, last_day = db
    fold = rolling_origin_folds(last_day, 28, 1)[0]
    meta, wide = load_wide(con, fold.valid_end)
    naive = wide[:, fold.train_end - 28 : fold.train_end]
    pred = pd.DataFrame({
        "id": np.repeat(meta["id"].to_numpy(), 28),
        "day_idx": np.tile(np.arange(fold.valid_start, fold.valid_end + 1), len(meta)),
        "yhat": naive.ravel(),
    })
    res = score_fold(con, fold, pred, horizon=28)
    assert 0.5 < res.total < 2.5, res
    assert len(res.per_level) == 12
