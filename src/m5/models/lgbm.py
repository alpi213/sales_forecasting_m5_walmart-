"""Global LightGBM model with a Tweedie objective.

Why this setup
--------------
* One global model across all ~30k series: it learns cross-sectional patterns (price effects,
  calendar effects, department behaviour) that per-series models cannot, and it is what won M5.
* Tweedie objective (variance power in (1,2)): daily item-store sales are non-negative,
  intermittent (many exact zeros) and over-dispersed. Tweedie is a Poisson-Gamma compound that
  handles the zero mass and the skew; MSE would over-predict on zeros, plain Poisson under-fits
  the dispersion.
* Direct multi-horizon: all sales-derived features are shifted by >= 28 days (see features.py),
  so the same model predicts every day of the horizon from observed data only.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from m5.data.features import CATEGORICAL, CategoryEncoder, feature_columns, load_features
from m5.data.load import load_calendar, load_prices, load_wide
from m5.evaluation.splits import Fold, rolling_origin_folds
from m5.evaluation.wrmsse import WRMSSEResult, dollar_sales_last_days, wrmsse

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: Fold
    best_iteration: int
    valid_rmse: float
    wrmsse: WRMSSEResult


def _lgb():
    import lightgbm as lgb  # lazy import: keeps tests independent of lightgbm

    return lgb


def _params(cfg: dict[str, Any]) -> dict[str, Any]:
    p = dict(cfg["lgbm"])
    p.pop("num_boost_round", None)
    p.pop("early_stopping_rounds", None)
    return p


def train_booster(
    train: pd.DataFrame,
    valid: pd.DataFrame | None,
    cfg: dict[str, Any],
    features: list[str],
    num_boost_round: int | None = None,
):
    lgb = _lgb()
    dtrain = lgb.Dataset(train[features], label=train["sales"], categorical_feature=CATEGORICAL, free_raw_data=False)
    valid_sets, callbacks = [dtrain], [lgb.log_evaluation(period=100)]
    if valid is not None:
        dvalid = lgb.Dataset(valid[features], label=valid["sales"], reference=dtrain, categorical_feature=CATEGORICAL)
        valid_sets.append(dvalid)
        callbacks.append(lgb.early_stopping(cfg["lgbm"]["early_stopping_rounds"], verbose=True))
    booster = lgb.train(
        _params(cfg),
        dtrain,
        num_boost_round=num_boost_round or cfg["lgbm"]["num_boost_round"],
        valid_sets=valid_sets,
        valid_names=["train", "valid"][: len(valid_sets)],
        callbacks=callbacks,
    )
    return booster


def predictions_to_wide(pred: pd.DataFrame, meta: pd.DataFrame, day_from: int, horizon: int) -> np.ndarray:
    """(n_series, horizon) array aligned with `meta`; series without a price row get 0."""
    out = np.zeros((len(meta), horizon), dtype=np.float64)
    row_of = pd.Series(np.arange(len(meta)), index=meta["id"])
    rows = row_of.reindex(pred["id"]).to_numpy()
    cols = pred["day_idx"].to_numpy() - day_from
    ok = ~np.isnan(rows) & (cols >= 0) & (cols < horizon)
    out[rows[ok].astype(int), cols[ok]] = pred["yhat"].to_numpy()[ok]
    return out


def score_fold(con: Any, fold: Fold, pred: pd.DataFrame, horizon: int) -> WRMSSEResult:
    meta, wide = load_wide(con, fold.valid_end)
    train_hist = wide[:, : fold.train_end]
    actual = wide[:, fold.valid_start - 1 : fold.valid_end]
    prices, cal = load_prices(con), load_calendar(con)
    dollars = dollar_sales_last_days(wide, meta, prices, cal, last_day=fold.train_end)
    forecast = predictions_to_wide(pred, meta, fold.valid_start, horizon)
    return wrmsse(train_hist, actual, forecast, meta, dollars)


def cross_validate(con: Any, cfg: dict[str, Any], last_day: int, artifacts_dir: str | Path) -> list[FoldResult]:
    horizon = cfg["data"]["horizon"]
    folds = rolling_origin_folds(last_day, horizon, cfg["cv"]["n_folds"], cfg["cv"]["gap"])
    features = feature_columns(cfg)
    results: list[FoldResult] = []
    for fold in folds:
        log.info("fold %s: train <= d_%d", fold.name, fold.train_end)
        train = load_features(con, cfg, fold.train_end - cfg["data"]["min_train_days"], fold.train_end)
        valid = load_features(con, cfg, fold.valid_start, fold.valid_end)
        enc = CategoryEncoder().fit(train, CATEGORICAL)
        train, valid = enc.transform(train), enc.transform(valid)
        booster = train_booster(train, valid, cfg, features)
        valid_pred = np.clip(booster.predict(valid[features], num_iteration=booster.best_iteration), 0, None)
        rmse = float(np.sqrt(np.mean((valid_pred - valid["sales"].to_numpy()) ** 2)))
        pred = pd.DataFrame({"id": valid["id"], "day_idx": valid["day_idx"], "yhat": valid_pred})
        score = score_fold(con, fold, pred, horizon)
        log.info("fold %s: best_iter=%d rmse=%.4f\n%s", fold.name, booster.best_iteration, rmse, score)
        results.append(FoldResult(fold, booster.best_iteration, rmse, score))
        pred.to_csv(Path(artifacts_dir) / f"lgbm_pred_{fold.name}.csv", index=False)
    summary = {
        "folds": [
            {"fold": r.fold.name, "best_iteration": r.best_iteration, "valid_rmse": r.valid_rmse,
             "wrmsse": r.wrmsse.total, "per_level": r.wrmsse.per_level}
            for r in results
        ],
        "mean_wrmsse": float(np.mean([r.wrmsse.total for r in results])),
    }
    with open(Path(artifacts_dir) / "lgbm_cv.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("mean WRMSSE over %d folds: %.4f", len(results), summary["mean_wrmsse"])
    return results


def fit_final(con: Any, cfg: dict[str, Any], last_day: int, num_rounds: int, artifacts_dir: str | Path) -> Path:
    """Refit on all history with the number of rounds chosen by CV; persist model + encoder + feature list."""
    features = feature_columns(cfg)
    train = load_features(con, cfg, last_day - cfg["data"]["min_train_days"], last_day)
    enc = CategoryEncoder().fit(train, CATEGORICAL)
    booster = train_booster(enc.transform(train), None, cfg, features, num_boost_round=num_rounds)
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "lgbm_final.txt"))
    enc.save(out / "lgbm_encoder.json")
    with open(out / "lgbm_final.json", "w") as f:
        json.dump({"features": features, "num_rounds": num_rounds, "last_day": last_day}, f, indent=2)
    imp = pd.DataFrame({"feature": features, "gain": booster.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp.to_csv(out / "lgbm_feature_importance.csv", index=False)
    log.info("top features:\n%s", imp.head(15).to_string(index=False))
    return out / "lgbm_final.txt"


def predict_horizon(con: Any, cfg: dict[str, Any], last_day: int, artifacts_dir: str | Path) -> pd.DataFrame:
    """Forecast days last_day+1 .. last_day+horizon with the saved final model."""
    lgb = _lgb()
    out = Path(artifacts_dir)
    with open(out / "lgbm_final.json") as f:
        info = json.load(f)
    booster = lgb.Booster(model_file=str(out / "lgbm_final.txt"))
    enc = CategoryEncoder.load(out / "lgbm_encoder.json")
    horizon = cfg["data"]["horizon"]
    fut = load_features(con, cfg, last_day + 1, last_day + horizon)
    fut = enc.transform(fut)
    yhat = np.clip(booster.predict(fut[info["features"]]), 0, None)
    return pd.DataFrame({"id": fut["id"], "day_idx": fut["day_idx"], "yhat": yhat})


def to_submission(pred: pd.DataFrame, ids: pd.Series, day_from: int, horizon: int) -> pd.DataFrame:
    """Kaggle-style wide frame: id, F1..F28."""
    meta = pd.DataFrame({"id": ids})
    wide = predictions_to_wide(pred, meta, day_from, horizon)
    sub = pd.DataFrame(wide, columns=[f"F{i}" for i in range(1, horizon + 1)])
    sub.insert(0, "id", ids.to_numpy())
    return sub
