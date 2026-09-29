"""Price elasticity via Double / Debiased Machine Learning (Chernozhukov et al., 2018).

Model (partially linear):
    log_q = theta * log_p + g(X) + eps
    log_p = m(X) + v
where X are confounders that move both price and demand: the item and store (fixed effects),
seasonality (week of year, year), SNAP and event days, recent demand (lagged log_q), and the
previous week's price. theta is the elasticity.

Why not a plain log-log regression: prices are not set at random. Retailers cut prices when
demand is expected to be weak (or run promotions in high-traffic weeks), so log_p is
correlated with the error term, and the naive slope is biased. DML fits the two nuisance
functions g and m with a flexible learner (gradient boosting), cross-fits them (each row's
nuisance prediction comes from a model that never saw that row), and regresses residual on
residual. Neyman orthogonality makes theta insensitive to first-order errors in g and m, so a
biased-but-consistent ML fit of the nuisances still gives a sqrt(n)-consistent theta with a
valid standard error.

Estimates are produced at three grains: category, department and item (pooled over stores).
Item-level estimates on a few dozen weeks are noisy; the hierarchical Bayesian model in
`elasticity_bayes.py` is the principled way to shrink them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

log = logging.getLogger(__name__)

NUMERIC_CONTROLS = ["week_of_year", "year", "week_idx", "snap_days", "event_days",
                    "lag_log_q_1", "lag_log_q_2", "lag_log_q_4", "lag_log_p_1"]
CATEGORICAL_CONTROLS = ["item_id", "store_id"]


@dataclass
class ElasticityEstimate:
    group: str
    theta: float
    se: float
    n: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.theta - 1.96 * self.se, self.theta + 1.96 * self.se


def _design(panel: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    X = panel[NUMERIC_CONTROLS].astype(float).copy()
    cat_idx = []
    for c in CATEGORICAL_CONTROLS:
        X[c] = panel[c].astype("category").cat.codes.astype(float)
        cat_idx.append(X.columns.get_loc(c))
    return X.to_numpy(), cat_idx


def _learner(cat_idx: list[int], seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=cat_idx, random_state=seed,
    )


def cross_fit_residuals(panel: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Add columns y_res (log_q - g_hat) and d_res (log_p - m_hat), each out-of-fold."""
    X, cat_idx = _design(panel)
    y = panel["log_q"].to_numpy(dtype=float)
    d = panel["log_p"].to_numpy(dtype=float)
    y_hat, d_hat = np.zeros_like(y), np.zeros_like(d)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(kf.split(X)):
        g = _learner(cat_idx, seed + k).fit(X[tr], y[tr])
        m = _learner(cat_idx, seed + 100 + k).fit(X[tr], d[tr])
        y_hat[te], d_hat[te] = g.predict(X[te]), m.predict(X[te])
        log.info("cross-fit fold %d/%d done", k + 1, n_folds)
    out = panel.copy()
    out["y_res"], out["d_res"] = y - y_hat, d - d_hat
    out["_g_r2"] = 1 - np.var(y - y_hat) / np.var(y)
    out["_m_r2"] = 1 - np.var(d - d_hat) / np.var(d)
    return out


def partial_out_theta(y_res: np.ndarray, d_res: np.ndarray) -> tuple[float, float]:
    """theta = sum(v*u)/sum(v^2), with the influence-function (HC0) standard error."""
    denom = np.sum(d_res * d_res)
    if denom <= 0 or len(y_res) < 3:
        return np.nan, np.nan
    theta = float(np.sum(d_res * y_res) / denom)
    psi = (y_res - theta * d_res) * d_res
    j = denom / len(y_res)
    se = float(np.sqrt(np.mean(psi**2) / (j**2) / len(y_res)))
    return theta, se


def estimate_by_group(res: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for g, sub in res.groupby(key):
        theta, se = partial_out_theta(sub["y_res"].to_numpy(), sub["d_res"].to_numpy())
        rows.append({key: g, "theta": theta, "se": se, "ci_low": theta - 1.96 * se, "ci_high": theta + 1.96 * se, "n": len(sub)})
    return pd.DataFrame(rows)


def naive_loglog(panel: pd.DataFrame, key: str) -> pd.DataFrame:
    """Uncontrolled log-log slope, reported to show the bias DML removes."""
    rows = []
    for g, sub in panel.groupby(key):
        x, y = sub["log_p"].to_numpy(), sub["log_q"].to_numpy()
        x, y = x - x.mean(), y - y.mean()
        rows.append({key: g, "theta_naive": float((x @ y) / (x @ x)) if (x @ x) > 0 else np.nan})
    return pd.DataFrame(rows)


def run_dml(panel: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> dict[str, pd.DataFrame]:
    res = cross_fit_residuals(panel, n_folds=n_folds, seed=seed)
    log.info("nuisance fit: R2(g)=%.3f R2(m)=%.3f", res["_g_r2"].iloc[0], res["_m_r2"].iloc[0])
    out = {}
    for key in ("cat_id", "dept_id", "item_id"):
        est = estimate_by_group(res, key).merge(naive_loglog(panel, key), on=key)
        out[key] = est.sort_values("theta")
    out["cat_id"] = out["cat_id"].sort_values("cat_id")
    log.info("category elasticities (DML vs naive):\n%s", out["cat_id"].to_string(index=False))
    return out
