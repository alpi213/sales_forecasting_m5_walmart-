"""Hierarchical Bayesian price elasticity (PyMC).

    log_q[n] ~ Normal(mu[n], sigma)
    mu[n]    = alpha[s(n)] + beta[s(n)] * (log_p[n] - mean_log_p[s(n)])
               + rho * lag_log_q_1[n] + g_snap * snap_days[n] + g_event * event_days[n]
               + sum_k (a_k sin(2*pi*k*w/52) + b_k cos(2*pi*k*w/52))          # annual seasonality
    beta[s]  = mu_beta[cat(s)] + tau_beta * z_beta[s],  z_beta ~ Normal(0, 1)   # non-centred
    alpha[s] = mu_alpha[dept(s)] + tau_alpha * z_alpha[s]
    mu_beta[c] ~ Normal(-1, 1),  tau_beta ~ HalfNormal(0.5)

Each item-store `s` gets its own elasticity beta[s], but it is shrunk toward its category mean
by an amount the data chooses (tau_beta). Item-stores with many informative weeks keep their
own estimate; sparse ones are pulled to the category. This is the answer to "we have 30k
products, most with too little price variation to estimate anything": partial pooling.

The non-centred parameterisation (beta = mu + tau * z) is essential for NUTS: with the centred
form, small tau creates a funnel geometry that produces divergences.

Runtime: NUTS on the full panel (~1.5M rows) is slow; `max_items_per_category` in the config
subsamples items, and `method="advi"` gives a fast mean-field approximation for a first look.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _subsample_items(panel: pd.DataFrame, max_items: int | None, seed: int) -> pd.DataFrame:
    if max_items is None:
        return panel
    rng = np.random.default_rng(seed)
    keep = []
    for _, sub in panel.groupby("cat_id"):
        items = sub["item_id"].unique()
        if len(items) > max_items:
            items = rng.choice(items, size=max_items, replace=False)
        keep.append(sub[sub["item_id"].isin(items)])
    return pd.concat(keep).reset_index(drop=True)


def prepare(panel: pd.DataFrame, max_items: int | None = None, seed: int = 0) -> dict[str, Any]:
    df = _subsample_items(panel, max_items, seed).copy()
    df["series"] = df["store_id"] + "|" + df["item_id"]
    s_codes, s_levels = pd.factorize(df["series"])
    series_meta = df.drop_duplicates("series").set_index("series").loc[s_levels]
    c_codes, c_levels = pd.factorize(series_meta["cat_id"])
    d_codes, d_levels = pd.factorize(series_meta["dept_id"])
    mean_log_p = df.groupby("series")["log_p"].transform("mean")
    w = df["week_of_year"].to_numpy(dtype=float)
    fourier = np.column_stack(
        [f(2 * np.pi * k * w / 52.0) for k in (1, 2) for f in (np.sin, np.cos)]
    )
    return {
        "y": df["log_q"].to_numpy(dtype=float),
        "dp": (df["log_p"] - mean_log_p).to_numpy(dtype=float),
        "lag_q": (df["lag_log_q_1"] - df["lag_log_q_1"].mean()).to_numpy(dtype=float),
        "snap": df["snap_days"].to_numpy(dtype=float) / 7.0,
        "event": df["event_days"].to_numpy(dtype=float) / 7.0,
        "fourier": fourier,
        "s": s_codes, "s_levels": list(s_levels),
        "cat_of_s": c_codes, "cat_levels": list(c_levels),
        "dept_of_s": d_codes, "dept_levels": list(d_levels),
        "n": len(df),
    }


def build_model(d: dict[str, Any]):
    import pymc as pm

    n_s, n_c, n_d = len(d["s_levels"]), len(d["cat_levels"]), len(d["dept_levels"])
    coords = {"series": d["s_levels"], "cat": d["cat_levels"], "dept": d["dept_levels"], "fourier": list(range(4))}
    with pm.Model(coords=coords) as model:
        mu_beta = pm.Normal("mu_beta", mu=-1.0, sigma=1.0, dims="cat")
        tau_beta = pm.HalfNormal("tau_beta", sigma=0.5)
        z_beta = pm.Normal("z_beta", 0.0, 1.0, dims="series")
        beta = pm.Deterministic("beta", mu_beta[d["cat_of_s"]] + tau_beta * z_beta, dims="series")

        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=2.0, dims="dept")
        tau_alpha = pm.HalfNormal("tau_alpha", sigma=1.0)
        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="series")
        alpha = pm.Deterministic("alpha", mu_alpha[d["dept_of_s"]] + tau_alpha * z_alpha, dims="series")

        rho = pm.Normal("rho", 0.0, 0.5)
        g_snap = pm.Normal("g_snap", 0.0, 0.5)
        g_event = pm.Normal("g_event", 0.0, 0.5)
        fcoef = pm.Normal("fourier_coef", 0.0, 0.5, dims="fourier")
        sigma = pm.HalfNormal("sigma", 1.0)

        s = d["s"]
        mu = (
            alpha[s] + beta[s] * d["dp"] + rho * d["lag_q"] + g_snap * d["snap"] + g_event * d["event"]
            + pm.math.dot(d["fourier"], fcoef)
        )
        pm.Normal("log_q", mu=mu, sigma=sigma, observed=d["y"])
    return model


def fit(d: dict[str, Any], draws: int, tune: int, chains: int, target_accept: float, method: str = "nuts", seed: int = 0):
    import pymc as pm

    model = build_model(d)
    with model:
        if method == "advi":
            approx = pm.fit(n=30_000, method="advi", random_seed=seed)
            idata = approx.sample(draws, random_seed=seed)
        else:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                              random_seed=seed, progressbar=False)
    return model, idata


def summarise(idata, d: dict[str, Any]) -> dict[str, pd.DataFrame]:
    import arviz as az

    cat = az.summary(idata, var_names=["mu_beta"], hdi_prob=0.95).reset_index()
    cat["cat_id"] = d["cat_levels"]
    series = az.summary(idata, var_names=["beta"], hdi_prob=0.95).reset_index()
    series["series"] = d["s_levels"]
    series[["store_id", "item_id"]] = series["series"].str.split("|", expand=True)
    hyper = az.summary(idata, var_names=["tau_beta", "rho", "g_snap", "g_event", "sigma"], hdi_prob=0.95)
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_div = int(idata.sample_stats["diverging"].sum())
        log.info("divergences: %d", n_div)
    return {"category": cat[["cat_id", "mean", "sd", "hdi_2.5%", "hdi_97.5%", "r_hat"]],
            "series": series[["store_id", "item_id", "mean", "sd", "hdi_2.5%", "hdi_97.5%"]],
            "hyper": hyper.reset_index()}
