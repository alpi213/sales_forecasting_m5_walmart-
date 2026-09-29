import numpy as np

from m5.data.synthetic import TRUE_ELASTICITY
from m5.models.elasticity_data import build_weekly_panel
from m5.models.elasticity_dml import naive_loglog, partial_out_theta, run_dml


def test_partial_out_theta_recovers_slope():
    rng = np.random.default_rng(0)
    v = rng.normal(size=5000)
    u = -1.3 * v + rng.normal(scale=0.5, size=5000)
    theta, se = partial_out_theta(u, v)
    assert abs(theta + 1.3) < 3 * se and se < 0.05


def test_panel_shape(cfg, db):
    con, last_day = db
    panel = build_weekly_panel(con, last_day, min_weeks=20)
    assert {"log_q", "log_p", "lag_log_q_4", "lag_log_p_1", "snap_days"} <= set(panel.columns)
    assert (panel["units"] >= 0).all() and (panel["price"] > 0).all()
    assert panel.groupby("series")["week_idx"].apply(lambda s: s.is_monotonic_increasing).all()


def test_dml_recovers_category_elasticity(cfg, db):
    con, last_day = db
    panel = build_weekly_panel(con, last_day, min_weeks=20)
    out = run_dml(panel, n_folds=3, seed=0)
    cats = out["cat_id"].set_index("cat_id")
    for cat, true in TRUE_ELASTICITY.items():
        est, se = cats.loc[cat, "theta"], cats.loc[cat, "se"]
        assert abs(est - true) < max(0.25, 3 * se), f"{cat}: {est:.2f} +- {se:.2f} vs {true}"
        assert np.isfinite(cats.loc[cat, "theta_naive"])
