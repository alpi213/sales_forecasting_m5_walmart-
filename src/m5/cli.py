"""Command-line orchestration: `m5 <stage> [--config ...] [--set key=value ...]`.

Stages are idempotent and read/write only through the DuckDB file and the artifacts directory,
so they can be chained by any scheduler (cron, Airflow, GitHub Actions):

    m5 make-synthetic            # small fake dataset in the M5 schema (for tests / smoke runs)
    m5 build-db                  # validate CSVs -> DuckDB long tables -> SQL feature table
    m5 train-lgbm                # rolling-origin CV, then refit on all data and save the model
    m5 train-patchtst            # same protocol for the PyTorch transformer
    m5 predict --model lgbm      # 28-day forecast -> artifacts/submission_lgbm.csv
    m5 elasticity-dml            # DML elasticities by category / dept / item
    m5 elasticity-bayes          # hierarchical Bayesian elasticities
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from m5.config import load_config
from m5.logging_utils import setup_logging, timed

log = logging.getLogger("m5")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="key.sub=value")
    p.add_argument("--log-level", default="INFO")


def cmd_make_synthetic(a: argparse.Namespace) -> None:
    from m5.data.synthetic import generate

    cfg = load_config(a.config, a.overrides)
    out = Path(a.out or cfg["paths"]["raw_dir"])
    with timed(log, f"synthetic data -> {out}"):
        d = generate(out, n_items_per_dept=a.items, n_days=a.days, seed=a.seed)
    log.info("%d series, %d days, %d price rows", len(d["sales"]), a.days, len(d["prices"]))


def cmd_build_db(a: argparse.Namespace) -> None:
    from m5.data.features import build_features_table
    from m5.data.load import build_database, connect, load_raw

    cfg = load_config(a.config, a.overrides)
    with timed(log, "read + validate raw CSVs"):
        sales, cal, prices = load_raw(cfg["paths"]["raw_dir"], cfg["paths"]["sales_file"])
    con = connect(cfg["paths"]["db_path"])
    with timed(log, "load long tables"):
        last_day = build_database(con, sales, cal, prices, horizon=cfg["data"]["horizon"])
    with timed(log, "build SQL feature table"):
        build_features_table(con, cfg, last_day)
    con.close()


def _open(cfg):
    from m5.data.load import connect, read_meta

    con = connect(cfg["paths"]["db_path"])
    meta = read_meta(con)
    return con, meta["last_day"]


def cmd_train_lgbm(a: argparse.Namespace) -> None:
    from m5.models import lgbm

    cfg = load_config(a.config, a.overrides)
    con, last_day = _open(cfg)
    art = Path(cfg["paths"]["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    with timed(log, "LightGBM cross-validation"):
        results = lgbm.cross_validate(con, cfg, last_day, art)
    if not a.cv_only:
        rounds = int(sum(r.best_iteration for r in results) / len(results) * 1.1)  # more data -> a few more rounds
        with timed(log, f"LightGBM final fit ({rounds} rounds)"):
            lgbm.fit_final(con, cfg, last_day, rounds, art)
    con.close()


def cmd_train_patchtst(a: argparse.Namespace) -> None:
    from m5.models import torch_train

    cfg = load_config(a.config, a.overrides)
    con, last_day = _open(cfg)
    art = Path(cfg["paths"]["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    with timed(log, "PatchTST cross-validation"):
        torch_train.cross_validate(con, cfg, last_day, art)
    if not a.cv_only:
        with timed(log, "PatchTST final fit + forecast"):
            sub = torch_train.fit_final_and_predict(con, cfg, last_day, art)
        sub.to_csv(art / "submission_patchtst.csv", index=False)
    con.close()


def cmd_predict(a: argparse.Namespace) -> None:
    from m5.data.load import run_sql
    from m5.models import lgbm

    cfg = load_config(a.config, a.overrides)
    con, last_day = _open(cfg)
    art = Path(cfg["paths"]["artifacts_dir"])
    horizon = cfg["data"]["horizon"]
    if a.model == "lgbm":
        with timed(log, "LightGBM forecast"):
            pred = lgbm.predict_horizon(con, cfg, last_day, art)
            ids = run_sql(con, "SELECT DISTINCT id FROM sales_long ORDER BY id")["id"]
            sub = lgbm.to_submission(pred, ids, last_day + 1, horizon)
    else:
        raise SystemExit("PatchTST forecasts are written by `train-patchtst` (submission_patchtst.csv)")
    out = art / f"submission_{a.model}.csv"
    sub.to_csv(out, index=False)
    log.info("wrote %s (%d series x %d days, mean forecast %.3f)", out, len(sub), horizon, sub.iloc[:, 1:].to_numpy().mean())
    con.close()


def cmd_elasticity_dml(a: argparse.Namespace) -> None:
    from m5.models.elasticity_data import build_weekly_panel
    from m5.models.elasticity_dml import run_dml

    cfg = load_config(a.config, a.overrides)
    con, last_day = _open(cfg)
    art = Path(cfg["paths"]["artifacts_dir"])
    e = cfg["elasticity"]
    with timed(log, "weekly panel"):
        panel = build_weekly_panel(con, last_day, e["min_weeks_per_item"])
    with timed(log, "DML"):
        out = run_dml(panel, n_folds=e["n_folds"])
    for key, df in out.items():
        df.to_csv(art / f"elasticity_dml_{key}.csv", index=False)
    con.close()


def cmd_elasticity_bayes(a: argparse.Namespace) -> None:
    from m5.models import elasticity_bayes as eb
    from m5.models.elasticity_data import build_weekly_panel

    cfg = load_config(a.config, a.overrides)
    con, last_day = _open(cfg)
    art = Path(cfg["paths"]["artifacts_dir"])
    e = cfg["elasticity"]
    with timed(log, "weekly panel"):
        panel = build_weekly_panel(con, last_day, e["min_weeks_per_item"])
    d = eb.prepare(panel, max_items=e["max_items_per_category"])
    log.info("Bayesian model on %d rows, %d item-stores", d["n"], len(d["s_levels"]))
    with timed(log, f"PyMC {a.method}"):
        _, idata = eb.fit(d, method=a.method, **e["bayes"])
    out = eb.summarise(idata, d)
    for key, df in out.items():
        df.to_csv(art / f"elasticity_bayes_{key}.csv", index=False)
    log.info("category elasticities:\n%s", out["category"].to_string(index=False))
    con.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="m5", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-synthetic"); _common(s)
    s.add_argument("--out", default=None); s.add_argument("--items", type=int, default=12)
    s.add_argument("--days", type=int, default=500); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_make_synthetic)

    s = sub.add_parser("build-db"); _common(s); s.set_defaults(fn=cmd_build_db)

    s = sub.add_parser("train-lgbm"); _common(s)
    s.add_argument("--cv-only", action="store_true"); s.set_defaults(fn=cmd_train_lgbm)

    s = sub.add_parser("train-patchtst"); _common(s)
    s.add_argument("--cv-only", action="store_true"); s.set_defaults(fn=cmd_train_patchtst)

    s = sub.add_parser("predict"); _common(s)
    s.add_argument("--model", choices=["lgbm", "patchtst"], default="lgbm"); s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("elasticity-dml"); _common(s); s.set_defaults(fn=cmd_elasticity_dml)

    s = sub.add_parser("elasticity-bayes"); _common(s)
    s.add_argument("--method", choices=["nuts", "advi"], default="nuts"); s.set_defaults(fn=cmd_elasticity_bayes)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_dir="artifacts/logs")
    try:
        args.fn(args)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception:  # noqa: BLE001 - top-level: log with traceback, non-zero exit for schedulers
        log.exception("stage %s failed", args.cmd)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
