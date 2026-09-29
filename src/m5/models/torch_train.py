"""Training and evaluation of PatchTST on the M5 sales matrix.

Data layout: a dense matrix Y of shape (n_series, T). A training sample is a pair
(Y[i, t-L:t], Y[i, t:t+H]) with t chosen every `sample_stride` days; validation uses the single
window ending at the fold's validation end, exactly like the LightGBM fold, so the two models
are scored on the same days with the same WRMSSE.

Series are dropped before their release (first non-zero day) so the model does not learn
from pre-launch zeros, and a window is only used if it starts after the release day.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from m5.data.load import load_calendar, load_prices, load_wide
from m5.evaluation.splits import rolling_origin_folds
from m5.evaluation.wrmsse import WRMSSEResult, dollar_sales_last_days, wrmsse
from m5.models.patchtst import (
    PatchTST,
    PatchTSTConfig,
    count_parameters,
    receptive_field_note,
    tweedie_deviance,
)

log = logging.getLogger(__name__)


class WindowDataset:
    """All (series, end) windows with `lookback` history and `horizon` targets inside [0, t_max).

    The dense sales tensor lives on `device` and a batch is one advanced-indexing gather from
    it, so there is no per-sample Python work (a DataLoader with per-sample `__getitem__` costs
    ~1M Python calls per epoch here).
    """

    def __init__(
        self, y: np.ndarray, release: np.ndarray, lookback: int, horizon: int, t_max: int, stride: int,
        device: torch.device | str = "cpu",
    ):
        self.y = torch.from_numpy(y.astype(np.float32)).to(device)
        self.lookback, self.horizon = lookback, horizon
        ends = np.arange(t_max - horizon, lookback - 1, -stride)  # window end t: history [t-L, t), target [t, t+H)
        # `ends` is decreasing; the windows allowed for series i are the first n_ok[i] entries
        # (those with end - lookback >= release[i], i.e. history starting after the release day)
        n_ok = np.searchsorted(-ends, -(np.asarray(release) + lookback), side="right")
        series = np.repeat(np.arange(y.shape[0]), n_ok)
        t_end = np.concatenate([ends[:k] for k in n_ok]) if len(series) else np.zeros(0, dtype=np.int64)
        self.index = np.column_stack([series, t_end]).astype(np.int64)
        self._index_t = torch.from_numpy(self.index).to(device)
        self._x_off = torch.arange(-lookback, 0, device=device)
        self._y_off = torch.arange(0, horizon, device=device)

    def __len__(self) -> int:
        return len(self.index)

    def gather(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(x, y, series_idx) for the windows at `rows` of `self.index`, on the dataset's device."""
        sel = self._index_t[rows.to(self._index_t.device)]
        i, t = sel[:, 0], sel[:, 1]
        x = self.y[i[:, None], t[:, None] + self._x_off]
        y = self.y[i[:, None], t[:, None] + self._y_off]
        return x, y, i

    def batches(self, batch_size: int, shuffle: bool):
        n = len(self)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for start in range(0, n, batch_size):
            yield self.gather(order[start : start + batch_size])


def release_days(y: np.ndarray) -> np.ndarray:
    has = (y != 0).any(axis=1)
    rel = np.argmax(y != 0, axis=1)
    rel[~has] = y.shape[1]  # never released: no windows
    return rel


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_cfg(cfg: dict[str, Any], n_series: int) -> PatchTSTConfig:
    p = cfg["patchtst"]
    return PatchTSTConfig(
        lookback=p["lookback"], horizon=cfg["data"]["horizon"], patch_len=p["patch_len"], stride=p["stride"],
        d_model=p["d_model"], n_heads=p["n_heads"], n_layers=p["n_layers"], d_ff=p["d_ff"],
        dropout=p["dropout"], n_series=n_series if p["use_series_embedding"] else 0,
    )


def _loss_fn(cfg: dict[str, Any]):
    p = cfg["patchtst"]
    if p["loss"] == "tweedie":
        return lambda y, mu: tweedie_deviance(y, mu, p["tweedie_power"])
    return lambda y, mu: torch.mean((y - mu) ** 2)


@torch.no_grad()
def forecast_last_window(model: PatchTST, y: np.ndarray, t_end: int, lookback: int, batch_size: int = 2048) -> np.ndarray:
    """Forecast the `horizon` days after day index t_end (exclusive), for every series."""
    model.eval()
    dev = next(model.parameters()).device
    hist = torch.from_numpy(y[:, t_end - lookback : t_end].astype(np.float32))
    out = []
    for start in range(0, len(hist), batch_size):
        xb = hist[start : start + batch_size].to(dev)
        idx = torch.arange(start, start + len(xb), device=dev)
        out.append(model(xb, idx).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_model(
    y: np.ndarray, release: np.ndarray, cfg: dict[str, Any], t_train_end: int,
    y_valid: np.ndarray | None, artifacts_dir: Path, tag: str,
) -> tuple[PatchTST, dict[str, Any]]:
    """Train on windows fully inside [0, t_train_end). Early-stop on the validation window loss."""
    p = cfg["patchtst"]
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])
    dev = _device()
    mcfg = _model_cfg(cfg, y.shape[0])
    model = PatchTST(mcfg).to(dev)
    log.info("PatchTST %s | %d params | %s", tag, count_parameters(model), receptive_field_note(mcfg))

    ds = WindowDataset(y, release, mcfg.lookback, mcfg.horizon, t_train_end, p["sample_stride"], device=dev)
    log.info("training windows: %d", len(ds))
    if len(ds) == 0:
        raise ValueError("no training windows: reduce lookback or increase history")
    steps_per_epoch = -(-len(ds) // p["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=max(1, p["epochs"] * steps_per_epoch))
    loss_fn = _loss_fn(cfg)

    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(p["epochs"]):
        model.train()
        tot, n = 0.0, 0
        for xb, yb, ib in ds.batches(p["batch_size"], shuffle=True):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(yb, model(xb, ib))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}; lower lr or use mse loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        rec = {"epoch": epoch, "train_loss": tot / n}
        if y_valid is not None:
            pred = forecast_last_window(model, y, t_train_end, mcfg.lookback)
            vloss = float(loss_fn(torch.from_numpy(y_valid.astype(np.float32)), torch.from_numpy(pred)))
            rec["valid_loss"] = vloss
            if vloss < best - 1e-6:
                best, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        history.append(rec)
        log.info("epoch %d: %s", epoch, {k: round(v, 5) for k, v in rec.items() if k != "epoch"})
        if y_valid is not None and bad >= p["early_stopping_patience"]:
            log.info("early stopping at epoch %d", epoch)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "config": asdict(mcfg)}, artifacts_dir / f"patchtst_{tag}.pt")
    with open(artifacts_dir / f"patchtst_{tag}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return model, {"best_valid_loss": best, "epochs_run": len(history)}


def _subsample(meta: pd.DataFrame, y: np.ndarray, max_series: int | None, seed: int):
    if max_series is None or max_series >= len(meta):
        return meta, y
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(meta), size=max_series, replace=False))
    return meta.iloc[keep].reset_index(drop=True), y[keep]


def cross_validate(con: Any, cfg: dict[str, Any], last_day: int, artifacts_dir: str | Path) -> dict[str, Any]:
    horizon = cfg["data"]["horizon"]
    p = cfg["patchtst"]
    artifacts_dir = Path(artifacts_dir)
    folds = rolling_origin_folds(last_day, horizon, cfg["cv"]["n_folds"], cfg["cv"]["gap"])
    meta_all, y_all = load_wide(con, last_day)
    meta_all, y_all = _subsample(meta_all, y_all, p["max_series"], p["seed"])
    prices, cal = load_prices(con), load_calendar(con)
    release = release_days(y_all)
    results = []
    for fold in folds:
        y_hist = y_all[:, : fold.train_end]
        y_valid = y_all[:, fold.valid_start - 1 : fold.valid_end]
        model, info = train_model(y_all, release, cfg, fold.train_end, y_valid, artifacts_dir, tag=fold.name)
        pred = forecast_last_window(model, y_all, fold.train_end, p["lookback"])
        pred[release >= fold.train_end] = 0.0  # not yet released at training end
        dollars = dollar_sales_last_days(y_all, meta_all, prices, cal, last_day=fold.train_end)
        score: WRMSSEResult = wrmsse(y_hist, y_valid, pred, meta_all, dollars)
        log.info("fold %s:\n%s", fold.name, score)
        results.append({"fold": fold.name, **info, "wrmsse": score.total, "per_level": score.per_level})
        pd.DataFrame(pred, columns=[f"F{i}" for i in range(1, horizon + 1)]).assign(id=meta_all["id"]).to_csv(
            artifacts_dir / f"patchtst_pred_{fold.name}.csv", index=False
        )
    summary = {"folds": results, "mean_wrmsse": float(np.mean([r["wrmsse"] for r in results]))}
    with open(artifacts_dir / "patchtst_cv.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("PatchTST mean WRMSSE: %.4f", summary["mean_wrmsse"])
    return summary


def fit_final_and_predict(con: Any, cfg: dict[str, Any], last_day: int, artifacts_dir: str | Path) -> pd.DataFrame:
    """Train on all history (no validation; epochs fixed by config) and forecast the next horizon."""
    p = cfg["patchtst"]
    artifacts_dir = Path(artifacts_dir)
    meta, y = load_wide(con, last_day)
    release = release_days(y)
    model, _ = train_model(y, release, cfg, last_day, None, artifacts_dir, tag="final")
    pred = forecast_last_window(model, y, last_day, p["lookback"])
    pred[release >= last_day] = 0.0
    sub = pd.DataFrame(pred, columns=[f"F{i}" for i in range(1, cfg["data"]["horizon"] + 1)])
    sub.insert(0, "id", meta["id"].to_numpy())
    return sub

