"""Rolling-origin validation folds.

Fold k (k = 0 is the most recent) validates on the `horizon` days ending at
`last_day - k * horizon` and trains on everything up to `gap` days before that window.
This mirrors the actual forecasting task: train on history, score on the next 28 days.
Random K-fold would leak future information through the lag features.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train_end: int      # last training day (inclusive)
    valid_start: int    # first validation day (inclusive)
    valid_end: int      # last validation day (inclusive)

    @property
    def name(self) -> str:
        return f"d{self.valid_start}-d{self.valid_end}"


def rolling_origin_folds(last_day: int, horizon: int, n_folds: int, gap: int = 0) -> list[Fold]:
    folds = []
    for k in range(n_folds):
        valid_end = last_day - k * horizon
        valid_start = valid_end - horizon + 1
        train_end = valid_start - 1 - gap
        if train_end < horizon:
            raise ValueError(f"fold {k}: not enough history (train_end={train_end})")
        folds.append(Fold(train_end=train_end, valid_start=valid_start, valid_end=valid_end))
    return folds
