"""Configuration loading and validation.

The config is a nested dict loaded from YAML. `--set a.b.c=value` overrides from the CLI are
applied with type coercion based on the existing value. Keeping this plain (no pydantic) keeps
the dependency footprint small; `validate_config` enforces the invariants the pipeline relies on.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config at {path} is not a mapping")
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        apply_override(cfg, item)
    validate_config(cfg)
    return cfg


def _coerce(old: Any, new: str) -> Any:
    if new.lower() in {"null", "none"}:
        return None
    if isinstance(old, bool):
        return new.lower() in {"1", "true", "yes"}
    if isinstance(old, int) and not isinstance(old, bool):
        return int(new)
    if isinstance(old, float):
        return float(new)
    if isinstance(old, list):
        return yaml.safe_load(new)
    if old is None:
        return yaml.safe_load(new)
    return new


def apply_override(cfg: dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ConfigError(f"Override must look like key.sub=value, got {item!r}")
    key, value = item.split("=", 1)
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise ConfigError(f"Unknown config section {p!r} in {key!r}")
        node = node[p]
    leaf = parts[-1]
    if leaf not in node:
        raise ConfigError(f"Unknown config key {key!r}")
    node[leaf] = _coerce(node[leaf], value)


def validate_config(cfg: dict[str, Any]) -> None:
    horizon = cfg["data"]["horizon"]
    if horizon <= 0:
        raise ConfigError("data.horizon must be positive")
    bad = [lag for lag in cfg["features"]["lags"] if lag < horizon]
    if bad:
        raise ConfigError(
            f"features.lags {bad} are smaller than the horizon {horizon}: "
            "a direct multi-horizon model would leak future sales."
        )
    if cfg["cv"]["n_folds"] < 1:
        raise ConfigError("cv.n_folds must be >= 1")
    p = cfg["patchtst"]
    if p["patch_len"] > p["lookback"]:
        raise ConfigError("patchtst.patch_len must be <= lookback")
    if not 1.0 < cfg["lgbm"]["tweedie_variance_power"] < 2.0:
        raise ConfigError("lgbm.tweedie_variance_power must be in (1, 2)")
