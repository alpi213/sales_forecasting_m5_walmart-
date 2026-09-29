import pytest

from m5.config import ConfigError, apply_override, load_config, validate_config


def test_override_types(cfg):
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    apply_override(c, "lgbm.learning_rate=0.1")
    apply_override(c, "patchtst.max_series=2000")
    apply_override(c, "features.lags=[28, 56]")
    assert c["lgbm"]["learning_rate"] == 0.1 and c["patchtst"]["max_series"] == 2000
    assert c["features"]["lags"] == [28, 56]


def test_unknown_key_rejected(cfg):
    with pytest.raises(ConfigError):
        apply_override(dict(cfg), "lgbm.not_a_key=1")


def test_leaky_lag_rejected(cfg):
    c = load_config("configs/default.yaml")
    c["features"]["lags"] = [7, 28]
    with pytest.raises(ConfigError, match="leak"):
        validate_config(c)
