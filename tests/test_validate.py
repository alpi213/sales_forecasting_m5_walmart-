import pytest

from m5.data.validate import DataValidationError, validate_all


def test_synthetic_passes(raw):
    validate_all(raw["sales"], raw["calendar"], raw["prices"])


def test_negative_sales_rejected(raw):
    s = raw["sales"].copy()
    s.loc[0, "d_10"] = -1
    with pytest.raises(DataValidationError, match="negative"):
        validate_all(s, raw["calendar"], raw["prices"])


def test_unknown_price_pair_rejected(raw):
    p = raw["prices"].copy()
    p.loc[len(p)] = ["CA_1", "NOPE_1_001", int(p["wm_yr_wk"].iloc[0]), 1.0]
    with pytest.raises(DataValidationError, match="not present"):
        validate_all(raw["sales"], raw["calendar"], p)


def test_calendar_too_short_rejected(raw):
    with pytest.raises(DataValidationError):
        n = sum(c.startswith("d_") for c in raw["sales"].columns)
        validate_all(raw["sales"], raw["calendar"].iloc[: n - 5], raw["prices"])
