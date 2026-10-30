"""Interval metrics (PICP, MPIW, NMPIW, interval score) and split-conformal margin."""
import numpy as np
import pytest

from src.utils.metrics import conformal_margin, interval_metrics


def test_known_interval_values():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    lo = np.array([0.0, 2.5, 2.0, 8.0])
    hi = np.array([2.0, 3.5, 4.0, 9.0])
    m = interval_metrics(y, lo, hi, alpha=0.5, y_range=10.0)
    # covered: rows 0, 2 -> 0.5; widths 2,1,2,1 -> 1.5
    assert m["picp"] == 0.5 and m["mpiw"] == 1.5 and m["nmpiw"] == pytest.approx(0.15)
    # score: widths + 2/0.5 * misses: row1 miss below by 0.5 -> +2, row3 above by 1 -> +4
    assert m["interval_score"] == pytest.approx((2 + 1 + 2 + 1 + 2 + 4) / 4)
    assert m["nominal_coverage"] == 0.5


def test_narrow_intervals_with_poor_coverage_are_penalised():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    narrow = interval_metrics(y, np.full_like(y, -0.01), np.full_like(y, 0.01), alpha=0.1)
    calibrated = interval_metrics(y, np.full_like(y, -1.645), np.full_like(y, 1.645), alpha=0.1)
    assert calibrated["picp"] == pytest.approx(0.9, abs=0.02)
    assert calibrated["interval_score"] < narrow["interval_score"]


def test_invalid_intervals_rejected():
    with pytest.raises(ValueError):
        interval_metrics([1.0, 2.0], [2.0, 1.0], [1.0, 3.0])      # lower > upper
    with pytest.raises(ValueError):
        interval_metrics([1.0, np.nan], [0.0, 0.0], [2.0, 2.0])


def test_conformal_margin_attains_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1)
    y_cal, y_new = rng.normal(0, 1, 2000), rng.normal(0, 1, 20000)
    point = np.zeros_like(y_cal)
    q = conformal_margin(y_cal, point, point, alpha=0.1)       # zero-width base interval
    cov = np.mean(np.abs(y_new) <= q)
    assert 0.88 <= cov <= 0.92
    # a base interval that already over-covers gets a negative (shrinking) margin
    assert conformal_margin(y_cal, point - 5, point + 5, alpha=0.1) < 0
