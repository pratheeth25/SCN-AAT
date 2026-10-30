"""Canonical regression evaluation shared by the CLI, comparison and sweep code.

Everything here works on *original-unit* arrays. Nothing is rescaled implicitly:
unit changes only happen through the explicit :func:`convert_units`.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Multiplicative factors to the base unit (kWh).
_TO_KWH = {"wh": 1e-3, "kwh": 1.0, "mwh": 1e3}


def convert_units(values, src: str, dst: str):
    """Explicit unit conversion, e.g. convert_units(y, "kWh", "MWh") == y / 1000."""
    try:
        factor = _TO_KWH[src.lower()] / _TO_KWH[dst.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown unit {exc}; supported: Wh, kWh, MWh") from None
    return np.asarray(values, dtype=np.float64) * factor


def describe(values, label: str = "") -> dict:
    """Summary statistics used for the stage-by-stage target/prediction prints."""
    v = np.asarray(values, dtype=np.float64).ravel()
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"label": label, "n": int(v.size), "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "std": float(v.std()),
            "q05": float(q[0]), "q25": float(q[1]), "q50": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4])}


def _as_1d(a, name):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2 and 1 in a.shape:
        a = a.ravel()
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D (or a column/row vector), got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return a


def evaluate_predictions(y_true, y_pred, unit: str = "kWh") -> dict:
    """
    Compute R², MSE, RMSE, MAE (scikit-learn) for predictions already in the same
    unit as ``y_true``. ``unit`` is only a label ("kWh" -> MSE in kWh²); it never
    rescales anything.

    Returns a flat dict: metrics, ``unit``, ``mse_unit`` and actual/predicted
    min/max/mean/std (``true_*`` / ``pred_*``) so a mismatched prediction scale
    (e.g. a forgotten or doubled inverse transform) is visible immediately.
    """
    yt, yp = _as_1d(y_true, "y_true"), _as_1d(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"length mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    mse = float(mean_squared_error(yt, yp))
    out = {"r2": float(r2_score(yt, yp)), "mse": mse, "rmse": float(np.sqrt(mse)),
           "mae": float(mean_absolute_error(yt, yp)), "unit": unit, "mse_unit": f"{unit}^2"}
    for tag, arr in (("true", yt), ("pred", yp)):
        out.update({f"{tag}_min": float(arr.min()), f"{tag}_max": float(arr.max()),
                    f"{tag}_mean": float(arr.mean()), f"{tag}_std": float(arr.std())})
    return out


def compute_regression_metrics(y_true, y_pred):
    """Backward-compatible wrapper (adds MAPE); delegates to evaluate_predictions."""
    m = evaluate_predictions(y_true, y_pred, unit="native")
    yt, yp = _as_1d(y_true, "y_true"), _as_1d(y_pred, "y_pred")
    mask = yt != 0
    m["mape"] = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.any() else 0.0
    return m


# ---------------------------------------------------------------------------
# Prediction-interval metrics
# ---------------------------------------------------------------------------

def interval_metrics(y_true, lower, upper, alpha: float = 0.1, y_range: float | None = None) -> dict:
    """
    PICP, MPIW, NMPIW and the Gneiting-Raftery interval score for [lower, upper].

    interval_score = mean( (u - l) + 2/alpha * (l - y) 1[y < l] + 2/alpha * (y - u) 1[y > u] )
    (lower is better; it penalises both width and misses, so narrow intervals with
    poor coverage are not rewarded). ``alpha`` is the nominal miss rate the score is
    evaluated at; it does not make the interval calibrated. ``y_range`` normalises
    MPIW (pass a training-set range to avoid using test information).
    """
    y, lo, hi = (_as_1d(a, n) for a, n in ((y_true, "y_true"), (lower, "lower"), (upper, "upper")))
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower and upper must have the same length")
    if np.any(lo > hi):
        raise ValueError("lower bound exceeds upper bound")
    width = hi - lo
    below, above = y < lo, y > hi
    score = width + (2.0 / alpha) * ((lo - y) * below + (y - hi) * above)
    rng = float(y_range) if y_range else float(y.max() - y.min()) or 1.0
    return {"picp": float(np.mean(~below & ~above)), "mpiw": float(width.mean()),
            "nmpiw": float(width.mean() / rng), "interval_score": float(score.mean()),
            "nominal_coverage": 1.0 - alpha}


def conformal_margin(y_cal, lower_cal, upper_cal, alpha: float = 0.1) -> float:
    """
    Split-conformal margin q for intervals [lower - q, upper + q] (CQR-style score
    s = max(lower - y, y - upper)). q is the ceil((n+1)(1-alpha))/n empirical
    quantile of the calibration scores. May be negative (interval shrinks).
    """
    y, lo, hi = (_as_1d(a, n) for a, n in ((y_cal, "y"), (lower_cal, "lower"), (upper_cal, "upper")))
    s = np.maximum(lo - y, y - hi)
    n = s.size
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))
