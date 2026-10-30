"""Canonical evaluation: values checked against hand-computed expectations."""
import numpy as np
import pytest

from src.utils.metrics import convert_units, describe, evaluate_predictions


def test_known_values():
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([1.0, 3.0, 3.0, 2.0])
    m = evaluate_predictions(yt, yp, unit="kWh")
    # errors 0,1,0,-2 -> MSE 5/4; MAE 3/4; SS_res 5, SS_tot 5 -> R2 0
    assert m["mse"] == pytest.approx(1.25)
    assert m["rmse"] == pytest.approx(np.sqrt(1.25))
    assert m["mae"] == pytest.approx(0.75)
    assert m["r2"] == pytest.approx(0.0)
    assert m["unit"] == "kWh" and m["mse_unit"] == "kWh^2"
    assert m["pred_max"] == 3.0 and m["true_min"] == 1.0
    assert m["true_mean"] == pytest.approx(2.5) and m["pred_std"] == pytest.approx(np.std(yp))


def test_perfect_and_mean_predictor():
    y = np.array([2.0, 4.0, 6.0])
    assert evaluate_predictions(y, y)["r2"] == pytest.approx(1.0)
    assert evaluate_predictions(y, np.full(3, y.mean()))["r2"] == pytest.approx(0.0)


def test_column_vector_flattened_and_validated():
    y = np.arange(5.0)
    assert evaluate_predictions(y, y.reshape(-1, 1))["mse"] == 0.0
    with pytest.raises(ValueError):
        evaluate_predictions(y, y[:4])
    with pytest.raises(ValueError):
        evaluate_predictions(y, np.r_[y[:4], np.nan])
    with pytest.raises(ValueError):
        evaluate_predictions(y, np.r_[y[:4], np.inf])


def test_unit_conversion_is_explicit_and_squares_mse():
    yt = np.array([100_000.0, 200_000.0, 300_000.0])   # kWh
    yp = np.array([110_000.0, 190_000.0, 330_000.0])
    kwh = evaluate_predictions(yt, yp, unit="kWh")
    mwh = evaluate_predictions(convert_units(yt, "kWh", "MWh"), convert_units(yp, "kWh", "MWh"), unit="MWh")
    assert kwh["mse"] == pytest.approx(np.mean((yt - yp) ** 2))
    assert mwh["mse"] == pytest.approx(kwh["mse"] / 1e6)
    assert mwh["rmse"] == pytest.approx(kwh["rmse"] / 1e3)
    assert mwh["r2"] == pytest.approx(kwh["r2"])            # unit-free
    # evaluate_predictions never rescales by itself
    assert evaluate_predictions(yt, yp, unit="MWh")["mse"] == pytest.approx(kwh["mse"])
    with pytest.raises(ValueError):
        convert_units(yt, "kWh", "furlong")


def test_describe():
    d = describe(np.arange(101.0), "x")
    assert d["n"] == 101 and d["q50"] == 50.0 and d["q95"] == 95.0 and d["min"] == 0.0


def test_it2_predicts_in_original_units():
    """One inverse transform: IT2 output must live on the scale of y, not of its standardised y."""
    from src.models.it2_anfis import IT2_TSK_ANFIS
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = 275_000 + 40_000 * X[:, 0]
    m = IT2_TSK_ANFIS(3, X=X, verbose=False)
    m.fit(X[:150], y[:150], X[150:], y[150:], epochs=20, patience=10)
    p = m.predict(X[150:])
    assert evaluate_predictions(y[150:], p)["pred_mean"] == pytest.approx(y.mean(), rel=0.1)
