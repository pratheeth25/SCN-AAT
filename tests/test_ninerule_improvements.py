"""9-rule improvement round: train+validation refit and the seasonal 9-rule model."""
import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src.experiments.framework import REGISTRY
from src.experiments.seasonal9 import N_RULES, SEASON_COLS, fit_seasonal9, seasonal_matrix
from src.experiments.temporal import refit_on_pretest
from src.models.it2_anfis import IT2_TSK_ANFIS


def _frame(n=420, seed=0):
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2015-01-01", periods=n)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                       "year": dt.year, "month": dt.month, "day": dt.day})
    season = np.sin(2 * np.pi * dt.dayofyear / 365.25)
    df[TARGET_COLUMN] = 2.6e5 + 2e4 * df.x1 + 3e4 * season + rng.normal(0, 4e3, n)
    return df


def _ridge_model(X, y, tr, lam=10.0):
    m = IT2_TSK_ANFIS(4, X=X[tr], verbose=False, random_state=0, device="cpu")
    m.fit(X[tr], y[tr], X[tr], y[tr], epochs=0, ridge_lambda=lam, verbose=False)
    return m


def test_refit_changes_only_consequents_and_ignores_test_rows():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 3)); y = 2e5 + 3e4 * X[:, 0] + rng.normal(0, 3e3, 300)
    tr, va, te = np.arange(0, 180), np.arange(180, 240), np.arange(240, 300)
    m = _ridge_model(X, y, tr)
    before = {k: v.copy() for k, v in m.get_antecedents().items()}
    r1 = refit_on_pretest(m, X, y, np.concatenate([tr, va]), 10.0)
    y_bad = y.copy(); y_bad[te] = 1e9                                  # corrupt only the test targets
    r2 = refit_on_pretest(m, X, y_bad, np.concatenate([tr, va]), 10.0)
    np.testing.assert_array_equal(r1.predict(X[te]), r2.predict(X[te]))   # test targets never used
    for k, v in r1.get_antecedents().items():
        np.testing.assert_allclose(v, before[k])                         # antecedents untouched
    assert not np.allclose(r1.predict(X[te]), m.predict(X[te]))          # consequents did change
    np.testing.assert_allclose(m.get_antecedents()["c1"], before["c1"])  # original model not mutated
    assert (r1.y_mean_, r1.y_scale_) == (m.y_mean_, m.y_scale_)


def test_seasonal_matrix_uses_train_rows_and_has_no_year():
    df = _frame()
    X, cols = seasonal_matrix(df, np.arange(250))
    assert "year" not in cols and set(SEASON_COLS) <= set(cols) and "t" not in cols
    assert X.shape == (len(df), len(cols)) and np.all(np.isfinite(X))


def test_seasonal9_rules_live_in_the_season_space_and_predict_finitely():
    df = _frame()
    y = df[TARGET_COLUMN].to_numpy(float)
    tr, va, te = np.arange(0, 250), np.arange(250, 320), np.arange(320, 420)
    X, cols = seasonal_matrix(df, tr)
    m, sec, cfg = fit_seasonal9(X, cols, y, tr, va, 0,
                                pso_budget={"n_particles": 6, "max_iter": 3, "patience": 3})
    ant = m.get_antecedents()
    assert ant["c1"].shape == (N_RULES, len(SEASON_COLS))
    assert np.all(ant["c1"] <= ant["c2"]) and np.all(ant["sigma"] > 0)
    pred = m.predict(X[te])
    assert np.all(np.isfinite(pred)) and abs(pred.mean() - y[tr].mean()) < 0.2 * y[tr].mean()
    assert cfg["pso_evals"] > 0 and cfg["sigma_scale"] in (0.3, 0.5, 1.0)
    m2, _, cfg2 = fit_seasonal9(X, cols, y, tr, va, 0,
                                pso_budget={"n_particles": 6, "max_iter": 3, "patience": 3})
    np.testing.assert_array_equal(pred, m2.predict(X[te]))             # reproducible


def test_new_models_are_registered():
    for n in ("M9_tscv_refit", "M7_noyear_ridge_refit", "M9_season", "M9_season_pso"):
        assert n in REGISTRY and REGISTRY[n].fuzzy
