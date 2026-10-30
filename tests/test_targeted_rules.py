"""Targeted 9-rule extension: constraints, finiteness, units, reproducibility, baseline preservation."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import IT2_EPOCHS, IT2_LAMBDA_L1, IT2_LAMBDA_L2, IT2_PATIENCE
from src.experiments.compare_models import fit_baseline7
from src.experiments.targeted9 import _metrics
from src.models.it2_anfis import IT2_TSK_ANFIS
from src.optimization.pso_rule_optimizer import PSOConfig
from src.optimization.targeted_rules import (
    build_extended_model, cluster_fallback, find_residual_patterns, init_rules_from_rows,
    optimize_new_rules,
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    X = np.column_stack([rng.standard_normal((n, 3)), dates.month.to_numpy(float)])
    y = 275_000 + 30_000 * X[:, 0] + 25_000 * (dates.month == 7) + rng.normal(0, 5_000, n)  # kWh
    idx = rng.permutation(n)
    tr, va, te = idx[:250], idx[250:320], idx[320:]
    d = {"Xtr": X[tr], "ytr": y[tr], "Xva": X[va], "yva": y[va], "Xte": X[te], "yte": y[te]}
    df_dates = pd.DataFrame({"year": dates.year, "month": dates.month, "day": dates.day}).iloc[tr]
    return d, df_dates


@pytest.fixture(scope="module")
def base(data):
    d, _ = data
    return fit_baseline7(d, seed=0)


def test_baseline_procedure_unchanged(data, base):
    """fit_baseline7 must equal the original inline baseline recipe exactly."""
    d, _ = data
    torch.manual_seed(0)
    ref = IT2_TSK_ANFIS(7, X=d["Xtr"], verbose=False, random_state=0, device="cpu")
    ref.fit(d["Xtr"], d["ytr"], d["Xva"], d["yva"], epochs=IT2_EPOCHS, patience=IT2_PATIENCE,
            batch_size=64, freeze_antecedents=True, lambda_l1=IT2_LAMBDA_L1,
            lambda_l2=IT2_LAMBDA_L2, verbose=False)
    np.testing.assert_array_equal(ref.predict(d["Xte"]), base.predict(d["Xte"]))


def test_residual_pattern_detects_injected_month(data, base):
    d, dates = data
    res = d["ytr"] - base.predict(d["Xtr"])
    patterns, table = find_residual_patterns(res, dates, 2, t_min=3.0)
    assert patterns and patterns[0][0] == "month=7"
    assert {"group", "mean_residual", "t", "n"} <= set(table.columns)


def test_no_pattern_in_noise_and_cluster_fallback():
    rng = np.random.default_rng(1)
    dates = pd.DataFrame({"year": 2016, "month": np.repeat(np.arange(1, 13), 30), "day": 1})
    patterns, _ = find_residual_patterns(rng.normal(0, 1, 360), dates, 2, t_min=10.0)
    assert patterns == []
    fb = cluster_fallback(rng.standard_normal((360, 3)), rng.normal(0, 1, 360), 2, seed=0)
    assert len(fb) == 2 and all(m.sum() > 0 for _, m in fb)
    assert not np.any(fb[0][1] & fb[1][1])


def test_new_rule_constraints_and_baseline_rules_preserved(data, base):
    d, _ = data
    Z = base._normalize_X(d["Xtr"])
    masks = [d["Xtr"][:, 3] == 7, d["Xtr"][:, 0] > 1]
    new = init_rules_from_rows(Z, masks, base.get_antecedents())
    assert np.all(new[0] <= new[1]) and np.all(new[2] > 0)
    assert np.all(new[2] >= base.get_antecedents()["sigma"].mean(0) - 1e-12)
    before = base.predict(d["Xte"])
    m9 = build_extended_model(base, d["Xtr"], new, seed=0)
    a7, a9 = base.get_antecedents(), m9.get_antecedents()
    assert m9.n_rules == 9
    for k in a7:
        np.testing.assert_allclose(a9[k][:7], a7[k], atol=1e-6)
    np.testing.assert_array_equal(before, base.predict(d["Xte"]))   # base not mutated


def _run_pso(d, base, seed):
    Z = base._normalize_X(d["Xtr"])
    new = init_rules_from_rows(Z, [d["Xtr"][:, 3] == 7, d["Xtr"][:, 0] > 1], base.get_antecedents())
    m9 = build_extended_model(base, d["Xtr"], new, seed=seed)
    return optimize_new_rules(m9, 2, d["Xtr"], d["ytr"], d["Xva"], d["yva"], ridge_lambda=10.0,
                              config=PSOConfig(n_particles=6, max_iter=4, seed=seed))


def test_targeted_pso_constraints_fixed_rules_and_reproducible(data, base):
    d, _ = data
    best, res, _ = _run_pso(d, base, 3)
    best2, res2, _ = _run_pso(d, base, 3)
    for k in best:
        np.testing.assert_array_equal(best[k], best2[k])                 # reproducible
        np.testing.assert_allclose(best[k][:7], base.get_antecedents()[k], atol=1e-6)
    assert np.all(best["c1"] <= best["c2"]) and np.all(best["sigma"] > 0)
    assert res.best_f <= res.history[0]["best_fitness"]                  # never worse than init swarm


def test_finite_predictions_and_units(data, base):
    d, _ = data
    best, _, _ = _run_pso(d, base, 0)
    m = build_extended_model(base, d["Xtr"], (best["c1"][7:], best["c2"][7:], best["sigma"][7:]), 0)
    m.fit(d["Xtr"], d["ytr"], d["Xva"], d["yva"], epochs=0, ridge_lambda=10.0, verbose=False)
    pred = m.predict(d["Xte"])
    assert np.all(np.isfinite(pred))
    assert abs(pred.mean() - d["ytr"].mean()) < 0.2 * d["ytr"].mean()  # kWh scale, not standardised
    r = _metrics("x", m, d, 0.0, 2, False)
    assert r["mse"] == pytest.approx(np.mean((pred - d["yte"]) ** 2) / 1e6)   # kWh^2 -> MWh^2
    assert r["rmse"] == pytest.approx(np.sqrt(r["mse"]))
    assert r["rules"] == 9 and r["n_params_total"] == 3 * 9 * 4 + 9 * 5 + 1
