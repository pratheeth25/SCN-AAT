"""PSO optimiser and rule-count search."""
import numpy as np
import pytest

from src.optimization.pso_rule_optimizer import (
    AntecedentCodec, IT2AntecedentObjective, PSOConfig, ParticleSwarmOptimizer, optimize_antecedents,
)
from src.optimization.rule_count_search import (
    search_rule_count, select_ridge_lambda, select_rule_count,
)
from src.models.it2_anfis import IT2_TSK_ANFIS


def sphere(x):
    return float(np.sum((x - 0.3) ** 2))


def test_pso_minimises_sphere_within_bounds():
    lb, ub = -np.ones(5), np.ones(5)
    res = ParticleSwarmOptimizer(PSOConfig(n_particles=20, max_iter=60, patience=60, seed=1)).minimize(sphere, lb, ub)
    assert res.best_f < 1e-3
    assert np.all(res.best_x >= lb) and np.all(res.best_x <= ub)
    # global best never gets worse
    best = [h["best_fitness"] for h in res.history]
    assert all(b2 <= b1 for b1, b2 in zip(best, best[1:]))


def test_pso_reproducible_and_early_stops():
    cfg = PSOConfig(n_particles=6, max_iter=200, patience=3, tol=1e9, seed=4)
    lb, ub = -np.ones(3), np.ones(3)
    r1 = ParticleSwarmOptimizer(cfg).minimize(sphere, lb, ub)
    r2 = ParticleSwarmOptimizer(cfg).minimize(sphere, lb, ub)
    assert np.array_equal(r1.best_x, r2.best_x)
    assert r1.n_iter == 3                       # huge tol -> never "improves" -> stops at patience


def test_pso_handles_nonfinite_and_bad_bounds():
    res = ParticleSwarmOptimizer(PSOConfig(n_particles=5, max_iter=3, seed=0)).minimize(
        lambda x: np.nan if x[0] > 0 else sphere(x), -np.ones(2), np.ones(2))
    assert np.isfinite(res.best_f)
    with pytest.raises(ValueError):
        ParticleSwarmOptimizer().minimize(sphere, np.ones(2), -np.ones(2))


def test_x0_seeds_the_swarm():
    x0 = np.full(4, 0.3)
    res = ParticleSwarmOptimizer(PSOConfig(n_particles=4, max_iter=0)).minimize(sphere, -np.ones(4), np.ones(4), x0=x0)
    assert res.best_f == pytest.approx(0.0)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((240, 3))
    y = 1000 + 50 * np.sin(2 * X[:, 0]) + 30 * X[:, 1] + rng.normal(0, 5, 240)
    return X[:160], y[:160], X[160:200], y[160:200], X[200:], y[200:]


def test_codec_roundtrip_and_constraints():
    codec = AntecedentCodec(4, np.full(3, -2.0), np.full(3, 2.0))
    rng = np.random.default_rng(1)
    for _ in range(20):
        c1, c2, s = codec.decode(codec.lb + rng.random(codec.dim) * (codec.ub - codec.lb))
        assert c1.shape == (4, 3) and np.all(c1 <= c2) and np.all(s > 0)
    m = IT2_TSK_ANFIS(4, X=rng.standard_normal((50, 3)), verbose=False, device="cpu")
    ant = m.get_antecedents()
    back = codec.decode(codec.encode(**ant))
    for a, b in zip((ant["c1"], ant["c2"], ant["sigma"]), back):
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_pso_improves_validation_fitness_over_lhs(data):
    Xtr, ytr, Xva, yva, _, _ = data
    m, res, obj = optimize_antecedents(4, Xtr, ytr, Xva, yva, ridge_lambda=1.0,
                                       config=PSOConfig(n_particles=8, max_iter=8, seed=0))
    assert res.history[-1]["best_fitness"] <= res.history[0]["best_fitness"]
    # model left in the best state
    assert np.mean((m.predict(Xva) - yva) ** 2) / np.var(yva) == pytest.approx(res.best_f, rel=1e-4)


def test_objective_complexity_penalty(data):
    Xtr, ytr, Xva, yva, _, _ = data
    m = IT2_TSK_ANFIS(3, X=Xtr, verbose=False, device="cpu")
    x = None
    o0 = IT2AntecedentObjective(m, Xtr, ytr, Xva, yva, 1.0, lambda_rule=0.0)
    x = o0.codec.encode(**m.get_antecedents())
    o1 = IT2AntecedentObjective(m, Xtr, ytr, Xva, yva, 1.0, lambda_rule=0.01)
    assert o1(x) - o0(x) == pytest.approx(0.03)


def test_rule_count_search_and_selection(data):
    Xtr, ytr, Xva, yva, Xte, yte = data
    rows = search_rule_count(Xtr, ytr, Xva, yva, [2, 4], config=PSOConfig(n_particles=4, max_iter=2),
                             ridge_lambda=1.0, X_report=Xte, y_report=yte)
    assert [r["K"] for r in rows] == [2, 4] and all("report_mse" in r for r in rows)
    # huge complexity weight always picks the smallest K; zero weight picks the best val MSE
    assert select_rule_count(rows, 10.0) == 2
    assert select_rule_count(rows, 0.0) == min(rows, key=lambda r: r["val_nmse"])["K"]


def test_select_ridge_uses_grid(data):
    Xtr, ytr, *_ = data
    assert select_ridge_lambda(Xtr, ytr, grid=(0.1, 10.0), n_rules=3, folds=3) in (0.1, 10.0)
