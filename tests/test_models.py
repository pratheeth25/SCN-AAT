"""
Comprehensive unit tests for IT2-ANFIS and baseline models.

Run with:
    python -m pytest tests/ -v
or:
    python -m pytest tests/test_models.py -v

All tests use small synthetic regression datasets so they run in seconds
without the real dataset.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.it2_anfis import IT2_TSK_ANFIS, _IT2TSKNetwork
from src.models.anfis_zero import ZeroOrderANFIS
from src.models.anfis_first import FirstOrderANFIS

import torch


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_data():
    """Small synthetic regression dataset in deliberately large-scale units."""
    rng = np.random.default_rng(0)
    N, F = 120, 5
    X = rng.standard_normal((N, F))
    # Targets on a KWh-like scale (~200 000) to match real use-case
    y = 2e5 + 3e4 * X[:, 0] - 2e4 * X[:, 1] + rng.standard_normal(N) * 5e3
    return X, y


@pytest.fixture
def it2_model(synth_data):
    """Fitted IT2-ANFIS with default settings."""
    X, y = synth_data
    model = IT2_TSK_ANFIS(n_rules=3, X=X[:80], verbose=False)
    model.fit(X[:80], y[:80], X[80:100], y[80:100], epochs=10, patience=10)
    return model, X, y


# ---------------------------------------------------------------------------
# Membership function tests
# ---------------------------------------------------------------------------

class TestMembershipFunctions:
    """Verify IT2 LMF/UMF properties."""

    def test_lmf_leq_umf(self, synth_data):
        """Per-feature LMF must be <= UMF for all inputs, rules, and features.

        This property is defined per-feature *before* the product t-norm
        is applied across features.  The network's _compute_memberships
        returns the per-rule product, so we probe a single-feature network.
        """
        rng = np.random.default_rng(42)
        N = 200
        X1 = rng.standard_normal((N, 1))  # 1 feature
        model = IT2_TSK_ANFIS(n_rules=3, X=X1, verbose=False)
        Z = torch.tensor(model._normalize_X(X1), dtype=torch.float32)
        mu_L, mu_U = model._net._compute_memberships(Z)
        # With F=1 the product equals the single-feature value -> ordering holds
        assert (mu_L.detach() <= mu_U.detach() + 1e-5).all(), (
            "Per-feature LMF must be <= UMF everywhere"
        )

    def test_membership_nonnegative(self, synth_data):
        X, y = synth_data
        model = IT2_TSK_ANFIS(n_rules=3, X=X, verbose=False)
        Z = torch.tensor(model._normalize_X(X), dtype=torch.float32)
        mu_L, mu_U = model._net._compute_memberships(Z)
        assert (mu_L >= 0).all()
        assert (mu_U >= 0).all()

    def test_umf_leq_one(self, synth_data):
        X, y = synth_data
        model = IT2_TSK_ANFIS(n_rules=3, X=X, verbose=False)
        Z = torch.tensor(model._normalize_X(X), dtype=torch.float32)
        _, mu_U = model._net._compute_memberships(Z)
        assert (mu_U <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# Firing strength tests
# ---------------------------------------------------------------------------

class TestFiringStrengths:
    def test_firing_sums_to_one(self, synth_data):
        X, y = synth_data
        model = IT2_TSK_ANFIS(n_rules=3, X=X, verbose=False)
        Z = torch.tensor(model._normalize_X(X), dtype=torch.float32)
        f_L, f_U = model._net.firing_weights(Z)
        np.testing.assert_allclose(
            f_L.detach().sum(dim=1).numpy(), np.ones(len(X)), atol=1e-5
        )
        np.testing.assert_allclose(
            f_U.detach().sum(dim=1).numpy(), np.ones(len(X)), atol=1e-5
        )

    def test_firing_nonnegative(self, synth_data):
        X, y = synth_data
        model = IT2_TSK_ANFIS(n_rules=3, X=X, verbose=False)
        Z = torch.tensor(model._normalize_X(X), dtype=torch.float32)
        f_L, f_U = model._net.firing_weights(Z)
        assert (f_L >= 0).all()
        assert (f_U >= 0).all()


# ---------------------------------------------------------------------------
# Prediction shape and finiteness
# ---------------------------------------------------------------------------

class TestPredictions:
    def test_predict_shape(self, it2_model):
        model, X, y = it2_model
        pred = model.predict(X[100:])
        assert pred.shape == (20,), f"Expected (20,), got {pred.shape}"

    def test_predict_finite(self, it2_model):
        model, X, y = it2_model
        pred = model.predict(X[100:])
        assert np.all(np.isfinite(pred)), "Predictions contain NaN or Inf"

    def test_interval_shape(self, it2_model):
        model, X, y = it2_model
        lo, hi = model.predict_interval(X[100:])
        assert lo.shape == (20,)
        assert hi.shape == (20,)

    def test_interval_ordering(self, it2_model):
        """Lower bound must be <= upper bound."""
        model, X, y = it2_model
        lo, hi = model.predict_interval(X[100:])
        assert np.all(lo <= hi + 1e-6), "Interval lower > upper"

    def test_interval_finite(self, it2_model):
        model, X, y = it2_model
        lo, hi = model.predict_interval(X[100:])
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))

    def test_predictions_in_original_units(self, it2_model):
        """Predictions should be in the same magnitude as y (~2e5), not normalised."""
        model, X, y = it2_model
        pred = model.predict(X[100:])
        # If target scaler wasn't applied, predictions would be near 0..1
        assert pred.mean() > 1e3, (
            f"Predictions look normalised; mean={pred.mean():.2f} expected ~2e5"
        )


# ---------------------------------------------------------------------------
# Target inverse-transform
# ---------------------------------------------------------------------------

class TestTargetScaling:
    def test_y_scale_fitted_only_on_train(self, synth_data):
        X, y = synth_data
        X_tr, y_tr = X[:80], y[:80]
        model = IT2_TSK_ANFIS(n_rules=3, X=X_tr, verbose=False)
        model.fit(X_tr, y_tr, epochs=5, patience=5)
        # y_mean_ / y_scale_ must match train-set statistics
        np.testing.assert_allclose(model.y_mean_, y_tr.mean(), rtol=1e-5)
        np.testing.assert_allclose(model.y_scale_, y_tr.std(), rtol=1e-5)

    def test_denormalize_roundtrip(self, synth_data):
        X, y = synth_data
        model = IT2_TSK_ANFIS(n_rules=3, X=X[:80], verbose=False)
        model.fit(X[:80], y[:80], epochs=3, patience=3)
        yn = model._normalize_y(y[:5])
        y_back = model._denormalize_y(yn)
        np.testing.assert_allclose(y_back, y[:5], rtol=1e-5)


# ---------------------------------------------------------------------------
# Early stopping & checkpoint restoration
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_best_epoch_recorded(self, synth_data):
        X, y = synth_data
        model = IT2_TSK_ANFIS(n_rules=3, X=X[:80], verbose=False)
        model.fit(X[:80], y[:80], X[80:100], y[80:100],
                  epochs=20, patience=5)
        assert hasattr(model, "best_epoch_")
        assert 1 <= model.best_epoch_ <= 20

    def test_history_length(self, synth_data):
        X, y = synth_data
        model = IT2_TSK_ANFIS(n_rules=3, X=X[:80], verbose=False)
        model.fit(X[:80], y[:80], epochs=8, patience=10)
        assert len(model.history_["train_mse"]) == 8


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_pred(self, synth_data):
        """Same numpy + torch seed -> identical predictions."""
        X, y = synth_data
        kwargs = dict(n_rules=3, verbose=False, random_state=7)
        fit_kwargs = dict(epochs=5, patience=5)

        torch.manual_seed(0)
        m1 = IT2_TSK_ANFIS(X=X[:80], **kwargs)
        m1.fit(X[:80], y[:80], **fit_kwargs)
        p1 = m1.predict(X[80:])

        torch.manual_seed(0)
        m2 = IT2_TSK_ANFIS(X=X[:80], **kwargs)
        m2.fit(X[:80], y[:80], **fit_kwargs)
        p2 = m2.predict(X[80:])

        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Metric correctness
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_r2_improves_above_negative_40(self, synth_data):
        """The fixed model should not produce R² = -40 on a simple linear task."""
        X, y = synth_data
        model = IT2_TSK_ANFIS(n_rules=5, X=X[:80], verbose=False)
        model.fit(X[:80], y[:80], X[80:100], y[80:100],
                  epochs=50, patience=20)
        from sklearn.metrics import r2_score
        r2 = r2_score(y[100:], model.predict(X[100:]))
        assert r2 > -1.0, f"R² still catastrophically negative: {r2:.4f}"


# ---------------------------------------------------------------------------
# End-to-end small pipeline test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_pipeline_runs(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((80, 4))
        y = 1e5 + 2e4 * X[:, 0] + rng.standard_normal(80) * 3e3
        model = IT2_TSK_ANFIS(n_rules=3, verbose=False)
        model.fit(X[:60], y[:60], X[60:70], y[60:70], epochs=10, patience=10)
        pred = model.predict(X[70:])
        lo, hi = model.predict_interval(X[70:])
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))
        assert np.all(lo <= hi + 1e-6)


# ---------------------------------------------------------------------------
# Baseline model tests
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_zero_order_anfis(self, synth_data):
        X, y = synth_data
        m = ZeroOrderANFIS(n_rules=3)
        m.fit(X[:80], y[:80])
        pred = m.predict(X[80:])
        assert pred.shape == (40,)
        assert np.all(np.isfinite(pred))

    def test_first_order_anfis(self, synth_data):
        X, y = synth_data
        m = FirstOrderANFIS(n_rules=3)
        m.fit(X[:80], y[:80])
        pred = m.predict(X[80:])
        assert pred.shape == (40,)
        assert np.all(np.isfinite(pred))
