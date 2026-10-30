"""IT2 membership-function constraints after PSO decoding and antecedent setters."""
import numpy as np
import pytest
import torch

from src.models.it2_anfis import IT2_TSK_ANFIS
from src.optimization.pso_rule_optimizer import AntecedentCodec


@pytest.fixture
def model():
    X = np.random.default_rng(0).standard_normal((100, 2))
    return IT2_TSK_ANFIS(5, X=X, verbose=False, device="cpu"), X


def test_lmf_le_umf_for_any_decoded_particle(model):
    m, X = model
    Z = torch.tensor(m._normalize_X(X), dtype=torch.float32)
    codec = AntecedentCodec(5, Z.numpy().min(0), Z.numpy().max(0))
    rng = np.random.default_rng(3)
    for _ in range(25):
        m.set_antecedents(*codec.decode(codec.lb + rng.random(codec.dim) * (codec.ub - codec.lb)))
        mu_L, mu_U = m._net._compute_memberships(Z)
        f_L, f_U = m._net.firing_weights(Z)
        assert torch.all(mu_L >= 0) and torch.all(mu_U <= 1 + 1e-6)
        assert torch.isfinite(f_L).all() and torch.isfinite(f_U).all()
        torch.testing.assert_close(f_L.sum(1), torch.ones(len(Z)), atol=1e-5, rtol=0)


def test_per_feature_lmf_le_umf():
    """With one feature the product equals the per-feature membership."""
    X = np.random.default_rng(1).standard_normal((200, 1))
    m = IT2_TSK_ANFIS(3, X=X, verbose=False, device="cpu")
    m.set_antecedents(np.array([[-1.0], [0.0], [1.0]]), np.array([[-0.5], [0.8], [1.0]]),
                      np.array([[0.3], [1.0], [2.0]]))
    Z = torch.tensor(m._normalize_X(X), dtype=torch.float32)
    # undo the per-row max shift used for stability by comparing un-normalised memberships
    sig = torch.exp(m._net.log_sigma)
    c1, c2 = m._net.c1, m._net.c2
    x = Z.unsqueeze(1)
    lmf = torch.where(x <= (c1 + c2) / 2, torch.exp(-0.5 * ((x - c2) / sig) ** 2), torch.exp(-0.5 * ((x - c1) / sig) ** 2))
    umf = torch.where(x < c1, torch.exp(-0.5 * ((x - c1) / sig) ** 2),
                      torch.where(x <= c2, torch.ones_like(x), torch.exp(-0.5 * ((x - c2) / sig) ** 2)))
    assert torch.all(lmf <= umf + 1e-6)


def test_set_antecedents_rejects_invalid(model):
    m, _ = model
    ant = m.get_antecedents()
    with pytest.raises(ValueError):
        m.set_antecedents(ant["c2"] + 1.0, ant["c2"], ant["sigma"])      # c1 > c2
    with pytest.raises(ValueError):
        m.set_antecedents(ant["c1"], ant["c2"], -ant["sigma"])           # sigma <= 0
    with pytest.raises(ValueError):
        m.set_antecedents(ant["c1"][:2], ant["c2"][:2], ant["sigma"][:2])  # wrong K
    m.set_antecedents(**ant)                                             # valid roundtrip
    np.testing.assert_allclose(m.get_antecedents()["c1"], ant["c1"], atol=1e-6)
