"""Particle Swarm Optimisation of IT2-ANFIS antecedents (membership functions / FOU).

Strategy (nested / hybrid, in the spirit of classic ANFIS hybrid learning):

* PSO searches the *antecedent* space only: for every rule j and input f an
  uncertain-mean centre m_jf, an FOU half-width h_jf >= 0 and a log spread
  log(sigma_jf). Decoding gives c1 = m - h, c2 = m + h, sigma = exp(.), so
  c1 <= c2 and sigma > 0 hold by construction (LMF <= UMF then follows from the
  membership definitions in ``it2_anfis``).
* For each particle the first-order TSK consequents are *solved*, not searched:
  with antecedents fixed the defuzzified output is linear in the consequents, so
  the ridge (L2) optimum is one small linear solve
  (``IT2_TSK_ANFIS._ridge_init_consequents``). Searching K*(F+1) consequent
  weights with PSO would multiply the dimension for no benefit.
* Fitness (validation only, never test):
      J = MSE_val / Var(y_val) + lambda_rule * K
  The first term is scale-free (= 1 - R^2_val); lambda_rule is the normalised
  validation-MSE reduction one extra rule must buy. For a fixed K the penalty
  is constant, so PSO's argmin does not depend on lambda_rule; lambda only
  matters when choosing K (see ``rule_count_search``).

The swarm is seeded with the repository's own LHS initialisation (particle 0)
and LHS draws from other random states, so the search starts from the paper's
initialisation rather than arbitrary points.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch

from ..models.it2_anfis import IT2_TSK_ANFIS


# ---------------------------------------------------------------------------
# Generic optimiser
# ---------------------------------------------------------------------------

@dataclass
class PSOConfig:
    """Global-best PSO with linearly decreasing inertia."""
    n_particles: int = 20
    max_iter: int = 40
    w_start: float = 0.9
    w_end: float = 0.4
    c_cognitive: float = 1.5
    c_social: float = 1.5
    vmax_frac: float = 0.2      # max |velocity| as a fraction of each box width
    patience: int = 10          # stop after this many iterations without improvement
    tol: float = 1e-6           # minimum improvement that resets patience
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PSOResult:
    best_x: np.ndarray
    best_f: float
    n_evals: int
    n_iter: int
    seconds: float
    history: list = field(default_factory=list)   # one dict per iteration


class ParticleSwarmOptimizer:
    """Box-constrained minimiser. Positions are clipped to [lb, ub] (absorbing walls)."""

    def __init__(self, config: PSOConfig | None = None):
        self.config = config or PSOConfig()

    def minimize(
        self,
        f: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        x0: np.ndarray | None = None,
        callback: Callable[[dict], None] | None = None,
    ) -> PSOResult:
        cfg = self.config
        lb, ub = np.asarray(lb, float), np.asarray(ub, float)
        if lb.shape != ub.shape or np.any(lb > ub):
            raise ValueError("bounds must have equal shape and lb <= ub")
        rng = np.random.default_rng(cfg.seed)
        P, D = int(cfg.n_particles), lb.size
        if P < 1 or cfg.max_iter < 0:
            raise ValueError("n_particles must be >= 1 and max_iter >= 0")
        width = ub - lb
        vmax = cfg.vmax_frac * width

        X = lb + rng.random((P, D)) * width
        if x0 is not None:
            x0 = np.atleast_2d(np.asarray(x0, float))
            k = min(len(x0), P)
            X[:k] = np.clip(x0[:k], lb, ub)
        V = rng.uniform(-1, 1, (P, D)) * vmax * 0.1

        t0 = time.perf_counter()
        fx = np.array([self._safe(f, x) for x in X])
        n_evals = P
        pbest, pbest_f = X.copy(), fx.copy()
        g = int(np.argmin(pbest_f))
        gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])
        history = [self._log(0, gbest_f, fx, n_evals, t0)]
        if callback:
            callback(history[-1])

        stale, it = 0, 0
        for it in range(1, int(cfg.max_iter) + 1):
            w = cfg.w_start + (cfg.w_end - cfg.w_start) * (it - 1) / max(cfg.max_iter - 1, 1)
            r1, r2 = rng.random((P, D)), rng.random((P, D))
            V = w * V + cfg.c_cognitive * r1 * (pbest - X) + cfg.c_social * r2 * (gbest - X)
            V = np.clip(V, -vmax, vmax)
            X = np.clip(X + V, lb, ub)
            fx = np.array([self._safe(f, x) for x in X])
            n_evals += P
            improved = fx < pbest_f
            pbest[improved], pbest_f[improved] = X[improved], fx[improved]
            g = int(np.argmin(pbest_f))
            if pbest_f[g] < gbest_f - cfg.tol:
                stale = 0
            else:
                stale += 1
            if pbest_f[g] < gbest_f:
                gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])
            history.append(self._log(it, gbest_f, fx, n_evals, t0))
            if callback:
                callback(history[-1])
            if stale >= cfg.patience:
                break
        return PSOResult(gbest, gbest_f, n_evals, it, time.perf_counter() - t0, history)

    @staticmethod
    def _safe(f, x) -> float:
        v = float(f(x))
        return v if np.isfinite(v) else np.inf

    @staticmethod
    def _log(it, best, fx, n_evals, t0) -> dict:
        finite = fx[np.isfinite(fx)]
        return {"iter": it, "best_fitness": float(best),
                "mean_fitness": float(finite.mean()) if finite.size else float("nan"),
                "n_evals": int(n_evals), "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# IT2-ANFIS antecedent encoding and objective
# ---------------------------------------------------------------------------

SIGMA_BOUNDS = (0.05, 5.0)     # normalised-input units
MAX_FOU_HALF_WIDTH = 1.0       # normalised-input units


class AntecedentCodec:
    """Maps a flat vector <-> (c1, c2, sigma), each of shape (K, F)."""

    def __init__(self, n_rules: int, z_min: np.ndarray, z_max: np.ndarray):
        self.K, self.F = int(n_rules), len(z_min)
        span = np.maximum(z_max - z_min, 1e-6)
        m_lb, m_ub = z_min - 0.1 * span, z_max + 0.1 * span
        tile = lambda a: np.tile(a, (self.K, 1)).ravel()
        n = self.K * self.F
        self.lb = np.concatenate([tile(m_lb), np.zeros(n), np.full(n, np.log(SIGMA_BOUNDS[0]))])
        self.ub = np.concatenate([tile(m_ub), np.full(n, MAX_FOU_HALF_WIDTH),
                                  np.full(n, np.log(SIGMA_BOUNDS[1]))])

    @property
    def dim(self) -> int:
        return 3 * self.K * self.F

    def decode(self, x: np.ndarray):
        n = self.K * self.F
        m = x[:n].reshape(self.K, self.F)
        h = np.abs(x[n:2 * n]).reshape(self.K, self.F)
        sigma = np.exp(x[2 * n:]).reshape(self.K, self.F)
        return m - h, m + h, sigma

    def encode(self, c1, c2, sigma) -> np.ndarray:
        c1, c2 = np.minimum(c1, c2), np.maximum(c1, c2)
        x = np.concatenate([((c1 + c2) / 2).ravel(), ((c2 - c1) / 2).ravel(),
                            np.log(np.clip(sigma, *SIGMA_BOUNDS)).ravel()])
        return np.clip(x, self.lb, self.ub)


class IT2AntecedentObjective:
    """Validation fitness of one antecedent configuration (consequents ridge-solved)."""

    def __init__(self, model: IT2_TSK_ANFIS, X_tr, y_tr, X_val, y_val,
                 ridge_lambda: float, lambda_rule: float = 0.0):
        self.model, self.ridge = model, float(ridge_lambda)
        self.lambda_rule = float(lambda_rule)
        y_tr = np.asarray(y_tr, float).ravel()
        # Same target standardisation that IT2_TSK_ANFIS.fit() uses (train rows only).
        model.y_mean_, model.y_scale_ = float(y_tr.mean()), float(y_tr.std()) or 1.0
        dev = model.device
        self.Zt = model._to_tensor_Z(np.asarray(X_tr, float))
        self.ynt = torch.tensor(model._normalize_y(y_tr), dtype=torch.float32, device=dev).unsqueeze(1)
        self.Zv = model._to_tensor_Z(np.asarray(X_val, float))
        self.yv = np.asarray(y_val, float).ravel()
        self.var_val = float(self.yv.var()) or 1.0
        Z = model._normalize_X(np.asarray(X_tr, float))[:, model._net.ant_idx.cpu().numpy()]
        self.codec = AntecedentCodec(model.n_rules, Z.min(0), Z.max(0))
        self.n_evals = 0

    def val_mse(self, x: np.ndarray) -> float:
        """Validation MSE in original target units for antecedent vector x."""
        self.model.set_antecedents(*self.codec.decode(x))
        self.model._ridge_init_consequents(self.Zt, self.ynt, self.ridge)
        with torch.no_grad():
            yn = self.model._net(self.Zv)[0].cpu().numpy().ravel()
        self.n_evals += 1
        pred = self.model._denormalize_y(yn)
        return float(np.mean((pred - self.yv) ** 2))

    def __call__(self, x: np.ndarray) -> float:
        return self.val_mse(x) / self.var_val + self.lambda_rule * self.model.n_rules


def lhs_initial_swarm(n_rules: int, X_tr, codec: AntecedentCodec, n: int, seed: int,
                      model_kwargs: dict | None = None) -> np.ndarray:
    """Encode the repository's LHS initialisation for n different random states."""
    out = []
    for i in range(n):
        m = IT2_TSK_ANFIS(n_rules, X=X_tr, verbose=False, random_state=seed * 1000 + i,
                          **(model_kwargs or {}))
        out.append(codec.encode(**m.get_antecedents()))
    return np.array(out)


def optimize_antecedents(n_rules: int, X_tr, y_tr, X_val, y_val, *, config: PSOConfig,
                         ridge_lambda: float, x0: np.ndarray | None = None,
                         device: str = "cpu", n_lhs_seeds: int | None = None):
    """
    Run PSO over the antecedents of a K-rule IT2-ANFIS.

    Returns (model, PSOResult, objective). ``model`` holds the best antecedents
    and the matching ridge-solved consequents; it is not yet fine-tuned.
    """
    model = IT2_TSK_ANFIS(n_rules, X=X_tr, verbose=False, random_state=config.seed, device=device)
    obj = IT2AntecedentObjective(model, X_tr, y_tr, X_val, y_val, ridge_lambda)
    n_lhs = config.n_particles // 2 if n_lhs_seeds is None else n_lhs_seeds
    seeds = lhs_initial_swarm(n_rules, X_tr, obj.codec, max(n_lhs, 1), config.seed,
                              {"device": device})
    if x0 is not None:
        seeds = np.vstack([np.atleast_2d(x0), seeds])
    res = ParticleSwarmOptimizer(config).minimize(obj, obj.codec.lb, obj.codec.ub, x0=seeds)
    obj.val_mse(res.best_x)             # leave the model in its best state
    return model, res, obj
