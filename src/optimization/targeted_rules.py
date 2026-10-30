"""Add two targeted rules to a fitted 7-rule IT2-ANFIS and optimise only those.

1. ``find_residual_patterns``: on *training* rows, group the baseline residuals
   by calendar period (month, year, weekend) and keep groups whose mean residual
   is significant (|t| >= ``t_min``, fixed a priori). The top two become rules.
   Fallback (fewer than two groups): k-means (k=2) on the standardised inputs of
   the training rows with the largest |residual| (top quartile).
2. ``init_rules_from_rows``: each new rule is centred on the mean standardised
   input of its rows; spread = row std per feature, floored at the baseline's
   (LHS) spread; FOU half-width = the baseline's. So c1 <= c2 and sigma > 0.
3. ``TargetedRuleObjective`` + ``optimize_new_rules``: PSO over the new rules'
   centres, FOU half-widths and spreads only; the original rules stay fixed.
   Consequents per particle are the ridge solution (all terms penalised);
   fitness = validation MSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..models.it2_anfis import IT2_TSK_ANFIS
from .pso_rule_optimizer import (
    AntecedentCodec, IT2AntecedentObjective, PSOConfig, ParticleSwarmOptimizer,
)


def find_residual_patterns(residuals, dates: pd.DataFrame, n_rules: int = 2,
                           t_min: float = 3.0, min_rows: int = 20):
    """
    Rank calendar groups of training residuals by |t| of their mean.

    ``dates`` must have year/month/day columns aligned with ``residuals``.
    Returns (patterns, table): patterns = list of (name, boolean row mask) for
    the top ``n_rules`` groups with |t| >= t_min (possibly fewer); table = all
    candidate groups with mean residual, n and t for the log.
    """
    r = np.asarray(residuals, float).ravel()
    dt = pd.to_datetime(dates[["year", "month", "day"]])
    groupings = {"month": dt.dt.month.to_numpy(), "year": dt.dt.year.to_numpy(),
                 "weekend": (dt.dt.dayofweek >= 5).astype(int).to_numpy()}
    rows = []
    for kind, labels in groupings.items():
        for v in np.unique(labels):
            mask = labels == v
            n = int(mask.sum())
            if n < min_rows:
                continue
            g = r[mask]
            t = g.mean() / (g.std(ddof=1) / np.sqrt(n)) if g.std(ddof=1) > 0 else 0.0
            rows.append({"group": f"{kind}={v}", "kind": kind, "value": int(v), "n": n,
                         "mean_residual": float(g.mean()), "t": float(t), "mask": mask})
    table = pd.DataFrame(rows).sort_values("t", key=np.abs, ascending=False)
    chosen = table[table.t.abs() >= t_min].head(n_rules)
    patterns = [(row.group, row.mask) for row in chosen.itertuples()]
    return patterns, table.drop(columns="mask").reset_index(drop=True)


def cluster_fallback(Z, residuals, n_rules: int, seed: int):
    """Training-only fallback: k-means on rows in the top |residual| quartile."""
    r = np.abs(np.asarray(residuals, float).ravel())
    top = r >= np.quantile(r, 0.75)
    labels = KMeans(n_rules, n_init=10, random_state=seed).fit_predict(Z[top])
    out = []
    for k in range(n_rules):
        mask = np.zeros(len(r), bool)
        mask[np.flatnonzero(top)[labels == k]] = True
        out.append((f"cluster{k}", mask))
    return out


def init_rules_from_rows(Z, masks, base_ant: dict):
    """New-rule antecedents from row groups (standardised antecedent space)."""
    half_fou = ((base_ant["c2"] - base_ant["c1"]) / 2).mean(axis=0)   # baseline FOU per feature
    sig_floor = base_ant["sigma"].mean(axis=0)                        # baseline LHS spread
    c1, c2, sig = [], [], []
    for mask in masks:
        rows = Z[mask]
        centre = rows.mean(axis=0)
        sig.append(np.maximum(rows.std(axis=0), sig_floor))
        c1.append(centre - half_fou)
        c2.append(centre + half_fou)
    return np.array(c1), np.array(c2), np.array(sig)


def build_extended_model(base: IT2_TSK_ANFIS, X_tr, new_ant, seed: int, device: str = "cpu"):
    """Fresh (K_base + K_new)-rule model: baseline antecedents + new ones; new consequents."""
    b = base.get_antecedents()
    K = b["c1"].shape[0] + new_ant[0].shape[0]
    m = IT2_TSK_ANFIS(K, X=X_tr, verbose=False, random_state=seed, device=device)
    m.set_antecedents(np.vstack([b["c1"], new_ant[0]]), np.vstack([b["c2"], new_ant[1]]),
                      np.vstack([b["sigma"], new_ant[2]]))
    return m


class TargetedRuleObjective(IT2AntecedentObjective):
    """Validation MSE / Var(y_val) when only the last ``n_new`` rules vary."""

    def __init__(self, model, fixed_ant: dict, n_new: int, X_tr, y_tr, X_val, y_val,
                 ridge_lambda: float):
        super().__init__(model, X_tr, y_tr, X_val, y_val, ridge_lambda)
        self.fixed = fixed_ant
        Z = self.Zt.cpu().numpy()[:, model._net.ant_idx.cpu().numpy()]
        self.codec = AntecedentCodec(n_new, Z.min(0), Z.max(0))   # only the new rules

    def val_mse(self, x):
        c1, c2, s = self.codec.decode(x)
        f = self.fixed
        self.model.set_antecedents(np.vstack([f["c1"], c1]), np.vstack([f["c2"], c2]),
                                   np.vstack([f["sigma"], s]))
        self.model._ridge_init_consequents(self.Zt, self.ynt, self.ridge)
        yn = self.model._net(self.Zv)[0].detach().cpu().numpy().ravel()
        self.n_evals += 1
        return float(np.mean((self.model._denormalize_y(yn) - self.yv) ** 2))


def optimize_new_rules(model9: IT2_TSK_ANFIS, n_new: int, X_tr, y_tr, X_val, y_val, *,
                       ridge_lambda: float, config: PSOConfig, objective_factory=None):
    """
    PSO over the last ``n_new`` rules only. Swarm: particle 0 = the data-driven
    initialisation, half the swarm = Gaussian jitter around it, rest uniform.
    Returns (best antecedents dict for all rules, PSOResult, objective).
    ``objective_factory(model9, fixed, n_new)`` may supply another TargetedRuleObjective
    subclass (e.g. rolling-origin fitness); the default is validation MSE.
    """
    ant = model9.get_antecedents()
    K0 = ant["c1"].shape[0] - n_new
    fixed = {k: v[:K0] for k, v in ant.items()}
    obj = (objective_factory(model9, fixed, n_new) if objective_factory else
           TargetedRuleObjective(model9, fixed, n_new, X_tr, y_tr, X_val, y_val, ridge_lambda))
    x_init = obj.codec.encode(ant["c1"][K0:], ant["c2"][K0:], ant["sigma"][K0:])
    rng = np.random.default_rng(config.seed)
    width = obj.codec.ub - obj.codec.lb
    n_jit = max(config.n_particles // 2, 1)
    jitter = x_init + rng.normal(0, 0.05, (n_jit - 1, x_init.size)) * width
    x0 = np.clip(np.vstack([x_init, jitter]), obj.codec.lb, obj.codec.ub)
    res = ParticleSwarmOptimizer(config).minimize(lambda x: obj.val_mse(x) / obj.var_val,
                                                  obj.codec.lb, obj.codec.ub, x0=x0)
    c1, c2, s = obj.codec.decode(res.best_x)
    best = {"c1": np.vstack([fixed["c1"], c1]), "c2": np.vstack([fixed["c2"], c2]),
            "sigma": np.vstack([fixed["sigma"], s])}
    return best, res, obj
