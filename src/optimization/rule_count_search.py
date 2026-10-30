"""Discrete rule-count search wrapped around the antecedent PSO.

Rule count K is discrete, so it is *enumerated* (never rounded from a continuous
particle coordinate):

Stage 1  for every K in the candidate set, run PSO on the antecedents with the
         same small budget and the same LHS-seeded swarm construction; record
         the best validation MSE.
Select   K* = argmin_K  MSE_val(K) / Var(y_val) + lambda_rule * K.
         Because the penalty depends only on K, the selection for any lambda is
         recomputed from the stage-1 table without re-running PSO, which gives
         the lambda sensitivity analysis for free.
Stage 2  ``refine`` continues PSO for K* from the stage-1 optimum with a larger
         budget (see ``compare_models``).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit

from ..models.it2_anfis import IT2_TSK_ANFIS
from .pso_rule_optimizer import PSOConfig, optimize_antecedents
import torch


def select_ridge_lambda(X_tr, y_tr, grid=(0.1, 1.0, 10.0, 100.0), n_rules: int = 7,
                        seed: int = 0, folds: int = 5, device: str = "cpu",
                        temporal: bool = False) -> float:
    """
    Choose the consequent ridge penalty by K-fold CV on the *training rows only*
    (LHS-initialised antecedents). Validation rows are reserved for the PSO
    fitness, test rows are never touched.

    ``temporal=True`` replaces the shuffled folds by expanding-window folds
    (rows must be in time order), so no fold is validated on data older than
    what it was fitted on.
    """
    X_tr, y_tr = np.asarray(X_tr, float), np.asarray(y_tr, float).ravel()
    scores = []
    for lam in grid:
        errs = []
        splitter = TimeSeriesSplit(folds) if temporal else KFold(folds, shuffle=True, random_state=seed)
        for a, b in splitter.split(X_tr):
            m = IT2_TSK_ANFIS(n_rules, X=X_tr[a], verbose=False, random_state=seed, device=device)
            m.y_mean_, m.y_scale_ = float(y_tr[a].mean()), float(y_tr[a].std()) or 1.0
            ynt = torch.tensor(m._normalize_y(y_tr[a]), dtype=torch.float32,
                               device=m.device).unsqueeze(1)
            m._ridge_init_consequents(m._to_tensor_Z(X_tr[a]), ynt, lam)
            errs.append(np.mean((m.predict(X_tr[b]) - y_tr[b]) ** 2))
        scores.append(np.mean(errs))
    return float(grid[int(np.argmin(scores))])


def search_rule_count(X_tr, y_tr, X_val, y_val, k_values, *, config: PSOConfig,
                      ridge_lambda: float, device: str = "cpu", X_report=None, y_report=None):
    """
    Stage-1 enumeration. Returns a list of dicts (one per K) with validation MSE,
    PSO cost and the best antecedent vector. If X_report/y_report are given, the
    stage-1 model's MSE on them is stored as ``report_mse`` -- for plotting only,
    it is never used for selection.
    """
    var_val = float(np.var(y_val)) or 1.0
    rows = []
    for K in k_values:
        cfg = PSOConfig(**{**config.to_dict(), "seed": config.seed * 100 + int(K)})
        model, res, obj = optimize_antecedents(int(K), X_tr, y_tr, X_val, y_val, config=cfg,
                                               ridge_lambda=ridge_lambda, device=device)
        val_mse = obj.val_mse(res.best_x)
        row = {"K": int(K), "val_mse": val_mse, "val_nmse": val_mse / var_val,
               "n_evals": res.n_evals, "n_iter": res.n_iter, "seconds": res.seconds,
               "best_x": res.best_x, "history": res.history}
        if X_report is not None:
            row["report_mse"] = float(np.mean((model.predict(X_report) - np.ravel(y_report)) ** 2))
        rows.append(row)
    return rows


def select_rule_count(rows, lambda_rule: float) -> int:
    """K* = argmin val_nmse + lambda_rule * K (ties -> fewer rules)."""
    J = [(r["val_nmse"] + lambda_rule * r["K"], r["K"]) for r in rows]
    return int(min(J)[1])
