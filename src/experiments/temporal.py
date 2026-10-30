"""Chronological-generalisation tools: forecast-safe features, rolling-origin splits and fitness.

Why (see docs/results.md §5): the raw ``year`` column extrapolates outside the training range
in a past -> future split, a single validation block is a shifted, noisy PSO target, and shuffled CV
lets later rows predict earlier ones. This module provides

* ``select_features``  - train-only MI selection under a feature policy (``raw`` | ``no_year``);
* ``rolling_origin_splits`` / ``expanding_folds`` - time-ordered splits (no row is validated or tested
  on data older than what the model saw);
* ``RollingRuleObjective`` - PSO fitness = mean validation MSE over expanding-window folds of the
  pre-test rows, with ridge consequents re-solved per fold;
* ``fit_temporal_family`` - M7_noyear, M7_noyear_ridge, M9_tscv.

Nothing here reads test rows: the test block is only predicted by the returned fitted models.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_regression

from ..config import (
    FEATURE_POLICIES, MI_RANDOM_STATE, ROLLING_FITNESS_FOLDS, ROLLING_INITIAL, ROLLING_VAL_FRAC,
    TARGET_COLUMN, TEMPORAL_RIDGE_GRID,
)
from ..models.it2_anfis import IT2_TSK_ANFIS
from ..optimization.pso_rule_optimizer import PSOConfig
from ..optimization.rule_count_search import select_ridge_lambda
from ..optimization.targeted_rules import (
    TargetedRuleObjective, build_extended_model, cluster_fallback, find_residual_patterns,
    init_rules_from_rows, optimize_new_rules,
)
from .compare_models import DEVICE, fit_baseline7
from .targeted9 import N_NEW, T_MIN

_DROP = {"raw": (), "no_year": ("year",)}


def select_features(df: pd.DataFrame, rows, policy: str = "raw") -> list[str]:
    """MI > 0 selection on ``rows`` only, after removing the columns the policy forbids."""
    if policy not in FEATURE_POLICIES:
        raise ValueError(f"unknown feature policy {policy!r}; choose from {FEATURE_POLICIES}")
    cols = [c for c in df.columns if c != TARGET_COLUMN and c not in _DROP[policy]]
    mi = mutual_info_regression(df[cols].iloc[rows], df[TARGET_COLUMN].iloc[rows],
                                random_state=MI_RANDOM_STATE)
    return [c for c, v in zip(cols, mi) if v > 0]


def rolling_origin_splits(n: int, n_folds: int = 5, initial: float = ROLLING_INITIAL,
                          val_frac: float = ROLLING_VAL_FRAC):
    """
    Expanding-window rolling-origin splits over a time-ordered series of length ``n``.

    Fold k tests on the k-th equal block of the last ``1 - initial`` of the series, trains on
    everything before the block, and uses the last ``val_frac`` of that training window as
    validation. Yields (fold, train_idx, val_idx, test_idx); all index arrays are ascending and
    strictly ordered train < val < test.
    """
    edges = np.linspace(initial, 1.0, n_folds + 1)
    for k in range(n_folds):
        start, end = int(round(n * edges[k])), int(round(n * edges[k + 1]))
        n_val = max(int(round(start * val_frac)), 1)
        yield k, np.arange(0, start - n_val), np.arange(start - n_val, start), np.arange(start, end)


def expanding_folds(n_pre: int, n_folds: int = ROLLING_FITNESS_FOLDS, first: float = 0.5):
    """(train_end, val_end) slice bounds of expanding-window folds over ``n_pre`` ordered rows."""
    edges = [int(round(n_pre * e)) for e in np.linspace(first, 1.0, n_folds + 1)]
    return [(edges[k], edges[k + 1]) for k in range(n_folds)]


class RollingRuleObjective(TargetedRuleObjective):
    """
    Fitness for the two new rules = mean over expanding-window folds of the pre-test rows
    (train + validation, in time order) of the fold's validation MSE. For every fold the
    consequents are the ridge solution on that fold's training window only.
    """

    def __init__(self, model, fixed_ant, n_new, X_tr, y_tr, X_pre, y_pre, folds, ridge_lambda):
        # X_val/y_val of the parent = all pre-test rows (used for Zv / yv / var_val).
        super().__init__(model, fixed_ant, n_new, X_tr, y_tr, X_pre, y_pre, ridge_lambda)
        self.folds = list(folds)
        self.yn_pre = torch.tensor(model._normalize_y(np.asarray(y_pre, float)),
                                   dtype=torch.float32, device=model.device).unsqueeze(1)

    def val_mse(self, x):
        c1, c2, s = self.codec.decode(x)
        f = self.fixed
        self.model.set_antecedents(np.vstack([f["c1"], c1]), np.vstack([f["c2"], c2]),
                                   np.vstack([f["sigma"], s]))
        errs = []
        for a, b in self.folds:
            self.model._ridge_init_consequents(self.Zv[:a], self.yn_pre[:a], self.ridge)
            yn = self.model._net(self.Zv[a:b])[0].detach().cpu().numpy().ravel()
            errs.append(np.mean((self.model._denormalize_y(yn) - self.yv[a:b]) ** 2))
        self.n_evals += 1
        return float(np.mean(errs))


def refit_on_pretest(model: IT2_TSK_ANFIS, X, y, rows, ridge_lambda: float) -> IT2_TSK_ANFIS:
    """
    Copy of ``model`` whose consequents are re-solved by ridge on ``rows`` (train + validation, i.e. all
    pre-test rows) after every selection step is finished. Antecedents, input scaling, target scaling
    and lambda are unchanged; test rows are not involved.
    """
    m = copy.deepcopy(model)
    rows = np.sort(np.asarray(rows))
    yn = torch.tensor(m._normalize_y(np.asarray(y, float)[rows]), dtype=torch.float32,
                      device=m.device).unsqueeze(1)
    m._ridge_init_consequents(m._to_tensor_Z(np.asarray(X, float)[rows]), yn, ridge_lambda)
    return m


def fit_temporal_family(X, y, tr, va, dates: pd.DataFrame, seed: int, pso_budget: dict,
                        fitness_folds: int = ROLLING_FITNESS_FOLDS, m7_noyear=None):
    """
    Fit the three chronological-generalisation models on ``no_year`` features ``X``.

    ``tr``/``va`` are row indices into X / y / dates. They are sorted here so that every CV and
    rolling fold is in time order whatever the protocol. Returns (fitted, info) where fitted maps
    name -> (model, seconds, n_rules, config).
    """
    tr, va = np.sort(np.asarray(tr)), np.sort(np.asarray(va))
    pre = np.sort(np.concatenate([tr, va]))
    d = {"Xtr": X[tr], "ytr": y[tr], "Xva": X[va], "yva": y[va]}
    fitted = {}

    t0 = time.perf_counter()
    m7n = m7_noyear if m7_noyear is not None else fit_baseline7(d, seed)
    t_base = time.perf_counter() - t0
    fitted["M7_noyear"] = (m7n, t_base, 7, {"features": "no_year", "consequents": "adam_l1_l2"})

    t0 = time.perf_counter()
    lam = select_ridge_lambda(d["Xtr"], d["ytr"], TEMPORAL_RIDGE_GRID, 7, seed, device=DEVICE,
                              temporal=True)
    t_lam = time.perf_counter() - t0
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    mr = IT2_TSK_ANFIS(7, X=d["Xtr"], verbose=False, random_state=seed, device=DEVICE)
    mr.fit(d["Xtr"], d["ytr"], d["Xva"], d["yva"], epochs=0, ridge_lambda=lam, verbose=False)
    fitted["M7_noyear_ridge"] = (mr, t_lam + time.perf_counter() - t0, 7,
                                 {"features": "no_year", "consequents": "ridge", "ridge_lambda": lam,
                                  "ridge_cv": "expanding_window"})

    # M9_tscv: residual-targeted rules on the no_year baseline, rolling-origin PSO fitness
    t0 = time.perf_counter()
    res_tr = d["ytr"] - m7n.predict(d["Xtr"])
    patterns, table = find_residual_patterns(res_tr, dates.iloc[tr], N_NEW, T_MIN)
    Ztr = m7n._normalize_X(d["Xtr"])
    if len(patterns) == N_NEW:
        source, masks = "residual_pattern", [m for _, m in patterns]
    else:
        source, masks = "cluster_fallback", [m for _, m in cluster_fallback(Ztr, res_tr, N_NEW, seed)]
    new_ant = init_rules_from_rows(Ztr, masks, m7n.get_antecedents())
    cand = build_extended_model(m7n, d["Xtr"], new_ant, seed, DEVICE)
    folds = expanding_folds(len(pre), fitness_folds)
    factory = lambda model, fixed, n_new: RollingRuleObjective(          # noqa: E731
        model, fixed, n_new, d["Xtr"], d["ytr"], X[pre], y[pre], folds, lam)
    best, res, _ = optimize_new_rules(cand, N_NEW, d["Xtr"], d["ytr"], d["Xva"], d["yva"],
                                      ridge_lambda=lam, config=PSOConfig(seed=seed, **pso_budget),
                                      objective_factory=factory)
    best_new = (best["c1"][-N_NEW:], best["c2"][-N_NEW:], best["sigma"][-N_NEW:])
    m9 = build_extended_model(m7n, d["Xtr"], best_new, seed, DEVICE)
    m9.fit(d["Xtr"], d["ytr"], d["Xva"], d["yva"], epochs=0, ridge_lambda=lam, verbose=False)
    fitted["M9_tscv"] = (m9, t_base + t_lam + time.perf_counter() - t0, 9,
                         {"features": "no_year", "consequents": "ridge", "ridge_lambda": lam,
                          "rule_source": source, "pso_evals": res.n_evals,
                          "fitness": f"mean val MSE over {fitness_folds} expanding-window folds",
                          "pso_start_fitness": res.history[0]["best_fitness"],
                          "pso_best_fitness": res.best_f})
    return fitted, {"rule_source": source, "ridge_lambda": lam}
