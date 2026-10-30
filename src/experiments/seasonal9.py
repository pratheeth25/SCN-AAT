"""Seasonal 9-rule IT2-ANFIS with PSO: the improved "9 rules + PSO" model.

Why this design:

* The two residual-targeted rules of M9 have antecedents over all ~15 standardised inputs. With a
  product t-norm a Gaussian rule in 15 dimensions fires strongly only very close to its centre, so
  most rules are almost inactive and the extra rules change little (M9_tscv vs M7_noyear_ridge: -0.6 %).
* Most of the predictive signal is seasonal and nonlinear (Random Forest: month is the top feature;
  calendar columns alone give RF 1307 MWh^2 on the random split, Ridge 1735).
* Raw ``year`` / ``t`` extrapolate on future data; cyclic day-of-year features cannot.

So all 9 rules live in a low-dimensional, bounded seasonal antecedent space (``doy_sin``, ``doy_cos``,
optionally ``weekend``); each rule's first-order consequent is linear in all forecast-safe inputs
(MI-selected ``no_year`` features + the seasonal columns). Rules are k-means initialised on training
rows; their spread and the ridge penalty are chosen by expanding-window CV on training rows; PSO then
refines all 9 rules' centres / FOU widths / spreads with the rolling-origin fitness of
``temporal.RollingRuleObjective`` (mean validation MSE over expanding-window folds of train+val,
consequents ridge-solved per fold). Test rows are never used.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit

from ..data import add_calendar_features
from ..models.it2_anfis import IT2_TSK_ANFIS
from ..optimization.pso_rule_optimizer import PSOConfig
from ..optimization.targeted_rules import optimize_new_rules
from .compare_models import DEVICE
from .temporal import RollingRuleObjective, expanding_folds, select_features

N_RULES = 9
SEASON_COLS = ("doy_sin", "doy_cos")
WEEKEND_COLS = ("doy_sin", "doy_cos", "weekend")
SIGMA_GRID = (0.3, 0.5, 1.0)                 # x k-means default spread (1.0, standardised units)
RIDGE_GRID = (1.0, 10.0, 100.0, 1000.0)


def seasonal_matrix(df_raw: pd.DataFrame, train_rows) -> tuple[np.ndarray, list[str]]:
    """All-row design matrix: MI-selected ``no_year`` features (train rows only) + seasonal columns."""
    cal = add_calendar_features(df_raw)
    feats = select_features(df_raw, train_rows, "no_year")
    cols = feats + [c for c in WEEKEND_COLS if c not in feats]
    return cal[cols].to_numpy(float), cols


def _model(X_tr, ai, sigma_scale, seed):
    return IT2_TSK_ANFIS(N_RULES, X=X_tr, verbose=False, random_state=seed, init="kmeans",
                         sigma_scale=sigma_scale, antecedent_idx=ai, device=DEVICE)


def _ridge_fit(m, X, y, lam):
    m.y_mean_, m.y_scale_ = float(y.mean()), float(y.std()) or 1.0
    yn = torch.tensor(m._normalize_y(y), dtype=torch.float32, device=m.device).unsqueeze(1)
    m._ridge_init_consequents(m._to_tensor_Z(X), yn, lam)
    return m


def select_sigma_ridge(X_tr, y_tr, ai, seed, folds: int = 5):
    """(sigma_scale, ridge lambda) by expanding-window CV on time-ordered training rows."""
    best, best_err = None, np.inf
    for ss, lam in itertools.product(SIGMA_GRID, RIDGE_GRID):
        errs = []
        for a, b in TimeSeriesSplit(folds).split(X_tr):
            m = _ridge_fit(_model(X_tr[a], ai, ss, seed), X_tr[a], y_tr[a], lam)
            errs.append(np.mean((m.predict(X_tr[b]) - y_tr[b]) ** 2))
        if np.mean(errs) < best_err:
            best, best_err = (ss, lam), float(np.mean(errs))
    return best


def fit_seasonal9(X, cols, y, tr, va, seed, *, ant_cols=SEASON_COLS, pso_budget: dict | None = None,
                  fitness_folds: int = 3):
    """
    Fit one seasonal 9-rule model on rows ``tr`` (selection also uses ``va``; rows are sorted).

    Returns (model, seconds, config). ``pso_budget=None`` gives the k-means + ridge model without PSO.
    """
    t0 = time.perf_counter()
    tr, va = np.sort(np.asarray(tr)), np.sort(np.asarray(va))
    pre = np.sort(np.concatenate([tr, va]))
    ai = [cols.index(c) for c in ant_cols]
    ss, lam = select_sigma_ridge(X[tr], y[tr], ai, seed)
    model = _model(X[tr], ai, ss, seed)
    cfg = {"n_rules": N_RULES, "antecedent_inputs": list(ant_cols), "init": "kmeans",
           "sigma_scale": ss, "ridge_lambda": lam, "selection": "expanding-window CV on train rows",
           "consequent_inputs": cols}
    if pso_budget is not None:
        folds = expanding_folds(len(pre), fitness_folds)
        factory = lambda m, fixed, n_new: RollingRuleObjective(           # noqa: E731
            m, fixed, n_new, X[tr], y[tr], X[pre], y[pre], folds, lam)
        best, res, _ = optimize_new_rules(model, N_RULES, X[tr], y[tr], X[va], y[va], ridge_lambda=lam,
                                          config=PSOConfig(seed=seed, **pso_budget),
                                          objective_factory=factory)
        model = _model(X[tr], ai, ss, seed)
        model.set_antecedents(best["c1"], best["c2"], best["sigma"])
        cfg.update(pso_budget=pso_budget, pso_evals=res.n_evals,
                   fitness=f"mean val MSE over {fitness_folds} expanding-window folds of train+val",
                   pso_start_fitness=res.history[0]["best_fitness"], pso_best_fitness=res.best_f)
    model.fit(X[tr], y[tr], X[va], y[va], epochs=0, ridge_lambda=lam, verbose=False)
    return model, time.perf_counter() - t0, cfg
