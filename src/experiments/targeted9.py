"""7-rule baseline vs 9-rule IT2-ANFIS with two residual-targeted rules.

Models per (split protocol, seed), all on identical rows / features / seed:
  M7            original 7-rule baseline (``compare_models.fit_baseline7``, unchanged)
  M9            7 baseline rules + 2 rules initialised from training residual patterns;
                consequents trained with the baseline's own Adam procedure
  M9_pso        as M9, but the 2 new rules' antecedents optimised by PSO (val MSE);
                consequents then trained with the baseline's Adam procedure
  M9_pso_ridge  same PSO antecedents; consequents = ridge solution (all terms penalised,
                lambda by CV on training rows), no Adam

Selection uses training (residuals, ridge CV) and validation (PSO fitness) rows only.
Test rows are predicted once per final model. Metrics in MWh (explicit conversion).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    PSO_RIDGE_GRID, REPORT_UNIT, RESULTS_DIR, TARGET_COLUMN, TARGET_UNIT,
)
from ..data import clean_and_prepare, load_csv
from ..optimization.pso_rule_optimizer import PSOConfig
from ..optimization.rule_count_search import select_ridge_lambda
from ..optimization.targeted_rules import (
    build_extended_model, cluster_fallback, find_residual_patterns, init_rules_from_rows,
    optimize_new_rules,
)
from ..utils.metrics import convert_units, evaluate_predictions
from .compare_models import (
    DEVICE, fit_baseline7, paired_stats, select_features_train_only, split_rows,
)

N_NEW = 2
T_MIN = 3.0
BUDGETS = {
    "quick": {"seeds": [0, 1], "pso": {"n_particles": 10, "max_iter": 10, "patience": 5}},
    "full": {"seeds": list(range(10)), "pso": {"n_particles": 20, "max_iter": 40, "patience": 10}},
}
MODELS = ["M7", "M9", "M9_pso", "M9_pso_ridge"]


def _n_params(model, optimised_ant_rules: int, adam: bool) -> tuple[int, int]:
    K, F = model._net.c1.shape
    total = 3 * K * F + K * (F + 1) + 1
    optimised = K * (F + 1) + (1 if adam else 0) + 3 * optimised_ant_rules * F
    return total, optimised


def _metrics(name, model, d, seconds, optimised_ant_rules, adam):
    to = lambda v: convert_units(v, TARGET_UNIT, REPORT_UNIT)
    pred = model.predict(d["Xte"])
    if not np.all(np.isfinite(pred)):
        raise FloatingPointError(f"{name}: non-finite test predictions")
    m = evaluate_predictions(to(d["yte"]), to(pred), REPORT_UNIT)
    val = float(np.mean((model.predict(d["Xva"]) - d["yva"]) ** 2) / 1e6)
    total, opt = _n_params(model, optimised_ant_rules, adam)
    return {"model": name, "rules": int(model.n_rules), "n_params_total": total,
            "n_params_optimised": opt, "runtime_s": seconds, "val_mse": val,
            **{k: m[k] for k in ("mse", "rmse", "mae", "r2")}}


def fit_models(d, dates_tr, seed, pso_budget, m7=None):
    """
    Fit M7, M9, M9_pso and M9_pso_ridge on one split.

    ``d`` holds Xtr/ytr/Xva/yva/Xte/yte; ``dates_tr`` has year/month/day for the training rows.
    Returns (fitted, info, conv): fitted maps name -> (model, seconds, n_antecedent_rules_optimised,
    adam); info/conv describe the rule initialisation and PSO. ``m7`` may be a pre-fitted
    ``fit_baseline7`` model (identical procedure, so results are unchanged).
    """
    fitted = {}
    t0 = time.perf_counter()
    if m7 is None:
        m7 = fit_baseline7(d, seed)
    t_base = time.perf_counter() - t0
    fitted["M7"] = (m7, t_base, 0, True)
    pred7_before = m7.predict(d["Xte"])

    # residual patterns on training rows -> 2 new rules
    t0 = time.perf_counter()
    res_tr = d["ytr"] - m7.predict(d["Xtr"])
    patterns, table = find_residual_patterns(res_tr, dates_tr, N_NEW, T_MIN)
    Ztr = m7._normalize_X(d["Xtr"])
    if len(patterns) == N_NEW:
        source, masks = "residual_pattern", [m for _, m in patterns]
        names = [n for n, _ in patterns]
    else:
        source = "cluster_fallback"
        fb = cluster_fallback(Ztr, res_tr, N_NEW, seed)
        names, masks = [n for n, _ in fb], [m for _, m in fb]
    new_ant = init_rules_from_rows(Ztr, masks, m7.get_antecedents())
    t_init = time.perf_counter() - t0
    info = {"rule_source": source, "new_rules": "; ".join(names),
            "top_groups": "; ".join(f"{r.group} (mean {r.mean_residual / 1e3:+.1f} MWh, t={r.t:+.1f})"
                                     for r in table.head(4).itertuples())}

    # M9: data-initialised rules, baseline consequent procedure
    t0 = time.perf_counter()
    m9 = fit_baseline7(d, seed, model=build_extended_model(m7, d["Xtr"], new_ant, seed, DEVICE))
    fitted["M9"] = (m9, t_base + t_init + time.perf_counter() - t0, 0, True)

    # targeted PSO (shared by M9_pso and M9_pso_ridge)
    t0 = time.perf_counter()
    ridge = select_ridge_lambda(d["Xtr"], d["ytr"], PSO_RIDGE_GRID, 7, seed, device=DEVICE)
    cand = build_extended_model(m7, d["Xtr"], new_ant, seed, DEVICE)
    best, res, _ = optimize_new_rules(cand, N_NEW, d["Xtr"], d["ytr"], d["Xva"], d["yva"],
                                      ridge_lambda=ridge, config=PSOConfig(seed=seed, **pso_budget))
    t_pso = time.perf_counter() - t0
    best_new = (best["c1"][-N_NEW:], best["c2"][-N_NEW:], best["sigma"][-N_NEW:])

    t0 = time.perf_counter()
    m9p = fit_baseline7(d, seed, model=build_extended_model(m7, d["Xtr"], best_new, seed, DEVICE))
    fitted["M9_pso"] = (m9p, t_base + t_init + t_pso + time.perf_counter() - t0, N_NEW, True)

    t0 = time.perf_counter()
    m9r = build_extended_model(m7, d["Xtr"], best_new, seed, DEVICE)
    m9r.fit(d["Xtr"], d["ytr"], d["Xva"], d["yva"], epochs=0, ridge_lambda=ridge, verbose=False)
    fitted["M9_pso_ridge"] = (m9r, t_base + t_init + t_pso + time.perf_counter() - t0, N_NEW, False)

    # the baseline must be untouched by everything above
    if not np.array_equal(pred7_before, m7.predict(d["Xte"])):
        raise RuntimeError("baseline predictions changed during the 9-rule pipeline")
    info.update(ridge_lambda=ridge, pso_evals=res.n_evals, pso_iters=res.n_iter,
                pso_start_fitness=res.history[0]["best_fitness"], pso_best_fitness=res.best_f)
    return fitted, info, list(res.history)


def run_one(df, protocol, seed, pso_budget, split=None):
    tr, va, te = split if split is not None else split_rows(len(df), protocol, seed)
    feats = select_features_train_only(df, tr)
    X, y = df[feats].to_numpy(float), df[TARGET_COLUMN].to_numpy(float)
    d = {"Xtr": X[tr], "ytr": y[tr], "Xva": X[va], "yva": y[va], "Xte": X[te], "yte": y[te]}
    base = {"protocol": protocol, "seed": seed, "n_features": len(feats)}
    fitted, info, conv = fit_models(d, df.iloc[tr], seed, pso_budget)
    rows = [_metrics(name, m, d, sec, n_opt, adam) for name, (m, sec, n_opt, adam) in fitted.items()]
    return [{**base, **r} for r in rows], {**base, **info}, [{**base, **h} for h in conv]


def run_experiment(mode="quick", protocols=("random",), seeds=None, out_root=None) -> Path:
    budget = BUDGETS[mode]
    seeds = list(budget["seeds"] if seeds is None else seeds)
    out = Path(out_root or RESULTS_DIR / "targeted9") / f"{mode}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)          # never overwrite earlier results
    (out / "config.json").write_text(json.dumps(
        {"mode": mode, "protocols": list(protocols), "seeds": seeds, "pso": budget["pso"],
         "n_new_rules": N_NEW, "t_min": T_MIN, "ridge_grid": PSO_RIDGE_GRID, "device": DEVICE,
         "units": {"fit": TARGET_UNIT, "report": REPORT_UNIT}}, indent=2))
    df = clean_and_prepare(load_csv())
    rows, infos, conv = [], [], []
    for protocol in protocols:
        for seed in seeds:
            t = time.perf_counter()
            r, info, c = run_one(df, protocol, seed, budget["pso"])
            rows += r; infos.append(info); conv += c
            print(f"[{protocol} seed {seed}] {time.perf_counter() - t:.0f}s  {info['rule_source']}: "
                  f"{info['new_rules']}  " + "  ".join(f"{x['model']}={x['mse']:.0f}" for x in r),
                  flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "per_seed_results.csv", index=False)
    pd.DataFrame(infos).to_csv(out / "rule_init_and_pso.csv", index=False)
    pd.DataFrame(conv).to_csv(out / "pso_convergence.csv", index=False)
    cols = ["mse", "rmse", "mae", "r2", "val_mse", "runtime_s", "rules", "n_params_total",
            "n_params_optimised"]
    summary = runs.groupby(["protocol", "model"], sort=False)[cols].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    tests = [{"protocol": p, **paired_stats(g, metric, "M7", cand)}
             for p, g in runs.groupby("protocol") for metric in ("mse", "r2")
             for cand in MODELS[1:]]
    pd.DataFrame(tests).to_csv(out / "paired_tests.csv", index=False)
    print(f"\nMSE in {REPORT_UNIT}^2, RMSE/MAE in {REPORT_UNIT}; mean ± sd over seeds")
    print(summary[["mse", "rmse", "mae", "r2", "runtime_s"]].round(3).to_string())
    print(pd.DataFrame(tests)[["protocol", "metric", "candidate", "ref_mean", "cand_mean",
                               "improvement_pct", "ci95_low", "ci95_high", "t_pvalue",
                               "wilcoxon_pvalue"]].round(4).to_string())
    print(f"\nResults written to {out}")
    return out
