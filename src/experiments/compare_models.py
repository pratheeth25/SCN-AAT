"""Baseline vs PSO-optimised IT2-ANFIS: matched-seed experiment.

Models (same split, same features, same seed for each run):
  A  baseline   : 7 rules, LHS init, frozen antecedents, Adam + L1/L2 (original code path)
  B  pso_rules  : K chosen from RULE_COUNT_CANDIDATES by stage-1 PSO + complexity penalty,
                  antecedents refined by stage-2 PSO, consequents ridge-solved then
                  fine-tuned with the existing L1/L2 Adam procedure
  C  pso_fixed7 : identical to B but K = 7 (ablation: parameter optimisation only)

Protocol per (split protocol, seed):
  * 64/16/20 train/val/test split; "random" (paper protocol) or "chronological".
  * MI feature selection fitted on the training rows only, applied to all models.
  * All selection (ridge lambda: CV on train; K, antecedents, early stopping: val)
    uses no test data. Test rows are predicted once per final model.
  * Point metrics in MWh (explicit kWh -> MWh conversion), interval metrics on the
    raw IT2 bounds and on split-conformal bounds calibrated on validation rows.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

from ..config import (
    DEFAULT_N_RULES, INTERVAL_ALPHA, IT2_EPOCHS, IT2_LAMBDA_L1, IT2_LAMBDA_L2, IT2_PATIENCE,
    MI_RANDOM_STATE, PSO_BUDGETS, PSO_LAMBDA_RULE, PSO_LAMBDA_SENSITIVITY, PSO_RIDGE_GRID,
    REPORT_UNIT, RESULTS_DIR, TARGET_COLUMN, TARGET_UNIT, TEST_SIZE, VAL_SIZE,
)
from ..data import clean_and_prepare, load_csv
from ..models.it2_anfis import IT2_TSK_ANFIS
from ..optimization.pso_rule_optimizer import PSOConfig, optimize_antecedents
from ..optimization.rule_count_search import (
    search_rule_count, select_ridge_lambda, select_rule_count,
)
from ..utils.metrics import conformal_margin, convert_units, evaluate_predictions, interval_metrics

DEVICE = "cpu"   # small data: CPU is faster than CUDA here and makes timings comparable


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def split_rows(n: int, protocol: str, seed: int):
    """64/16/20 split. 'random' = paper protocol; 'chronological' = past -> future."""
    idx = np.arange(n)
    if protocol == "random":
        tv, te = train_test_split(idx, test_size=TEST_SIZE, random_state=seed)
        tr, va = train_test_split(tv, test_size=VAL_SIZE, random_state=seed)
    elif protocol == "chronological":
        n_te = int(round(n * TEST_SIZE))
        n_va = int(round((n - n_te) * VAL_SIZE))
        tr, va, te = idx[: n - n_te - n_va], idx[n - n_te - n_va: n - n_te], idx[n - n_te:]
    else:
        raise ValueError(f"unknown split protocol {protocol!r}")
    return tr, va, te


def select_features_train_only(df: pd.DataFrame, rows) -> list[str]:
    """MI > 0 feature selection fitted on training rows only (no leakage)."""
    X = df.drop(columns=[TARGET_COLUMN]).iloc[rows]
    mi = mutual_info_regression(X, df[TARGET_COLUMN].iloc[rows], random_state=MI_RANDOM_STATE)
    return [c for c, v in zip(X.columns, mi) if v > 0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def n_trainable_params(model: IT2_TSK_ANFIS, antecedents_trainable: bool) -> int:
    net = model._net
    n = net.consequents.numel() + net.logit_q.numel()
    if antecedents_trainable:
        n += net.c1.numel() + net.c2.numel() + net.log_sigma.numel()
    return int(n)


def raw_bounds(model: IT2_TSK_ANFIS, X):
    """Type-reduced lower/upper outputs (original units) and their crossing rate."""
    with torch.no_grad():
        _, yl, yu = model._net(model._to_tensor_Z(np.asarray(X, float)))
    yl = model._denormalize_y(yl.cpu().numpy().ravel())
    yu = model._denormalize_y(yu.cpu().numpy().ravel())
    return np.minimum(yl, yu), np.maximum(yl, yu), float(np.mean(yl > yu))


def latency_ms_per_sample(model, X, repeats: int = 20) -> float:
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - t)
    return float(np.median(times) / len(X) * 1e3)


def finetune(model: IT2_TSK_ANFIS, X_tr, y_tr, X_va, y_va, ridge, epochs, seed):
    """Existing consequent training: ridge solution as epoch 0, then Adam + L1/L2, val early stop."""
    torch.manual_seed(seed)
    model.fit(X_tr, y_tr, X_va, y_va, epochs=epochs, patience=IT2_PATIENCE, batch_size=64,
              freeze_antecedents=True, lambda_l1=IT2_LAMBDA_L1, lambda_l2=IT2_LAMBDA_L2,
              ridge_lambda=ridge, verbose=False)
    return model


def evaluate_model(name, model, K, ant_trainable, train_s, d, alpha, y_range):
    """All point / interval / efficiency metrics for one fitted model."""
    Xte, yte, Xva, yva = d["Xte"], d["yte"], d["Xva"], d["yva"]
    to = lambda v: convert_units(v, TARGET_UNIT, REPORT_UNIT)
    pred = model.predict(Xte)
    row = {"model": name, "K": K, "n_params": n_trainable_params(model, ant_trainable),
           "n_antecedent_params": 3 * K * model._net.c1.shape[1],
           "train_seconds": train_s, "latency_ms_per_sample": latency_ms_per_sample(model, Xte)}
    row.update({k: v for k, v in evaluate_predictions(to(yte), to(pred), REPORT_UNIT).items()
                if k in ("mse", "rmse", "mae", "r2", "pred_std")})
    lo, hi, cross = raw_bounds(model, Xte)
    row["bound_crossing_rate"] = cross
    raw = interval_metrics(to(yte), to(lo), to(hi), alpha, y_range)
    row.update({f"raw_{k}": v for k, v in raw.items()})
    # split-conformal calibration on validation rows
    lo_v, hi_v, _ = raw_bounds(model, Xva)
    q = conformal_margin(yva, lo_v, hi_v, alpha)
    lo_c, hi_c = lo - q, hi + q
    mid = (lo_c + hi_c) / 2
    lo_c, hi_c = np.minimum(lo_c, mid), np.maximum(hi_c, mid)
    row["conformal_margin_mwh"] = float(to(q))
    row.update({f"conf_{k}": v for k, v in
                interval_metrics(to(yte), to(lo_c), to(hi_c), alpha, y_range).items()})
    intervals = pd.DataFrame({"model": name, "test_row": d["te_idx"], "y_true": to(yte),
                              "y_pred": to(pred), "raw_lower": to(lo), "raw_upper": to(hi),
                              "conf_lower": to(lo_c), "conf_upper": to(hi_c)})
    return row, intervals


def fit_baseline7(d: dict, seed: int, n_rules: int = DEFAULT_N_RULES,
                  model: IT2_TSK_ANFIS | None = None) -> IT2_TSK_ANFIS:
    """Original model A: LHS init, frozen antecedents, Adam + L1/L2, validation early stopping.

    Pass ``model`` (antecedents already set) to train its consequents with exactly
    the same procedure.
    """
    torch.manual_seed(seed)
    m = model or IT2_TSK_ANFIS(n_rules, X=d["Xtr"], verbose=False, random_state=seed, device=DEVICE)
    m.fit(d["Xtr"], d["ytr"], d["Xva"], d["yva"], epochs=IT2_EPOCHS, patience=IT2_PATIENCE,
          batch_size=64, freeze_antecedents=True, lambda_l1=IT2_LAMBDA_L1,
          lambda_l2=IT2_LAMBDA_L2, verbose=False)
    return m


def save_checkpoint(path: Path, model: IT2_TSK_ANFIS, meta: dict):
    torch.save({"state_dict": model._net.state_dict(), "x_mean": model.x_mean_,
                "x_scale": model.x_scale_, "y_mean": model.y_mean_, "y_scale": model.y_scale_,
                **meta}, path)


# ---------------------------------------------------------------------------
# One matched run
# ---------------------------------------------------------------------------

def run_one(df, protocol, seed, budget, out: Path, lambda_rule=PSO_LAMBDA_RULE,
            alpha=INTERVAL_ALPHA, models=("A", "B", "C"), split=None, feats=None):
    """``split=(tr, va, te)`` / ``feats`` override the built-in protocol split / MI selection."""
    tr, va, te = split if split is not None else split_rows(len(df), protocol, seed)
    feats = feats or select_features_train_only(df, tr)
    X = df[feats].to_numpy(float)
    y = df[TARGET_COLUMN].to_numpy(float)
    d = {"Xtr": X[tr], "ytr": y[tr], "Xva": X[va], "yva": y[va], "Xte": X[te], "yte": y[te],
         "te_idx": te}
    y_range = float(convert_units(y[tr].max() - y[tr].min(), TARGET_UNIT, REPORT_UNIT))
    tag = f"{protocol}_seed{seed}"
    rows, intervals, conv = [], [], []
    base = {"protocol": protocol, "seed": seed, "n_features": len(feats)}

    # ---- A: baseline (original procedure, K = 7) ----
    if "A" in models:
        t = time.perf_counter()
        mA = fit_baseline7(d, seed)
        r, iv = evaluate_model("A_baseline7", mA, DEFAULT_N_RULES, False, time.perf_counter() - t,
                               d, alpha, y_range)
        rows.append({**base, **r}); intervals.append(iv)
        save_checkpoint(out / "checkpoints" / f"{tag}_A.pt", mA, {"K": DEFAULT_N_RULES})

    s1_rows, sens = [], []
    if not ({"B", "C"} & set(models)):
        for iv in intervals:
            iv.insert(0, "seed", seed); iv.insert(0, "protocol", protocol)
        return rows, s1_rows, sens, conv, pd.concat(intervals), feats

    # ---- shared: ridge lambda (train-row CV) + stage-1 rule-count enumeration ----
    t_shared = time.perf_counter()
    ridge = select_ridge_lambda(d["Xtr"], d["ytr"], PSO_RIDGE_GRID, DEFAULT_N_RULES, seed,
                                device=DEVICE)
    s1 = PSOConfig(seed=seed, **budget["stage1"])
    k_values = sorted(set(budget["k_values"]) | {DEFAULT_N_RULES})
    stage1 = search_rule_count(d["Xtr"], d["ytr"], d["Xva"], d["yva"], k_values, config=s1,
                               ridge_lambda=ridge, device=DEVICE, X_report=d["Xte"],
                               y_report=d["yte"])
    shared_s = time.perf_counter() - t_shared
    by_k = {r["K"]: r for r in stage1}
    s1_rows = [{**base, "K": r["K"], "val_mse_mwh2": r["val_mse"] / 1e6, "val_nmse": r["val_nmse"],
                "test_mse_mwh2_report_only": r["report_mse"] / 1e6, "n_evals": r["n_evals"],
                "seconds": r["seconds"], "ridge_lambda": ridge} for r in stage1]
    for r in stage1:
        conv += [{**base, "model": f"stage1_K{r['K']}", "stage": 1, **h} for h in r["history"]]
    sens = [{**base, "lambda_rule": lam, "selected_K": select_rule_count(stage1, lam)}
            for lam in PSO_LAMBDA_SENSITIVITY]

    # ---- B (selected K) and C (K = 7): stage-2 PSO from the stage-1 optimum + fine-tune ----
    K_star = select_rule_count(stage1, lambda_rule)
    for name, K in (("B_pso_rules", K_star), ("C_pso_fixed7", DEFAULT_N_RULES)):
        if name[0] not in models:
            continue
        t = time.perf_counter()
        s2 = PSOConfig(seed=seed + 7919, **budget["stage2"])
        m, res, obj = optimize_antecedents(K, d["Xtr"], d["ytr"], d["Xva"], d["yva"], config=s2,
                                           ridge_lambda=ridge, x0=by_k[K]["best_x"], device=DEVICE)
        val_after_pso = obj.val_mse(res.best_x) / 1e6      # before fine-tuning changes consequents
        finetune(m, d["Xtr"], d["ytr"], d["Xva"], d["yva"], ridge, budget["finetune_epochs"], seed)
        own_s = time.perf_counter() - t
        # B pays for the whole enumeration; C only for its own stage-1 K = 7 run.
        train_s = own_s + (shared_s if name.startswith("B") else by_k[K]["seconds"])
        r, iv = evaluate_model(name, m, K, True, train_s, d, alpha, y_range)
        r.update({"ridge_lambda": ridge, "lambda_rule": lambda_rule,
                  "pso_evals": res.n_evals + (sum(x["n_evals"] for x in stage1)
                                              if name.startswith("B") else by_k[K]["n_evals"]),
                  "val_mse_after_pso_mwh2": val_after_pso})
        rows.append({**base, **r}); intervals.append(iv)
        conv += [{**base, "model": name, "stage": 2, **h} for h in res.history]
        save_checkpoint(out / "checkpoints" / f"{tag}_{name[0]}.pt", m, {"K": K, "ridge": ridge})
        ant = m.get_antecedents()
        (out / "membership_params" / f"{tag}_{name[0]}.json").write_text(json.dumps(
            {"K": K, "features": feats, "space": "standardised inputs (train mean/std)",
             **{k: v.tolist() for k, v in ant.items()}}))

    for iv in intervals:
        iv.insert(0, "seed", seed); iv.insert(0, "protocol", protocol)
    return rows, s1_rows, sens, conv, pd.concat(intervals), feats


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_stats(df_runs: pd.DataFrame, metric: str, a: str, b: str) -> dict:
    """Paired comparison on matched (protocol, seed) runs: b relative to a (a = reference)."""
    p = df_runs.pivot_table(index="seed", columns="model", values=metric)
    if a not in p or b not in p:
        return {}
    x, y = p[a].to_numpy(), p[b].to_numpy()
    diff = y - x
    n = len(diff)
    out = {"metric": metric, "reference": a, "candidate": b, "n_pairs": n,
           "ref_mean": float(x.mean()), "cand_mean": float(y.mean()),
           "mean_diff": float(diff.mean()),
           "improvement_pct": float(100 * (x.mean() - y.mean()) / x.mean()) if metric in
           ("mse", "rmse", "mae") else float("nan")}
    if n >= 2 and diff.std(ddof=1) > 0:
        sd = diff.std(ddof=1)
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        out.update(ci95_low=float(diff.mean() - h), ci95_high=float(diff.mean() + h),
                   cohen_dz=float(diff.mean() / sd),
                   t_pvalue=float(stats.ttest_rel(y, x).pvalue))
        try:
            out["wilcoxon_pvalue"] = float(stats.wilcoxon(y, x).pvalue)
        except ValueError:
            out["wilcoxon_pvalue"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_experiment(mode: str = "quick", protocols=("random",), seeds=None, out_root=None,
                   lambda_rule: float = PSO_LAMBDA_RULE, models=("A", "B", "C")) -> Path:
    if mode not in PSO_BUDGETS:
        raise ValueError(f"mode must be one of {list(PSO_BUDGETS)}")
    budget = copy.deepcopy(PSO_BUDGETS[mode])
    seeds = list(budget["seeds"] if seeds is None else seeds)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(out_root or RESULTS_DIR / "pso") / f"{mode}_{stamp}"
    if out.exists():
        raise FileExistsError(out)       # never overwrite earlier results
    for sub in ("checkpoints", "membership_params", "figures"):
        (out / sub).mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({
        "mode": mode, "models": list(models), "protocols": list(protocols), "seeds": seeds, "budget": budget,
        "lambda_rule": lambda_rule, "lambda_sensitivity": PSO_LAMBDA_SENSITIVITY,
        "ridge_grid": PSO_RIDGE_GRID, "interval_alpha": INTERVAL_ALPHA,
        "baseline": {"K": DEFAULT_N_RULES, "epochs": IT2_EPOCHS, "patience": IT2_PATIENCE,
                     "lambda_l1": IT2_LAMBDA_L1, "lambda_l2": IT2_LAMBDA_L2},
        "split": {"test": TEST_SIZE, "val_of_trainval": VAL_SIZE}, "device": DEVICE,
        "units": {"fit": TARGET_UNIT, "report": REPORT_UNIT}}, indent=2))

    df = clean_and_prepare(load_csv())
    all_rows, all_s1, all_sens, all_conv, all_iv, feats_log = [], [], [], [], [], {}
    for protocol in protocols:
        for seed in seeds:
            t = time.perf_counter()
            rows, s1, sens, conv, iv, feats = run_one(df, protocol, seed, budget, out, lambda_rule,
                                                     models=models)
            all_rows += rows; all_s1 += s1; all_sens += sens; all_conv += conv; all_iv.append(iv)
            feats_log[f"{protocol}_seed{seed}"] = feats
            ks = {r["model"]: r["K"] for r in rows}
            print(f"[{protocol} seed {seed}] {time.perf_counter() - t:.0f}s  K*={ks.get('B_pso_rules', '-')}  "
                  + "  ".join(f"{r['model']}: MSE={r['mse']:.1f}" for r in rows), flush=True)
            # write incrementally so a crash keeps finished runs
            pd.DataFrame(all_rows).to_csv(out / "per_seed_results.csv", index=False)

    runs = pd.DataFrame(all_rows)
    pd.DataFrame(all_s1).to_csv(out / "rule_count_search.csv", index=False)
    pd.DataFrame(all_sens).to_csv(out / "lambda_sensitivity.csv", index=False)
    pd.DataFrame(all_conv).to_csv(out / "pso_convergence.csv", index=False)
    pd.concat(all_iv).to_csv(out / "prediction_intervals.csv", index=False)
    (out / "selected_features.json").write_text(json.dumps(feats_log, indent=1))

    metrics = ["mse", "rmse", "mae", "r2", "K", "n_params", "train_seconds",
               "latency_ms_per_sample", "raw_picp", "raw_mpiw", "raw_interval_score",
               "conf_picp", "conf_mpiw", "conf_nmpiw", "conf_interval_score",
               "bound_crossing_rate"]
    summary = runs.groupby(["protocol", "model"])[metrics].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    tests = []
    for protocol, g in runs.groupby("protocol"):
        for metric in ("mse", "rmse", "mae", "r2", "conf_interval_score", "conf_picp"):
            for a, b in (("A_baseline7", "B_pso_rules"), ("A_baseline7", "C_pso_fixed7"),
                         ("C_pso_fixed7", "B_pso_rules")):
                s = paired_stats(g, metric, a, b)
                if s:
                    tests.append({"protocol": protocol, **s})
    pd.DataFrame(tests).to_csv(out / "paired_tests.csv", index=False)
    print(f"\nResults written to {out}")
    return out
