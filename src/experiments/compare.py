"""Original vs. improved model comparison over repeated 64/16/20 splits.

Run from the repo root:  python -m src.experiments.compare [--seeds 5]

"original"  = the repo's pre-fix configuration (MI-selected raw features, IT2 with
              LHS init / frozen antecedents, SVR on the raw-kWh target).
"improved"  = calendar features + the fixes described in README (k-means init,
              antecedents on time/season, ridge-initialised consequents, CV-tuned
              hyper-parameters, standardised-target SVR).

Metrics are computed on the held-out test split in original units and reported in
MWh (kWh / 1000) so MSE is comparable with Table 2 of the paper (MWh^2).
Model selection only ever sees train/validation data; the test split is used once.
"""

import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ..config import REPORT_UNIT, RESULTS_DIR, TARGET_COLUMN, TARGET_UNIT, TEST_SIZE, VAL_SIZE
from ..data import add_calendar_features, clean_and_prepare, load_csv
from ..data.preprocessing import CALENDAR_FEATURES
from ..features import select_features_mi
from ..models.anfis_first import FirstOrderANFIS
from ..models.anfis_zero import ZeroOrderANFIS
from ..models.it2_anfis import IT2_TSK_ANFIS
from ..models.svm_model import train_svm
from ..utils.metrics import convert_units, evaluate_predictions

warnings.filterwarnings("ignore")
# Antecedent inputs of the improved IT2-ANFIS (time trend + annual cycle).
ANTECEDENT_COLS = ["t", "doy_sin", "doy_cos"]


def _it2(X, y, ai, R, sigma_scale, ridge, Xv=None, yv=None, epochs=0, seed=0):
    torch.manual_seed(seed)
    m = IT2_TSK_ANFIS(R, X=X, verbose=False, random_state=seed, init="kmeans",
                      sigma_scale=sigma_scale, antecedent_idx=ai)
    m.fit(X, y, Xv, yv, epochs=epochs, patience=15, freeze_antecedents=False,
          lambda_l1=1e-3, lambda_l2=1e-3, eta_cons=2e-3, eta_ant=1e-3, ridge_lambda=ridge)
    return m


def make_cv(seed, temporal=False):
    """Shuffled 5-fold CV (original) or expanding-window CV (rows in time order)."""
    return TimeSeriesSplit(5) if temporal else KFold(5, shuffle=True, random_state=seed)


def tune_it2(X, y, ai, seed, temporal=False):
    """5-fold CV (train+val only) over rules / sigma / ridge, ridge-fit only (cheap)."""
    grid = list(itertools.product([12, 20, 30], [0.3, 0.5, 1.0], [10.0, 30.0, 100.0]))
    kf = make_cv(seed, temporal)
    scores = []
    for R, ss, lam in grid:
        errs = []
        for tr, va in kf.split(X):
            m = _it2(X[tr], y[tr], ai, R, ss, lam, seed=seed)
            errs.append(np.mean((m.predict(X[va]) - y[va]) ** 2))
        scores.append(np.mean(errs))
    return grid[int(np.argmin(scores))]


def tune_simple(make, X, y, grid, seed, temporal=False):
    kf = make_cv(seed, temporal)
    scores = [np.mean([np.mean((make(**g).fit(X[tr], y[tr]).predict(X[va]) - y[va]) ** 2)
                       for tr, va in kf.split(X)]) for g in grid]
    return grid[int(np.argmin(scores))]


def load_frames():
    """(original-feature frame, improved-feature frame); both end with the target column."""
    df = clean_and_prepare(load_csv())
    # MI selection (drops only zero-MI columns) is run on the full table, as in the
    # original pipeline; kept as-is for the "original" arm.
    df_orig = select_features_mi(df)
    df_new = add_calendar_features(df).drop(columns=["year", "month", "day"])
    assert all(c in df_new.columns for c in CALENDAR_FEATURES)
    return df_orig, df_new


def split_indices(n, seed):
    """64/16/20 row split. Same function and seed => same rows for every model/workflow."""
    i_tv, i_te = train_test_split(np.arange(n), test_size=TEST_SIZE, random_state=seed)
    i_tr, i_va = train_test_split(i_tv, test_size=VAL_SIZE, random_state=seed)
    return i_tr, i_va, i_te, i_tv


def fit_it2_original(X, y, i_tr, i_va, seed, n_rules=7, epochs=100, patience=30, verbose=False):
    torch.manual_seed(seed)
    m = IT2_TSK_ANFIS(n_rules, X=X[i_tr], verbose=verbose, random_state=42)
    m.fit(X[i_tr], y[i_tr], X[i_va], y[i_va], epochs=epochs, patience=patience, batch_size=64,
          freeze_antecedents=True, lambda_l1=0.005, lambda_l2=0.01)
    return m


def fit_it2_improved(X, cols, y, i_tr, i_va, i_tv, seed, temporal=False):
    ai = [cols.index(c) for c in ANTECEDENT_COLS]
    R, ss, lam = tune_it2(X[i_tv], y[i_tv], ai, seed, temporal)   # CV on train+val only
    m = _it2(X[i_tr], y[i_tr], ai, R, ss, lam, X[i_va], y[i_va], epochs=60, seed=seed)
    return m, dict(rules=R, sigma_scale=ss, ridge=lam)


def score(y_true_native, y_pred_native):
    """Single scoring path: explicit TARGET_UNIT -> REPORT_UNIT conversion, then sklearn."""
    return evaluate_predictions(convert_units(y_true_native, TARGET_UNIT, REPORT_UNIT),
                                convert_units(y_pred_native, TARGET_UNIT, REPORT_UNIT),
                                unit=REPORT_UNIT)


def run_seed(df_orig, df_new, seed):
    rows = []
    Xo = df_orig.drop(columns=[TARGET_COLUMN]).values
    Xn_df = df_new.drop(columns=[TARGET_COLUMN])
    Xn = Xn_df.values
    y = df_orig[TARGET_COLUMN].values.astype(float)
    assert np.array_equal(y, df_new[TARGET_COLUMN].values.astype(float)), "target rows differ"
    i_tr, i_va, i_te, i_tv = split_indices(len(y), seed)
    yte = y[i_te]

    def rec(name, version, pred):
        rows.append({"seed": seed, "model": name, "version": version, **score(yte, pred)})

    # ---------------- original configuration ----------------
    rec("Dummy(mean)", "baseline", DummyRegressor().fit(Xo[i_tr], y[i_tr]).predict(Xo[i_te]))
    rec("IT2-ANFIS", "original", fit_it2_original(Xo, y, i_tr, i_va, seed).predict(Xo[i_te]))
    rec("ANFIS-0", "original", ZeroOrderANFIS().fit(Xo[i_tr], y[i_tr]).predict(Xo[i_te]))
    rec("ANFIS-1", "original", FirstOrderANFIS().fit(Xo[i_tr], y[i_tr]).predict(Xo[i_te]))
    rec("RF", "original", RandomForestRegressor(300, random_state=42, n_jobs=-1)
        .fit(Xo[i_tr], y[i_tr]).predict(Xo[i_te]))
    rec("SVM", "original", make_pipeline(StandardScaler(), SVR()).fit(Xo[i_tr], y[i_tr])
        .predict(Xo[i_te]))

    # ---------------- improved configuration ----------------
    m, hp = fit_it2_improved(Xn, list(Xn_df.columns), y, i_tr, i_va, i_tv, seed)
    rec("IT2-ANFIS", "improved", m.predict(Xn[i_te]))
    rows[-1].update(hp)

    g0 = tune_simple(lambda n_rules, ridge_alpha: ZeroOrderANFIS(n_rules, 42, ridge_alpha),
                     Xn[i_tv], y[i_tv], [dict(n_rules=r, ridge_alpha=a) for r in (5, 10, 20, 30)
                                         for a in (1e-3, 1e-1)], seed)
    rec("ANFIS-0", "improved", ZeroOrderANFIS(g0["n_rules"], 42, g0["ridge_alpha"])
        .fit(Xn[i_tr], y[i_tr]).predict(Xn[i_te]))
    g1 = tune_simple(lambda n_rules, ridge_alpha: FirstOrderANFIS(n_rules, 42, ridge_alpha),
                     Xn[i_tv], y[i_tv], [dict(n_rules=r, ridge_alpha=a) for r in (3, 5, 7, 10)
                                         for a in (1.0, 30.0, 300.0)], seed)
    rec("ANFIS-1", "improved", FirstOrderANFIS(g1["n_rules"], 42, g1["ridge_alpha"])
        .fit(Xn[i_tr], y[i_tr]).predict(Xn[i_te]))
    rec("RF", "improved", RandomForestRegressor(300, random_state=42, n_jobs=-1)
        .fit(Xn[i_tr], y[i_tr]).predict(Xn[i_te]))
    svm, _ = train_svm(Xn[i_tv], y[i_tv], tune=True)
    rec("SVM", "improved", svm.predict(Xn[i_te]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    df_orig, df_new = load_frames()
    rows = []
    for s in range(args.seeds):
        rows += run_seed(df_orig, df_new, s)
        print(f"seed {s} done", flush=True)
    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "comparison_runs.csv", index=False)   # per-split metrics
    print(f"\nTest metrics, MSE in {REPORT_UNIT}^2 (fit in {TARGET_UNIT}; explicit conversion "
          f"before scoring), mean/std over {args.seeds} splits")
    summ = (res.groupby(["model", "version"])[["mse", "rmse", "mae", "r2"]]
            .agg(["mean", "std"]).round(3))
    summ.to_csv(RESULTS_DIR / "comparison_summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
