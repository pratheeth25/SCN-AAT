"""Append-only experiment result files (runs / predictions / errors) and their aggregation.

Layout of one run directory ``results/experiments/<run_id>/``::

    runs.csv               one row per (model, protocol, seed, fold); failed runs have status="failed"
    predictions.csv        held-out predictions of every successful run (MWh)
    errors.csv             one row per failed run (exception type, message, traceback tail)
    summary.csv            built from runs.csv (successful runs only)
    paired_tests.csv       built from runs.csv (matched seeds / folds)
    experiment_config.json dataset, arguments, splits, features, versions

Rows are appended and flushed as soon as a run finishes, so an interrupted experiment keeps
everything completed so far. Nothing is overwritten: every invocation creates a new run directory.
Units: metrics and predictions are in MWh (MSE in MWh^2); models are fitted in kWh and converted
once through ``utils.metrics.convert_units``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_models import paired_stats

RUN_COLUMNS = [
    "run_id", "timestamp", "model_name", "model_version", "protocol", "seed", "fold",
    "train_start", "train_end", "val_start", "val_end", "test_start", "test_end",
    "n_train", "n_val", "n_test", "n_features", "feature_policy", "n_rules",
    "mse", "rmse", "mae", "r2", "unit", "runtime_seconds", "status", "error", "config_json",
]
PRED_COLUMNS = ["run_id", "model_name", "protocol", "seed", "fold", "obs_id", "date",
                "actual", "predicted", "residual", "abs_error", "unit"]
ERROR_COLUMNS = ["run_id", "timestamp", "model_name", "protocol", "seed", "fold",
                 "error_type", "error_message", "traceback", "config_json"]
METRICS = ["mse", "rmse", "mae", "r2"]
BASELINE_MODEL = "M7"
DUMMY_MODEL = "Dummy"


def _append(path: Path, columns: list[str], rows: list[dict]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
        fh.flush()


class ResultStore:
    def __init__(self, run_dir: Path, run_id: str):
        self.dir, self.run_id = Path(run_dir), run_id
        self.dir.mkdir(parents=True, exist_ok=False)          # never reuse / overwrite a run
        self.runs, self.preds, self.errors = (self.dir / f for f in
                                              ("runs.csv", "predictions.csv", "errors.csv"))

    def add_run(self, row: dict) -> None:
        _append(self.runs, RUN_COLUMNS, [{"run_id": self.run_id, **row}])

    def add_predictions(self, df: pd.DataFrame) -> None:
        _append(self.preds, PRED_COLUMNS, df.assign(run_id=self.run_id).to_dict("records"))

    def add_error(self, row: dict) -> None:
        _append(self.errors, ERROR_COLUMNS, [{"run_id": self.run_id, **row}])

    def write_config(self, config: dict) -> None:
        (self.dir / "experiment_config.json").write_text(json.dumps(config, indent=2, default=str))

    def finalize(self) -> None:
        """Rebuild summary.csv and paired_tests.csv from the rows actually saved in runs.csv."""
        if not self.runs.exists():
            return
        runs = pd.read_csv(self.runs)
        build_summary(runs).to_csv(self.dir / "summary.csv", index=False)
        build_paired_tests(runs).to_csv(self.dir / "paired_tests.csv", index=False)


def prediction_frame(*, model_name, protocol, seed, fold, obs_id, dates, actual, predicted) -> pd.DataFrame:
    """Rows for predictions.csv; actual / predicted must already be in MWh."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    res = actual - predicted
    return pd.DataFrame({"model_name": model_name, "protocol": protocol, "seed": seed, "fold": fold,
                         "obs_id": np.asarray(obs_id),
                         "date": pd.DatetimeIndex(pd.to_datetime(np.asarray(dates))).strftime("%Y-%m-%d"),
                         "actual": actual, "predicted": predicted, "residual": res,
                         "abs_error": np.abs(res), "unit": "MWh"})


# ---------------------------------------------------------------------------
# Aggregation (pure functions of the saved rows)
# ---------------------------------------------------------------------------

def build_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Mean / std per (protocol, model) over *successful* runs, run counts, and MSE improvement
    relative to M7 (and to the Dummy mean predictor) in the same protocol, in percent
    (positive = lower MSE than the reference). Failed runs are counted, never averaged;
    a metric with no successful run is NaN, not 0.
    """
    cols = ["protocol", "model_name", "model_version", "n_ok", "n_failed"]
    stats = [f"{m}_{s}" for m in METRICS + ["runtime_seconds", "n_rules"] for s in ("mean", "std")]
    ext = ["mse_vs_M7_pct", "mse_vs_Dummy_pct"]
    if runs.empty:
        return pd.DataFrame(columns=cols + stats + ext)
    ok = runs[runs["status"] == "ok"]
    rows = []
    for (proto, model), g in runs.groupby(["protocol", "model_name"], sort=False):
        o = ok[(ok.protocol == proto) & (ok.model_name == model)]
        row = {"protocol": proto, "model_name": model, "model_version": g["model_version"].iloc[0],
               "n_ok": len(o), "n_failed": int((g["status"] == "failed").sum())}
        for m in METRICS + ["runtime_seconds", "n_rules"]:
            v = o[m].astype(float)
            row[f"{m}_mean"] = v.mean() if v.notna().any() else np.nan
            row[f"{m}_std"] = v.std(ddof=1) if v.notna().sum() > 1 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    for ref, col in ((BASELINE_MODEL, "mse_vs_M7_pct"), (DUMMY_MODEL, "mse_vs_Dummy_pct")):
        base = out[out.model_name == ref].set_index("protocol")["mse_mean"]
        b = out["protocol"].map(base)
        out[col] = 100.0 * (b - out["mse_mean"]) / b
    return out


# (reference, candidate) pairs compared with matched seeds / folds
PAIRS = [("M7", "M9"), ("M7", "M9_pso"), ("M7", "M9_pso_ridge"), ("M7", "PSO_B_rulecount"),
         ("M7", "PSO_C_fixed7"), ("M7", "M7_noyear"), ("M7", "M7_noyear_ridge"), ("M7", "M9_tscv"),
         ("M7", "IT2_improved"), ("M7_noyear", "M7_noyear_ridge"), ("M7_noyear", "M9_tscv"),
         ("M9_pso_ridge", "M9_tscv"), ("M7_noyear_ridge", "M9_tscv"), ("Dummy", "M7"), ("Dummy", "M7_noyear"),
         ("Dummy", "M9_tscv"),
         # 9-rule improvement round: seasonal 9-rule models and train+validation refits
         ("M7", "M9_season_pso"), ("M9_season", "M9_season_pso"), ("M9_tscv", "M9_season_pso"),
         ("M7", "M9_tscv_refit"), ("M9_tscv", "M9_tscv_refit"), ("M7_noyear_ridge_refit", "M9_tscv_refit"),
         ("M7_noyear_ridge", "M7_noyear_ridge_refit"), ("M9_pso_ridge", "M9_tscv_refit"),
         ("Dummy", "M9_tscv_refit")]
PAIR_COLUMNS = ["protocol", "metric", "reference", "candidate", "n_pairs", "pairing", "ref_mean",
                "cand_mean", "mean_diff", "improvement_pct", "ci95_low", "ci95_high", "cohen_dz",
                "t_pvalue", "wilcoxon_pvalue", "t_pvalue_holm"]


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN-safe)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(~np.isnan(p))
    order = idx[np.argsort(p[idx])]
    m, run = len(order), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def build_paired_tests(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Paired t / Wilcoxon / 95% CI on matched units: seed for random and chronological protocols,
    fold (seed-averaged) for rolling-origin. Holm correction over all pairs of one protocol and
    metric. Only successful runs of both models enter; n_pairs is always reported.
    """
    ok = runs[runs["status"] == "ok"]
    rows = []
    for proto, g in ok.groupby("protocol"):
        key = "fold" if proto == "rolling" else "seed"
        for metric in METRICS:
            block = []
            for ref, cand in PAIRS:
                sub = g[g.model_name.isin([ref, cand])]
                if sub.model_name.nunique() < 2:
                    continue
                piv = sub.assign(seed=sub[key]).rename(columns={"model_name": "model"})
                s = paired_stats(piv[["seed", "model", metric]], metric, ref, cand)
                if s and s["n_pairs"] >= 2:
                    block.append({"protocol": proto, "pairing": key, **s})
            if block:
                ps = holm(np.array([b.get("t_pvalue", np.nan) for b in block]))
                for b, hp in zip(block, ps):
                    b["t_pvalue_holm"] = hp
                rows += block
    df = pd.DataFrame(rows)
    for c in PAIR_COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[PAIR_COLUMNS]
