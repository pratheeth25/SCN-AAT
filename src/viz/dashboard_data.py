"""Loading and aggregation helpers behind ``streamlit_app.py`` (pure pandas, no Streamlit import).

Everything shown in the dashboard is read from the CSV / JSON files written by
``scripts/run_all_experiments.py``; nothing is hard-coded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EXPERIMENTS_DIR
from ..experiments.results_store import RUN_COLUMNS, build_paired_tests, build_summary  # noqa: F401

METRIC_LABELS = {"mse": "MSE (MWh²)", "rmse": "RMSE (MWh)", "mae": "MAE (MWh)", "r2": "R²"}
HIGHER_IS_BETTER = {"mse": False, "rmse": False, "mae": False, "r2": True}


def default_dir() -> Path:
    return Path(os.environ.get("XUQ_EXPERIMENTS_DIR", EXPERIMENTS_DIR))


def discover_runs(root: Path | str | None = None) -> list[str]:
    """Run ids (newest first) that have a runs.csv."""
    root = Path(root) if root else default_dir()
    if not root.exists():
        return []
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "runs.csv").exists()]
    return [d.name for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)]


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def load_table(root, run_ids, name: str) -> pd.DataFrame:
    """Concatenate ``<run>/<name>.csv`` over run ids; a ``run_id`` column is always present."""
    root = Path(root)
    parts = []
    for rid in run_ids:
        df = _read(root / rid / f"{name}.csv")
        if not df.empty:
            df["run_id"] = rid
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_runs(root, run_ids) -> pd.DataFrame:
    runs = load_table(root, run_ids, "runs")
    if runs.empty:
        return pd.DataFrame(columns=RUN_COLUMNS)
    for c in ("mse", "rmse", "mae", "r2", "n_rules", "runtime_seconds", "seed", "fold"):
        if c in runs:
            runs[c] = pd.to_numeric(runs[c], errors="coerce")
    for c in RUN_COLUMNS:
        if c not in runs:
            runs[c] = np.nan
    return runs


def load_config(root, run_id) -> dict:
    p = Path(root) / run_id / "experiment_config.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except json.JSONDecodeError:
        return {}


def parse_config_json(s) -> dict:
    try:
        return json.loads(s) if isinstance(s, str) and s else {}
    except json.JSONDecodeError:
        return {}


def filter_runs(runs: pd.DataFrame, protocols=None, models=None, seeds=None) -> pd.DataFrame:
    out = runs
    if protocols:
        out = out[out.protocol.isin(protocols)]
    if models:
        out = out[out.model_name.isin(models)]
    if seeds:
        out = out[out.seed.isin(seeds)]
    return out


def dedupe_ok(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Pool several run folders: keep one successful row per (model, protocol, seed, fold), the first
    in the order given, and drop failed rows whose key has a success elsewhere. Failures that were
    never repaired are kept so they stay visible.
    """
    key = ["model_name", "protocol", "seed", "fold"]
    ok = runs[runs.status == "ok"].drop_duplicates(key, keep="first")
    bad = runs[runs.status != "ok"]
    bad = bad.merge(ok[key].assign(_fixed=True), on=key, how="left")
    bad = bad[bad["_fixed"].isna()].drop(columns="_fixed")
    return pd.concat([ok, bad], ignore_index=True)


def best_per_protocol(summary: pd.DataFrame, metric: str = "mse") -> pd.DataFrame:
    """Best model per protocol by mean metric (lowest MSE/RMSE/MAE, highest R²) over successful runs."""
    s = summary[summary.n_ok > 0].dropna(subset=[f"{metric}_mean"])
    if s.empty:
        return s
    grp = s.groupby("protocol")[f"{metric}_mean"]
    pick = grp.idxmax() if HIGHER_IS_BETTER[metric] else grp.idxmin()
    return s.loc[pick].reset_index(drop=True)


def degradation(summary: pd.DataFrame, metric: str = "mse", src: str = "random", dst: str = "chronological"):
    """Per model: metric under ``src`` vs ``dst`` protocol, absolute and percentage change."""
    col = f"{metric}_mean"
    p = summary.pivot_table(index="model_name", columns="protocol", values=col)
    if src not in p or dst not in p:
        return pd.DataFrame()
    out = pd.DataFrame({f"{src}": p[src], f"{dst}": p[dst]})
    out["abs_change"] = out[dst] - out[src]
    out["pct_change"] = 100.0 * out["abs_change"] / out[src].abs()
    sign = -1 if HIGHER_IS_BETTER[metric] else 1          # positive = worse
    out["worse_under_dst"] = sign * out["abs_change"] > 0
    if "M7" in p.index and not HIGHER_IS_BETTER[metric]:
        for proto in (src, dst):
            out[f"vs_M7_{proto}_pct"] = 100.0 * (p.loc["M7", proto] - p[proto]) / abs(p.loc["M7", proto])
        if src == "random" and dst == "chronological":
            out["pattern"] = np.where((out["vs_M7_random_pct"] > 0) & (out["vs_M7_chronological_pct"] < 0),
                                      "better than M7 on random, worse on chronological", "")
    return out.reset_index()


def membership_files(root, run_id) -> list[Path]:
    return sorted((Path(root) / run_id / "artifacts" / "pso_search" / "membership_params").glob("*.json"))


def membership_curves(params: dict, rule: int, feature_idx: int, n: int = 200):
    """UMF / LMF of an uncertain-mean Gaussian MF (standardised input units)."""
    c1, c2, s = (np.asarray(params[k])[rule, feature_idx] for k in ("c1", "c2", "sigma"))
    x = np.linspace(min(c1, c2) - 4 * s, max(c1, c2) + 4 * s, n)
    g = lambda c: np.exp(-0.5 * ((x - c) / s) ** 2)               # noqa: E731
    umf = np.where(x < c1, g(c1), np.where(x > c2, g(c2), 1.0))
    lmf = np.where(x <= (c1 + c2) / 2, g(c2), g(c1))
    return x, lmf, umf
