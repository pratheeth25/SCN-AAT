#!/usr/bin/env python3
"""Pool experiment run folders and print the tables used in docs/results.md.

  python scripts/summarize_runs.py full_random_chrono_10seeds full_rolling_3seeds repair_improved_random ...

Pooling keeps one successful row per (model, protocol, seed, fold) (first run listed wins), so a
repair run replaces earlier failures without double counting. Writes pooled_runs.csv,
pooled_summary.csv and pooled_paired_tests.csv to results/experiments/_pooled/ (overwritten each time;
the source run folders are never modified).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import EXPERIMENTS_DIR  # noqa: E402
from src.experiments.results_store import build_paired_tests, build_summary  # noqa: E402
from src.viz.dashboard_data import dedupe_ok, load_runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--root", type=Path, default=EXPERIMENTS_DIR)
    a = ap.parse_args()
    runs = dedupe_ok(load_runs(a.root, a.runs))
    out = a.root / "_pooled"
    out.mkdir(exist_ok=True)
    summ, tests = build_summary(runs), build_paired_tests(runs)
    runs.to_csv(out / "pooled_runs.csv", index=False)
    summ.to_csv(out / "pooled_summary.csv", index=False)
    tests.to_csv(out / "pooled_paired_tests.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print(f"{len(runs)} pooled rows; failed (unrepaired): {int((runs.status != 'ok').sum())}")
    cols = ["model_name", "n_ok", "n_failed", "mse_mean", "mse_std", "rmse_mean", "mae_mean", "r2_mean",
            "n_rules_mean", "runtime_seconds_mean", "mse_vs_M7_pct", "mse_vs_Dummy_pct"]
    for proto, g in summ.groupby("protocol"):
        print(f"\n== {proto} ==")
        print(g.sort_values("mse_mean")[cols].round(2).to_string(index=False))
    t = tests[tests.metric == "mse"]
    for proto, g in t.groupby("protocol"):
        print(f"\n== paired MSE tests, {proto} (pairing by {g.pairing.iloc[0]}) ==")
        print(g[["reference", "candidate", "n_pairs", "ref_mean", "cand_mean", "improvement_pct", "ci95_low",
                 "ci95_high", "t_pvalue", "wilcoxon_pvalue", "t_pvalue_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
