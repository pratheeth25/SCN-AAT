#!/usr/bin/env python3
"""
Command-line pipeline: load data → clean → feature selection (MI) → train IT2-ANFIS → report metrics.

Run from the repository root, e.g.:
  python run_pipeline.py
  python run_pipeline.py --data data/raw/Data-Melbourne_F_fixed.csv --n-rules 7 --epochs 50
  python run_pipeline.py --data path/to/file.csv --no-train   # preprocessing + feature selection only
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import (
    RAW_DIR, DEFAULT_CSV_NAME, DEFAULT_N_RULES, IT2_EPOCHS, IT2_PATIENCE, RANDOM_STATE,
    TARGET_COLUMN, TARGET_UNIT, REPORT_UNIT,
)
from src.data import load_csv, clean_and_prepare
from src.experiments.compare import (
    fit_it2_improved, fit_it2_original, load_frames, score, split_indices,
)
from src.features import select_features_mi
from src.utils.metrics import describe, evaluate_predictions


def _stats(label, v, unit=TARGET_UNIT):
    d = describe(v, label)
    print(f"  {label:<26} n={d['n']:<5} [{unit}] min={d['min']:.1f} q05={d['q05']:.1f} "
          f"median={d['q50']:.1f} q95={d['q95']:.1f} max={d['max']:.1f} "
          f"mean={d['mean']:.1f} std={d['std']:.1f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the IT2-ANFIS pipeline: load CSV → clean → MI feature selection → train IT2-ANFIS → print test R² and MSE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        type=Path,
        default=RAW_DIR / DEFAULT_CSV_NAME,
        help="Path to raw CSV (e.g. data/raw/Data-Melbourne_F_fixed.csv)",
    )
    p.add_argument(
        "--n-rules",
        type=int,
        default=DEFAULT_N_RULES,
        help="Number of IT2-ANFIS rules",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=IT2_EPOCHS,
        help="Max training epochs",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=IT2_PATIENCE,
        help="Early stopping patience",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random state for train/test split",
    )
    p.add_argument(
        "--profile",
        choices=["original", "improved"],
        default="original",
        help="original = paper-style IT2 (LHS init, frozen antecedents); "
             "improved = calendar features + k-means/ridge-initialised IT2 (see README)",
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Only run preprocessing and feature selection; do not train the model",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose training output",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading data...")
    raw = load_csv(data_path)
    print(f"  Raw shape: {raw.shape}")
    tcol = "total_grid" if "total_grid" in raw.columns else TARGET_COLUMN
    print(f"  Target column in CSV: '{tcol}' (CSV has no unit metadata; treated as {TARGET_UNIT})")
    _stats("raw CSV target", raw[tcol].values)
    df_clean = clean_and_prepare(raw)
    print(f"  Clean shape: {df_clean.shape}; target now '{TARGET_COLUMN}' (renamed only, values unchanged)")
    _stats("after cleaning", df_clean[TARGET_COLUMN].values)

    if args.no_train:
        print("Done (--no-train: skipping training).")
        return

    df_orig, df_new = load_frames()
    df = df_orig if args.profile == "original" else df_new
    cols = [c for c in df.columns if c != TARGET_COLUMN]
    print(f"  Profile '{args.profile}': {len(cols)} features: {cols}")
    X = df[cols].values
    y = df[TARGET_COLUMN].values.astype(float)
    _stats("model target (fit unit)", y)

    i_tr, i_va, i_te, i_tv = split_indices(len(y), args.seed)
    for name, idx in (("train", i_tr), ("val", i_va), ("test", i_te)):
        _stats(f"{name} split", y[idx])
    print(f"  Split seed {args.seed}: {len(i_tr)}/{len(i_va)}/{len(i_te)} train/val/test (random, 64/16/20)")

    if args.profile == "original":
        print(f"Training IT2-ANFIS original (n_rules={args.n_rules}, epochs={args.epochs}, patience={args.patience})...")
        model = fit_it2_original(X, y, i_tr, i_va, args.seed, n_rules=args.n_rules,
                                 epochs=args.epochs, patience=args.patience, verbose=not args.quiet)
    else:
        print("Training IT2-ANFIS improved (hyper-parameters chosen by 5-fold CV on train+val)...")
        model, hp = fit_it2_improved(X, cols, y, i_tr, i_va, i_tv, args.seed)
        print(f"  selected: {hp}")

    y_pred = model.predict(X[i_te])           # already in {TARGET_UNIT}; inverse-transformed once inside the model
    _stats("test truth", y[i_te])
    _stats("test prediction", y_pred)
    m = score(y[i_te], y_pred)
    print(f"\n--- Test results ({REPORT_UNIT}, MSE in {REPORT_UNIT}^2; converted explicitly from {TARGET_UNIT}) ---")
    print(f"  R2:   {m['r2']:.4f}")
    print(f"  MSE:  {m['mse']:.2f}")
    print(f"  RMSE: {m['rmse']:.2f}")
    print(f"  MAE:  {m['mae']:.2f}")
    nat = evaluate_predictions(y[i_te], y_pred, unit=TARGET_UNIT)   # independent native-unit evaluation
    print(f"  (native units: MSE = {nat['mse']:,.0f} {TARGET_UNIT}^2, RMSE = {nat['rmse']:,.0f} {TARGET_UNIT}, R2 = {nat['r2']:.4f})")


if __name__ == "__main__":
    main()
