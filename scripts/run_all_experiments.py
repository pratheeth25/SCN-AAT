#!/usr/bin/env python3
"""Central experiment runner: every model x protocol x seed, one evaluation path, CSV outputs.

Examples (PowerShell, from the repository root):
  python scripts/run_all_experiments.py --mode quick
  python scripts/run_all_experiments.py --mode full --protocols random chronological
  python scripts/run_all_experiments.py --mode full --protocols chronological --seeds 10
  python scripts/run_all_experiments.py --mode quick --models M7 M7_noyear M9_tscv --protocols chronological
  python scripts/run_all_experiments.py --list-models

Results go to results/experiments/<run_id>/ (runs.csv, summary.csv, predictions.csv, errors.csv,
paired_tests.csv, experiment_config.json). Existing runs are never overwritten.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")      # CPU: faster for this data size, comparable timings

from src.experiments.framework import (  # noqa: E402
    FAMILIES, MODES, PROTOCOLS, REGISTRY, ExperimentArgs, run_experiments,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, default="quick",
                   help="quick = smoke test (1 seed, small PSO budgets, 2 rolling folds); "
                        "full = 10 seeds, full budgets, 5 rolling folds (unless overridden)")
    p.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=None,
                   help="evaluation protocols (default: random chronological rolling)")
    p.add_argument("--models", nargs="+", default=["all"], metavar="MODEL",
                   help="model names or 'all' (see --list-models)")
    p.add_argument("--seeds", type=int, default=None,
                   help="number of seeds, using seed values 0..N-1 (default: 1 quick / 10 full)")
    p.add_argument("--seed-values", nargs="+", type=int, default=None,
                   help="explicit seed values; overrides --seeds")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="parent directory of the run folder (default: results/experiments)")
    p.add_argument("--run-id", default=None, help="custom run id (default: <mode>_<timestamp>_<hex>)")
    p.add_argument("--rolling-folds", type=int, default=None,
                   help="number of rolling-origin test blocks (default: 2 quick / 5 full)")
    p.add_argument("--rolling-initial", type=float, default=0.5,
                   help="fraction of the series in the first rolling training window")
    p.add_argument("--pso-particles", type=int, default=None, help="override M9/M9_tscv PSO swarm size")
    p.add_argument("--pso-iters", type=int, default=None, help="override M9/M9_tscv PSO max iterations")
    p.add_argument("--fitness-folds", type=int, default=3,
                   help="expanding-window folds in the M9_tscv PSO fitness")
    p.add_argument("--lambda-rule", type=float, default=0.005, help="rule-count penalty for PSO_B_rulecount")
    p.add_argument("--data-path", type=Path, default=None, help="alternative CSV")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first failing model instead of logging it")
    p.add_argument("--list-models", action="store_true", help="print the model registry and exit")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.list_models:
        for fam, names in FAMILIES.items():
            print(f"[{fam}]")
            for n in names:
                print(f"  {n:18s} {REGISTRY[n].version:9s} {REGISTRY[n].description}")
        return None
    quick = a.mode == "quick"
    seeds = a.seed_values or list(range(a.seeds if a.seeds else (1 if quick else 10)))
    return run_experiments(ExperimentArgs(
        mode=a.mode, protocols=tuple(a.protocols or PROTOCOLS), models=tuple(a.models), seeds=tuple(seeds),
        out_dir=a.out_dir, run_id=a.run_id, rolling_folds=a.rolling_folds or (2 if quick else 5),
        rolling_initial=a.rolling_initial, pso_particles=a.pso_particles, pso_iters=a.pso_iters,
        fitness_folds=a.fitness_folds, lambda_rule=a.lambda_rule, fail_fast=a.fail_fast,
        data_path=a.data_path))


if __name__ == "__main__":
    main()
