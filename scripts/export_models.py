#!/usr/bin/env python3
"""Train and save the 7-rule, 9-rule and 9-rule + PSO IT2-ANFIS models (seed 0).

Uses exactly the runner's code path (``src/experiments/framework.py``), so each saved model's test MSE
equals the seed-0 row of results/experiments/full_random_chrono_10seeds/runs.csv.

  python scripts/export_models.py                       # random + chronological, seed 0 -> models/
  python scripts/export_models.py --protocols random --seed 3

Load a model:
  from src.utils.persistence import load_it2
  model, meta = load_it2("models/random_seed0/M9_pso_ridge.pt")
  model.predict(X)          # X: columns in meta["features"], kWh output
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import targeted9  # noqa: E402
from src.experiments.framework import Frames, RunContext, make_splits  # noqa: E402
from src.experiments.temporal import fit_temporal_family  # noqa: E402
from src.config import ROLLING_FITNESS_FOLDS  # noqa: E402
from src.utils.metrics import convert_units, evaluate_predictions  # noqa: E402
from src.utils.persistence import save_it2  # noqa: E402

MODELS = ("M7", "M9", "M9_pso", "M9_pso_ridge", "M9_tscv")


def export(protocol: str, seed: int, out: Path, data_path=None) -> list[dict]:
    frames = Frames.load(data_path)
    split = make_splits(protocol, len(frames.y), seed)[0]
    budget = targeted9.BUDGETS["full"]["pso"]
    ctx = RunContext(frames, split, {"targeted": targeted9.BUDGETS["full"]}, out, {})
    d, feats, _ = ctx.data("raw")
    fitted, _, _ = targeted9.fit_models(d, frames.dates.iloc[split.tr], seed, budget)
    models = {n: (m, feats, "raw") for n, (m, *_rest) in fitted.items()}
    _, feats_ny, X_ny = ctx.data("no_year")
    tf, _ = fit_temporal_family(X_ny, frames.y, split.tr, split.va, frames.dates, seed, budget,
                                ROLLING_FITNESS_FOLDS)
    models["M9_tscv"] = (tf["M9_tscv"][0], feats_ny, "no_year")

    rows, folder = [], out / f"{protocol}_seed{seed}"
    for name in MODELS:
        model, cols, policy = models[name]
        Xte = frames.raw[cols].to_numpy(float)[split.te]
        mwh = lambda v: convert_units(v, "kWh", "MWh")  # noqa: E731
        m = evaluate_predictions(mwh(frames.y[split.te]), mwh(model.predict(Xte)), "MWh")
        meta = {"model": name, "protocol": protocol, "seed": seed, "feature_policy": policy,
                "features": list(cols), "n_rules": int(model.n_rules), "target_unit": "kWh",
                "test_mse_mwh2": m["mse"], "test_rmse_mwh": m["rmse"], "test_mae_mwh": m["mae"], "test_r2": m["r2"],
                "test_period": [str(frames.dt.iloc[split.te].min().date()), str(frames.dt.iloc[split.te].max().date())]}
        save_it2(model, folder / f"{name}.pt", meta)
        rows.append(meta)
        print(f"{protocol:13s} seed {seed}  {name:13s} rules {model.n_rules}  test MSE {m['mse']:8.1f} MWh²")
    (folder / "manifest.json").write_text(json.dumps(rows, indent=2))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--protocols", nargs="+", default=["random", "chronological"], choices=["random", "chronological"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "models")
    p.add_argument("--data-path", default=None)
    a = p.parse_args()
    for proto in a.protocols:
        export(proto, a.seed, a.out, a.data_path)


if __name__ == "__main__":
    main()
