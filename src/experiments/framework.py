"""Central experiment framework: protocols, model registry and the runner behind
``scripts/run_all_experiments.py``.

Every model is trained on the training rows (validation rows only for early stopping / selection),
predicts the held-out rows once, and is scored by the single canonical path
``utils.metrics.evaluate_predictions`` after one explicit kWh -> MWh conversion.

Protocols
  random         paper protocol, random 64/16/20 split per seed
  chronological  first 64 % train, next 16 % validation, last 20 % test (seed changes initialisation only)
  rolling        expanding-window rolling-origin: ``rolling_folds`` consecutive test blocks over the
                 last 50 % of the series; each fold trains on everything before its block

Model implementations are reused from the existing modules (compare.py, compare_models.py,
targeted9.py, temporal.py); this file only wires them to a common interface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ..config import (
    DEFAULT_CSV_NAME, EXPERIMENTS_DIR, PSO_BUDGETS, PSO_LAMBDA_RULE, RAW_DIR, REPORT_UNIT,
    ROLLING_FITNESS_FOLDS, ROLLING_INITIAL, TARGET_COLUMN, TARGET_UNIT,
)
from ..data import add_calendar_features, clean_and_prepare, load_csv
from ..models.anfis_first import FirstOrderANFIS
from ..models.anfis_zero import ZeroOrderANFIS
from ..models.svm_model import train_svm
from ..utils.metrics import convert_units, evaluate_predictions
from . import compare as cmp
from . import targeted9
from .compare_models import DEVICE, fit_baseline7, run_one as run_pso_search, select_features_train_only, split_rows
from .results_store import ResultStore, prediction_frame
from .seasonal9 import N_RULES, fit_seasonal9, seasonal_matrix
from .temporal import fit_temporal_family, refit_on_pretest, rolling_origin_splits, select_features

warnings.filterwarnings("ignore")

# Worker processes for RF / SVM grid search. n_jobs=-1 exhausted the Windows paging file
# (BrokenProcessPool / MemoryError) when two experiment processes overlapped.
N_JOBS = 2
PROTOCOLS = ("random", "chronological", "rolling")
MODES = ("quick", "full")
to_mwh = lambda v: convert_units(v, TARGET_UNIT, REPORT_UNIT)       # noqa: E731


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    version: str          # baseline | original | improved | targeted | pso | temporal
    fuzzy: bool
    description: str


_SPECS = [
    ("Dummy", "simple", "baseline", False, "training-mean predictor"),
    ("RF_original", "simple", "original", False, "Random Forest, 300 trees, MI-selected raw features"),
    ("SVM_original", "simple", "original", False, "SVR(rbf) on standardised inputs, raw target (as in the original pipeline)"),
    ("ANFIS0_original", "anfis_original", "original", True, "zero-order ANFIS, 7 rules, raw features"),
    ("ANFIS1_original", "anfis_original", "original", True, "first-order ANFIS, 7 rules, raw features"),
    ("IT2_original", "it2_original", "original", True, "IT2-ANFIS 7 rules, original recipe (init seed 42)"),
    ("M7", "it2_original", "original", True, "reference baseline: IT2-ANFIS 7 rules, LHS init, frozen antecedents, Adam+L1/L2"),
    ("ANFIS0_improved", "improved", "improved", True, "zero-order ANFIS, calendar features, tuned"),
    ("ANFIS1_improved", "improved", "improved", True, "first-order ANFIS, calendar features, tuned"),
    ("IT2_improved", "improved", "improved", True, "IT2-ANFIS, calendar features, k-means init, ridge consequents, tuned"),
    ("RF_improved", "improved", "improved", False, "Random Forest on calendar features"),
    ("SVM_improved", "improved", "improved", False, "SVR on scaled target, CV-tuned"),
    ("M9", "targeted9", "targeted", True, "M7 + 2 residual-targeted rules, Adam consequents"),
    ("M9_pso", "targeted9", "targeted", True, "M9 with the 2 new rules' antecedents tuned by PSO (K = 9 fixed)"),
    ("M9_pso_ridge", "targeted9", "targeted", True, "M9_pso with ridge-solved consequents"),
    ("PSO_B_rulecount", "pso_search", "pso", True, "PSO on all antecedents with rule-count search K in 3..15"),
    ("PSO_C_fixed7", "pso_search", "pso", True, "PSO on all antecedents, rule count fixed at 7"),
    ("M7_noyear", "temporal", "temporal", True, "M7 recipe on forecast-safe features (raw year removed)"),
    ("M7_noyear_ridge", "temporal", "temporal", True, "M7_noyear antecedents, ridge consequents, lambda by expanding-window CV"),
    ("M9_tscv", "temporal", "temporal", True, "M9_pso_ridge on no_year features with rolling-origin PSO fitness and expanding-window ridge CV"),
    ("M7_noyear_ridge_refit", "temporal", "temporal", True, "M7_noyear_ridge with consequents re-solved on train+validation after selection (7-rule control)"),
    ("M9_tscv_refit", "temporal", "temporal", True, "M9_tscv with consequents re-solved on train+validation after selection"),
    ("M9_season", "seasonal9", "seasonal", True, "9 rules on day-of-year antecedents (k-means init), linear consequents on no_year + season inputs, ridge; no PSO"),
    ("M9_season_pso", "seasonal9", "seasonal", True, "M9_season with all 9 rules refined by PSO under the rolling-origin fitness"),
]
REGISTRY = {n: ModelSpec(n, f, v, z, d) for n, f, v, z, d in _SPECS}
FAMILIES: dict[str, list[str]] = {}
for _s in REGISTRY.values():
    FAMILIES.setdefault(_s.family, []).append(_s.name)

# (model, protocol) -> reason it cannot run; reported in runs.csv as status="unsupported", never skipped silently
UNSUPPORTED: dict[tuple[str, str], str] = {}


# ---------------------------------------------------------------------------
# Data / splits / context
# ---------------------------------------------------------------------------

@dataclass
class Frames:
    raw: pd.DataFrame            # cleaned table incl. year/month/day, target last
    cal: pd.DataFrame            # calendar-feature table (year/month/day dropped), target last
    y: np.ndarray                # target, kWh
    dates: pd.DataFrame          # year / month / day
    dt: pd.Series                # datetime per row

    @classmethod
    def load(cls, path=None) -> "Frames":
        raw = clean_and_prepare(load_csv(path))
        cal = add_calendar_features(raw).drop(columns=["year", "month", "day"])
        dates = raw[["year", "month", "day"]]
        return cls(raw, cal, raw[TARGET_COLUMN].to_numpy(float), dates, pd.to_datetime(dates))


@dataclass
class Split:
    protocol: str
    seed: int
    fold: int
    tr: np.ndarray
    va: np.ndarray
    te: np.ndarray


def make_splits(protocol: str, n: int, seed: int, rolling_folds: int = 5,
                rolling_initial: float = ROLLING_INITIAL) -> list[Split]:
    if protocol in ("random", "chronological"):
        tr, va, te = split_rows(n, protocol, seed)
        return [Split(protocol, seed, 0, tr, va, te)]
    if protocol == "rolling":
        return [Split("rolling", seed, k, tr, va, te)
                for k, tr, va, te in rolling_origin_splits(n, rolling_folds, rolling_initial)]
    raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")


@dataclass
class Outcome:
    name: str
    pred_mwh: np.ndarray
    seconds: float
    n_rules: float = np.nan
    config: dict = field(default_factory=dict)
    policy: str = "raw"
    n_features: int = 0


class RunContext:
    """One (protocol, seed, fold): split, cached feature matrices and shared fitted baselines."""

    def __init__(self, frames: Frames, split: Split, budgets: dict, artifacts: Path, opts: dict):
        self.f, self.s, self.budgets, self.artifacts, self.opts = frames, split, budgets, artifacts, opts
        self._cache: dict = {}
        self.temporal = split.protocol != "random"

    def data(self, policy: str = "raw"):
        if ("data", policy) not in self._cache:
            tr, va, te = self.s.tr, self.s.va, self.s.te
            feats = (select_features_train_only(self.f.raw, tr) if policy == "raw"
                     else select_features(self.f.raw, tr, policy))
            X = self.f.raw[feats].to_numpy(float)
            d = {"Xtr": X[tr], "ytr": self.f.y[tr], "Xva": X[va], "yva": self.f.y[va],
                 "Xte": X[te], "yte": self.f.y[te]}
            self._cache[("data", policy)] = (d, feats, X)
        return self._cache[("data", policy)]

    def m7(self):
        if "m7" not in self._cache:
            d = self.data("raw")[0]
            t = time.perf_counter()
            self._cache["m7"] = (fit_baseline7(d, self.s.seed), time.perf_counter() - t)
        return self._cache["m7"]

    def outcome(self, name, model_or_pred, seconds, *, n_rules=np.nan, config=None, policy="raw"):
        d, feats, _ = self.data(policy)
        pred = model_or_pred.predict(d["Xte"]) if hasattr(model_or_pred, "predict") else model_or_pred
        return Outcome(name, to_mwh(pred), seconds, n_rules, config or {}, policy, len(feats))


# ---------------------------------------------------------------------------
# Model families (each returns a list[Outcome]; all reuse existing implementations)
# ---------------------------------------------------------------------------

def fam_simple(ctx: RunContext):
    d = ctx.data("raw")[0]
    out = []
    t = time.perf_counter()
    out.append(ctx.outcome("Dummy", DummyRegressor().fit(d["Xtr"], d["ytr"]), time.perf_counter() - t))
    t = time.perf_counter()
    rf = RandomForestRegressor(300, random_state=42, n_jobs=N_JOBS).fit(d["Xtr"], d["ytr"])
    out.append(ctx.outcome("RF_original", rf, time.perf_counter() - t, config={"n_estimators": 300}))
    t = time.perf_counter()
    svm = make_pipeline(StandardScaler(), SVR()).fit(d["Xtr"], d["ytr"])
    out.append(ctx.outcome("SVM_original", svm, time.perf_counter() - t, config={"C": 1.0}))
    return out


def fam_anfis_original(ctx: RunContext):
    d = ctx.data("raw")[0]
    out = []
    for name, cls in (("ANFIS0_original", ZeroOrderANFIS), ("ANFIS1_original", FirstOrderANFIS)):
        t = time.perf_counter()
        m = cls().fit(d["Xtr"], d["ytr"])
        out.append(ctx.outcome(name, m, time.perf_counter() - t, n_rules=m.n_rules))
    return out


def fam_it2_original(ctx: RunContext):
    d, _, X = ctx.data("raw")
    s = ctx.s
    t = time.perf_counter()
    m_orig = cmp.fit_it2_original(X, ctx.f.y, s.tr, s.va, s.seed)
    out = [ctx.outcome("IT2_original", m_orig, time.perf_counter() - t, n_rules=7,
                       config={"init_seed": 42, "epochs": 100, "patience": 30})]
    m7, sec = ctx.m7()
    out.append(ctx.outcome("M7", m7, sec, n_rules=7, config={"init": "lhs", "antecedents": "frozen"}))
    return out


def fam_improved(ctx: RunContext):
    s, y = ctx.s, ctx.f.y
    Xn_df = ctx.f.cal.drop(columns=[TARGET_COLUMN])
    Xn, cols = Xn_df.to_numpy(float), list(Xn_df.columns)
    tv = np.sort(np.concatenate([s.tr, s.va])) if ctx.temporal else np.concatenate([s.tr, s.va])
    T = ctx.temporal
    cv = TimeSeriesSplit(5) if T else 5
    out = []

    def add(name, model, t0, **kw):
        pred = model.predict(Xn[s.te])
        out.append(Outcome(name, to_mwh(pred), time.perf_counter() - t0, kw.pop("n_rules", np.nan),
                           kw, "calendar", len(cols)))

    t0 = time.perf_counter()
    m, hp = cmp.fit_it2_improved(Xn, cols, y, s.tr, s.va, tv, s.seed, temporal=T)
    add("IT2_improved", m, t0, n_rules=hp["rules"], **hp, cv="expanding_window" if T else "shuffled_kfold")

    t0 = time.perf_counter()
    g0 = cmp.tune_simple(lambda n_rules, ridge_alpha: ZeroOrderANFIS(n_rules, 42, ridge_alpha),
                         Xn[tv], y[tv], [dict(n_rules=r, ridge_alpha=a) for r in (5, 10, 20, 30)
                                         for a in (1e-3, 1e-1)], s.seed, temporal=T)
    add("ANFIS0_improved", ZeroOrderANFIS(g0["n_rules"], 42, g0["ridge_alpha"]).fit(Xn[s.tr], y[s.tr]),
        t0, **g0)
    t0 = time.perf_counter()
    g1 = cmp.tune_simple(lambda n_rules, ridge_alpha: FirstOrderANFIS(n_rules, 42, ridge_alpha),
                         Xn[tv], y[tv], [dict(n_rules=r, ridge_alpha=a) for r in (3, 5, 7, 10)
                                         for a in (1.0, 30.0, 300.0)], s.seed, temporal=T)
    add("ANFIS1_improved", FirstOrderANFIS(g1["n_rules"], 42, g1["ridge_alpha"]).fit(Xn[s.tr], y[s.tr]),
        t0, **g1)
    t0 = time.perf_counter()
    add("RF_improved", RandomForestRegressor(300, random_state=42, n_jobs=N_JOBS).fit(Xn[s.tr], y[s.tr]),
        t0, n_estimators=300)
    t0 = time.perf_counter()
    svm, _ = train_svm(Xn[tv], y[tv], tune=True, cv=cv, n_jobs=N_JOBS)
    add("SVM_improved", svm, t0, best_params=str(getattr(svm, "best_params_", {})))
    return out


def fam_targeted9(ctx: RunContext):
    d, _, _ = ctx.data("raw")
    m7, m7_sec = ctx.m7()
    budget = ctx.budgets["targeted"]["pso"]
    fitted, info, _ = targeted9.fit_models(d, ctx.f.dates.iloc[ctx.s.tr], ctx.s.seed, budget, m7=m7)
    out = []
    for name, (model, sec, n_opt, adam) in fitted.items():
        if name == "M7":
            continue
        out.append(ctx.outcome(name, model, sec + m7_sec, n_rules=model.n_rules,
                               config={**info, "pso_budget": budget, "new_rules_optimised": n_opt}))
    return out


def fam_pso_search(ctx: RunContext):
    s = ctx.s
    d, feats, _ = ctx.data("raw")
    out_dir = ctx.artifacts / "pso_search"
    for sub in ("checkpoints", "membership_params"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    tag = s.protocol if s.protocol != "rolling" else f"rolling_f{s.fold}"
    rows, _, _, _, intervals, _ = run_pso_search(
        ctx.f.raw, tag, s.seed, copy.deepcopy(ctx.budgets["pso_search"]), out_dir,
        ctx.opts.get("lambda_rule", PSO_LAMBDA_RULE), models=("B", "C"),
        split=(s.tr, s.va, s.te), feats=feats)
    names = {"B_pso_rules": "PSO_B_rulecount", "C_pso_fixed7": "PSO_C_fixed7"}
    out = []
    for r in rows:
        iv = intervals[intervals.model == r["model"]]
        assert np.array_equal(iv.test_row.to_numpy(), s.te)
        out.append(Outcome(names[r["model"]], iv.y_pred.to_numpy(float), r["train_seconds"], r["K"],
                           {"ridge_lambda": r["ridge_lambda"], "lambda_rule": r["lambda_rule"],
                            "pso_evals": r["pso_evals"], "n_params": r["n_params"]}, "raw", len(feats)))
    return out


def fam_temporal(ctx: RunContext):
    d, feats, X = ctx.data("no_year")
    s = ctx.s
    fitted, _ = fit_temporal_family(
        X, ctx.f.y, s.tr, s.va, ctx.f.dates, s.seed, ctx.budgets["targeted"]["pso"],
        ctx.opts.get("fitness_folds", ROLLING_FITNESS_FOLDS))
    pre = np.concatenate([s.tr, s.va])
    for src in ("M7_noyear_ridge", "M9_tscv"):
        m, sec, K, cfg = fitted[src]
        t = time.perf_counter()
        refit = refit_on_pretest(m, X, ctx.f.y, pre, cfg["ridge_lambda"])
        fitted[f"{src}_refit"] = (refit, sec + time.perf_counter() - t, K,
                                  {**cfg, "consequents_fitted_on": "train+validation (after selection)"})
    return [ctx.outcome(n, m, sec, n_rules=K, config=cfg, policy="no_year")
            for n, (m, sec, K, cfg) in fitted.items()]


def fam_seasonal9(ctx: RunContext):
    s = ctx.s
    X, cols = seasonal_matrix(ctx.f.raw, s.tr)
    out = []
    for name, budget in (("M9_season", None), ("M9_season_pso", ctx.budgets["targeted"]["pso"])):
        m, sec, cfg = fit_seasonal9(X, cols, ctx.f.y, s.tr, s.va, s.seed, pso_budget=budget,
                                    fitness_folds=ctx.opts.get("fitness_folds", ROLLING_FITNESS_FOLDS))
        out.append(Outcome(name, to_mwh(m.predict(X[s.te])), sec, N_RULES, cfg, "no_year+season", len(cols)))
    return out


FAMILY_FUNCS = {"simple": fam_simple, "anfis_original": fam_anfis_original, "it2_original": fam_it2_original,
                "improved": fam_improved, "targeted9": fam_targeted9, "pso_search": fam_pso_search,
                "temporal": fam_temporal, "seasonal9": fam_seasonal9}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class ExperimentArgs:
    mode: str = "quick"
    protocols: tuple = ("random", "chronological", "rolling")
    models: tuple = ("all",)
    seeds: tuple = (0,)
    out_dir: Path | None = None
    run_id: str | None = None
    rolling_folds: int = 2
    rolling_initial: float = ROLLING_INITIAL
    pso_particles: int | None = None
    pso_iters: int | None = None
    fitness_folds: int = ROLLING_FITNESS_FOLDS
    lambda_rule: float = PSO_LAMBDA_RULE
    fail_fast: bool = False
    data_path: Path | None = None


def resolve_models(requested) -> list[str]:
    names = list(REGISTRY) if (not requested or "all" in requested) else list(requested)
    bad = [n for n in names if n not in REGISTRY]
    if bad:
        raise ValueError(f"unknown model(s) {bad}; available: {list(REGISTRY)}")
    return names


def make_budgets(args: ExperimentArgs) -> dict:
    t = copy.deepcopy(targeted9.BUDGETS[args.mode])
    if args.pso_particles:
        t["pso"]["n_particles"] = args.pso_particles
    if args.pso_iters:
        t["pso"]["max_iter"] = args.pso_iters
    return {"targeted": t, "pso_search": copy.deepcopy(PSO_BUDGETS[args.mode])}


def _versions() -> dict:
    import numpy, pandas, scipy, sklearn
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__}


def _dataset_info(frames: Frames, path) -> dict:
    p = Path(path) if path else RAW_DIR / DEFAULT_CSV_NAME
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "n_rows": len(frames.y),
            "first_date": str(frames.dt.min().date()), "last_date": str(frames.dt.max().date()),
            "target": TARGET_COLUMN, "fit_unit": TARGET_UNIT, "report_unit": REPORT_UNIT,
            "columns": list(frames.raw.columns)}


def _split_info(frames: Frames, s: Split) -> dict:
    dt = frames.dt
    rng = lambda idx: (str(dt.iloc[idx].min().date()), str(dt.iloc[idx].max().date()))   # noqa: E731
    return {"protocol": s.protocol, "seed": s.seed, "fold": s.fold, "n_train": len(s.tr),
            "n_val": len(s.va), "n_test": len(s.te), "train_range": rng(s.tr), "val_range": rng(s.va),
            "test_range": rng(s.te), "contiguous": s.protocol != "random"}


def _run_row(frames, s, spec, o: Outcome | None, status, error=""):
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "model_name": spec.name,
           "model_version": spec.version, "protocol": s.protocol, "seed": s.seed, "fold": s.fold,
           "n_train": len(s.tr), "n_val": len(s.va), "n_test": len(s.te), "unit": REPORT_UNIT,
           "status": status, "error": error}
    dt = frames.dt
    for tag, idx in (("train", s.tr), ("val", s.va), ("test", s.te)):
        row[f"{tag}_start"], row[f"{tag}_end"] = dt.iloc[idx].min().date(), dt.iloc[idx].max().date()
    if o is not None:
        m = evaluate_predictions(to_mwh(frames.y[s.te]), o.pred_mwh, REPORT_UNIT)
        row.update({k: m[k] for k in ("mse", "rmse", "mae", "r2")})
        row.update(n_rules=o.n_rules, runtime_seconds=o.seconds, n_features=o.n_features,
                   feature_policy=o.policy, config_json=json.dumps(o.config, default=str))
    return row


def run_experiments(args: ExperimentArgs) -> Path:
    if args.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    for p in args.protocols:
        if p not in PROTOCOLS:
            raise ValueError(f"unknown protocol {p!r}; choose from {PROTOCOLS}")
    names = resolve_models(args.models)
    run_id = args.run_id or f"{args.mode}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    base = Path(args.out_dir) if args.out_dir else EXPERIMENTS_DIR
    store = ResultStore(base / run_id, run_id)
    budgets = make_budgets(args)
    frames = Frames.load(args.data_path)
    config = {"run_id": run_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": args.mode,
              "protocols": list(args.protocols), "models": names, "seeds": list(args.seeds),
              "rolling": {"folds": args.rolling_folds, "initial": args.rolling_initial},
              "budgets": budgets, "fitness_folds": args.fitness_folds, "lambda_rule": args.lambda_rule,
              "args": {k: str(v) for k, v in vars(args).items()},
              "dataset": _dataset_info(frames, args.data_path), "versions": _versions(),
              "device": DEVICE, "n_jobs": N_JOBS, "model_registry": {n: vars(REGISTRY[n]) for n in names},
              "feature_policies": {"raw": "MI>0 on training rows, all columns incl. year",
                                   "no_year": "MI>0 on training rows, raw `year` removed"},
              "splits": [], "features": {},
              "unsupported": {f"{m}|{p}": r for (m, p), r in UNSUPPORTED.items()}}
    store.write_config(config)
    index_dir = Path(args.out_dir) if args.out_dir else EXPERIMENTS_DIR
    index = index_dir / "index.csv"
    index.parent.mkdir(parents=True, exist_ok=True)

    def log_index(status):
        new = not index.exists()
        with open(index, "a", encoding="utf-8", newline="") as fh:
            if new:
                fh.write("run_id,time,mode,protocols,n_seeds,n_models,status,path\n")
            fh.write(f"{run_id},{time.strftime('%Y-%m-%dT%H:%M:%S')},{args.mode},{'+'.join(args.protocols)},"
                     f"{len(args.seeds)},{len(names)},{status},{store.dir}\n")

    log_index("started")
    status = "finished"
    opts = {"lambda_rule": args.lambda_rule, "fitness_folds": args.fitness_folds}
    wanted = {f: [n for n in ms if n in names] for f, ms in FAMILIES.items()}
    try:
        for protocol in args.protocols:
            for seed in args.seeds:
                for split in make_splits(protocol, len(frames.y), seed, args.rolling_folds, args.rolling_initial):
                    t_ctx = time.perf_counter()
                    ctx = RunContext(frames, split, budgets, store.dir / "artifacts", opts)
                    config["splits"].append(_split_info(frames, split))
                    n_ok = n_bad = 0
                    for fam, fnames in wanted.items():
                        todo = []
                        for n in fnames:
                            if (n, protocol) in UNSUPPORTED:
                                store.add_run(_run_row(frames, split, REGISTRY[n], None, "unsupported",
                                                       UNSUPPORTED[(n, protocol)]))
                            else:
                                todo.append(n)
                        if not todo:
                            continue
                        try:
                            outcomes = {o.name: o for o in FAMILY_FUNCS[fam](ctx)}
                            missing = [n for n in todo if n not in outcomes]
                            if missing:
                                raise RuntimeError(f"family {fam} did not return {missing}")
                            err = None
                        except Exception as exc:               # a failed family must not erase other runs
                            if args.fail_fast:
                                raise
                            outcomes, err = {}, exc
                        for n in todo:
                            spec = REGISTRY[n]
                            if err is not None:
                                tb = "".join(traceback.format_exception(err))[-1500:]
                                store.add_run(_run_row(frames, split, spec, None, "failed", f"{type(err).__name__}: {err}"))
                                store.add_error({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "model_name": n,
                                                 "protocol": protocol, "seed": seed, "fold": split.fold,
                                                 "error_type": type(err).__name__, "error_message": str(err),
                                                 "traceback": tb,
                                                 "config_json": json.dumps({"family": fam, "mode": args.mode})})
                                n_bad += 1
                                continue
                            o = outcomes[n]
                            store.add_run(_run_row(frames, split, spec, o, "ok"))
                            store.add_predictions(prediction_frame(
                                model_name=n, protocol=protocol, seed=seed, fold=split.fold, obs_id=split.te,
                                dates=frames.dt.iloc[split.te], actual=to_mwh(frames.y[split.te]),
                                predicted=o.pred_mwh))
                            config["features"][f"{protocol}|{seed}|{split.fold}|{o.policy}"] = (
                                ctx.data(o.policy)[1] if o.policy in ("raw", "no_year") else "see config_json")
                            n_ok += 1
                    print(f"[{protocol} seed {seed} fold {split.fold}] {time.perf_counter() - t_ctx:.0f}s  "
                          f"ok={n_ok} failed={n_bad}", flush=True)
                    store.finalize()                       # keep summary current after every context
    except KeyboardInterrupt:
        status = "interrupted"
        print("interrupted: completed runs were kept", flush=True)
    except Exception:
        status = "aborted"
        raise
    finally:
        config["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        config["status"] = status
        store.write_config(config)
        store.finalize()
        log_index(status)
    print(f"\nResults written to {store.dir}")
    return store.dir
