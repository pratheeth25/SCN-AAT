"""Temporal split ordering, leakage safeguards, metric / unit correctness, CSV schemas, aggregation
and failure isolation of the central experiment framework."""
import json

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src.experiments.framework import (
    REGISTRY, ExperimentArgs, Frames, make_splits, resolve_models, run_experiments,
)
from src.experiments.results_store import (
    ERROR_COLUMNS, PRED_COLUMNS, RUN_COLUMNS, ResultStore, build_paired_tests, build_summary, holm,
    prediction_frame,
)
from src.experiments.temporal import (
    RollingRuleObjective, expanding_folds, rolling_origin_splits, select_features,
)
from src.utils.metrics import convert_units, evaluate_predictions
from src.viz.dashboard_data import dedupe_ok


# ------------------------------- split ordering / leakage -------------------------------

def test_chronological_split_is_ordered_and_disjoint():
    (s,) = make_splits("chronological", 1382, seed=3)
    assert s.tr.max() < s.va.min() and s.va.max() < s.te.min()
    assert len(np.intersect1d(s.tr, s.va)) == len(np.intersect1d(s.va, s.te)) == 0
    assert len(s.tr) + len(s.va) + len(s.te) == 1382
    assert s.te[-1] == 1381 and abs(len(s.te) / 1382 - 0.2) < 0.01


def test_chronological_split_does_not_depend_on_seed():
    a, b = make_splits("chronological", 500, 0)[0], make_splits("chronological", 500, 9)[0]
    assert np.array_equal(a.te, b.te) and np.array_equal(a.tr, b.tr)


def test_random_split_is_a_partition_and_seed_dependent():
    a, b = make_splits("random", 500, 0)[0], make_splits("random", 500, 1)[0]
    for s in (a, b):
        assert len(np.unique(np.concatenate([s.tr, s.va, s.te]))) == 500
    assert not np.array_equal(a.te, b.te)


def test_rolling_origin_never_trains_on_the_future():
    n = 1382
    splits = list(rolling_origin_splits(n, 5))
    assert len(splits) == 5
    prev_end = None
    for k, tr, va, te in splits:
        assert tr.max() < va.min() and va.max() < te.min()
        assert va.max() + 1 == te.min() and tr.max() + 1 == va.min()    # contiguous, nothing skipped
        if prev_end is not None:
            assert te.min() == prev_end                                  # test blocks tile the tail
        prev_end = te.max() + 1
    assert prev_end == n and splits[0][3].min() >= int(0.49 * n)         # first test block starts at ~50 %
    assert len(splits[-1][1]) > len(splits[0][1])                         # training window expands


def test_expanding_folds_are_ordered_and_cover_the_end():
    folds = expanding_folds(1000, 3)
    assert folds[-1][1] == 1000 and folds[0][0] == 500
    for (a, b), (a2, _) in zip(folds, folds[1:]):
        assert a < b == a2


# ------------------------------- preprocessing / feature leakage -------------------------------

def _toy_frame(n=300):
    rng = np.random.default_rng(0)
    dt = pd.date_range("2014-01-01", periods=n)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "year": dt.year,
                       "month": dt.month, "day": dt.day})
    df[TARGET_COLUMN] = 3 * df.x1 + 0.01 * np.arange(n)
    return df


def test_no_year_policy_drops_year_and_raw_keeps_it():
    df = _toy_frame()
    rows = np.arange(200)
    assert "year" in select_features(df, rows, "raw")
    assert "year" not in select_features(df, rows, "no_year")
    with pytest.raises(ValueError):
        select_features(df, rows, "bogus")


def test_feature_selection_uses_only_given_rows():
    """Changing the target outside the training rows must not change the selection."""
    df = _toy_frame()
    rows = np.arange(150)
    a = select_features(df, rows, "no_year")
    df2 = df.copy()
    df2.loc[150:, TARGET_COLUMN] = np.random.default_rng(5).normal(size=len(df) - 150) * 1e6
    assert select_features(df2, rows, "no_year") == a


def test_rolling_objective_fold_error_ignores_later_rows():
    """Fold k fits ridge on rows [:a] and scores rows [a:b]; later rows must not influence fold k."""
    from src.models.it2_anfis import IT2_TSK_ANFIS
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 3)); y = 2e5 + 3e4 * X[:, 0] + rng.normal(0, 3e3, 120)
    m = IT2_TSK_ANFIS(3, X=X[:60], verbose=False, random_state=0, device="cpu")
    fixed = {k: v[:2] for k, v in m.get_antecedents().items()}
    folds = [(60, 90), (90, 120)]
    obj = RollingRuleObjective(m, fixed, 1, X[:60], y[:60], X, y, folds, 10.0)
    x = obj.codec.encode(*(v[2:] for v in m.get_antecedents().values()))
    base = obj.val_mse(x)
    y2 = y.copy(); y2[90:] += 1e6                       # corrupt only rows after fold-0's validation block
    obj2 = RollingRuleObjective(m, fixed, 1, X[:60], y[:60], X, y2, folds, 10.0)
    errs = []
    for o in (obj, obj2):
        a, b = folds[0]
        o.model._ridge_init_consequents(o.Zv[:a], o.yn_pre[:a], o.ridge)
        yn = o.model._net(o.Zv[a:b])[0].detach().cpu().numpy().ravel()
        errs.append(np.mean((o.model._denormalize_y(yn) - o.yv[a:b]) ** 2))
    assert errs[0] == pytest.approx(errs[1])
    assert np.isfinite(base)


# ------------------------------- metrics / units -------------------------------

def test_metrics_and_unit_conversion_pipeline():
    y_kwh = np.array([200_000., 250_000., 300_000.]); p_kwh = np.array([210_000., 240_000., 290_000.])
    y, p = convert_units(y_kwh, "kWh", "MWh"), convert_units(p_kwh, "kWh", "MWh")
    m = evaluate_predictions(y, p, "MWh")
    assert m["mse"] == pytest.approx(100.0) and m["rmse"] == pytest.approx(10.0)
    assert m["mae"] == pytest.approx(10.0)
    assert m["r2"] == pytest.approx(1 - 300 / 5000)             # SSres = 3*100, SStot = 2*50^2
    assert evaluate_predictions(y_kwh, p_kwh, "kWh")["mse"] == pytest.approx(m["mse"] * 1e6)


def test_prediction_frame_columns_and_residual_sign():
    df = prediction_frame(model_name="M", protocol="random", seed=0, fold=0, obs_id=np.array([3, 4]),
                          dates=pd.Series(pd.to_datetime(["2015-01-02", "2015-01-03"])),
                          actual=[10.0, 20.0], predicted=[12.0, 15.0])
    df["run_id"] = "r"
    assert set(PRED_COLUMNS) <= set(df.columns)
    assert df.residual.tolist() == [-2.0, 5.0] and df.abs_error.tolist() == [2.0, 5.0]
    assert df.date.tolist() == ["2015-01-02", "2015-01-03"] and set(df.unit) == {"MWh"}


# ------------------------------- CSV schemas / aggregation / failure isolation -------------------------------

def _run_rows():
    rows = []
    for proto in ("random", "chronological"):
        for seed in range(4):
            base = {"protocol": proto, "seed": seed, "fold": 0, "model_version": "x", "status": "ok",
                    "n_rules": 7.0, "runtime_seconds": 1.0, "rmse": 1.0, "mae": 1.0, "r2": 0.1}
            rows.append({**base, "model_name": "M7", "mse": 100.0 + seed})
            rows.append({**base, "model_name": "Dummy", "mse": 200.0 + seed})
            rows.append({**base, "model_name": "M9_tscv", "mse": 90.0 + 2 * seed})
    rows.append({"protocol": "random", "seed": 9, "fold": 0, "model_name": "M9_tscv", "model_version": "x",
                 "status": "failed", "mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan,
                 "n_rules": np.nan, "runtime_seconds": np.nan})
    return pd.DataFrame(rows)


def test_summary_counts_failures_and_never_averages_them():
    s = build_summary(_run_rows()).set_index(["protocol", "model_name"])
    r = s.loc[("random", "M9_tscv")]
    assert r.n_ok == 4 and r.n_failed == 1
    assert r.mse_mean == pytest.approx(np.mean([90, 92, 94, 96]))
    assert s.loc[("random", "M7"), "mse_vs_M7_pct"] == pytest.approx(0.0)
    assert s.loc[("random", "M9_tscv"), "mse_vs_M7_pct"] == pytest.approx(100 * (101.5 - 93) / 101.5)
    assert s.loc[("random", "Dummy"), "mse_vs_Dummy_pct"] == pytest.approx(0.0)


def test_summary_all_failed_is_nan_not_zero():
    runs = _run_rows()
    runs.loc[runs.model_name == "M9_tscv", "status"] = "failed"
    runs.loc[runs.model_name == "M9_tscv", "mse"] = np.nan
    r = build_summary(runs).set_index(["protocol", "model_name"]).loc[("random", "M9_tscv")]
    assert r.n_ok == 0 and np.isnan(r.mse_mean) and np.isnan(r.mse_vs_M7_pct)


def test_paired_tests_use_matched_seeds_and_holm():
    t = build_paired_tests(_run_rows())
    row = t[(t.protocol == "random") & (t.metric == "mse") & (t.reference == "M7")
            & (t.candidate == "M9_tscv")].iloc[0]
    assert row.n_pairs == 4 and row.pairing == "seed"        # failed seed 9 has no partner
    assert row.mean_diff == pytest.approx(np.mean([-10, -9, -8, -7]))
    assert row.t_pvalue_holm >= row.t_pvalue
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def test_dedupe_ok_repairs_failures_without_double_counting():
    key = {"model_name": "M7", "protocol": "random", "seed": 0, "fold": 0}
    runs = pd.DataFrame([{**key, "status": "failed", "mse": np.nan},
                         {**key, "status": "ok", "mse": 1.0},
                         {**key, "status": "ok", "mse": 2.0},                       # duplicate success: first kept
                         {**key, "seed": 1, "status": "failed", "mse": np.nan}])    # never repaired: stays visible
    out = dedupe_ok(runs)
    assert sorted(out.status) == ["failed", "ok"]
    assert out[out.status == "ok"].mse.iloc[0] == 1.0 and out[out.status == "failed"].seed.iloc[0] == 1


def test_store_appends_never_overwrites_and_survives_a_failed_run(tmp_path):
    st = ResultStore(tmp_path / "r1", "r1")
    base = {"model_name": "M7", "model_version": "original", "protocol": "random", "seed": 0, "fold": 0}
    st.add_run({**base, "status": "ok", "mse": 1.0})
    st.add_run({**base, "model_name": "M9", "status": "failed", "error": "ValueError: boom"})
    st.add_error({"model_name": "M9", "protocol": "random", "seed": 0, "fold": 0,
                  "error_type": "ValueError", "error_message": "boom"})
    before = pd.read_csv(st.runs)
    st.add_run({**base, "seed": 1, "status": "ok", "mse": 2.0})
    after = pd.read_csv(st.runs)
    assert list(after.columns) == RUN_COLUMNS and len(after) == 3
    pd.testing.assert_frame_equal(before, after.iloc[:2].reset_index(drop=True), check_dtype=False)
    assert list(pd.read_csv(st.errors).columns) == ERROR_COLUMNS
    st.finalize()
    s = pd.read_csv(st.dir / "summary.csv")
    assert s.set_index("model_name").loc["M9", "n_failed"] == 1
    with pytest.raises(FileExistsError):
        ResultStore(tmp_path / "r1", "r1")                    # an existing run directory is never reused


def test_registry_and_model_resolution():
    assert {"Dummy", "M7", "M9", "M9_pso", "M9_pso_ridge", "M7_noyear", "M9_tscv",
            "IT2_original", "IT2_improved", "RF_original", "SVM_original"} <= set(REGISTRY)
    assert resolve_models(["all"]) == list(REGISTRY)
    with pytest.raises(ValueError, match="unknown"):
        resolve_models(["nope"])


# ------------------------------- end-to-end smoke (tiny) -------------------------------

def test_end_to_end_chronological_and_failure_isolation(tmp_path, monkeypatch):
    import src.experiments.framework as fw

    def boom(ctx):
        raise RuntimeError("injected failure")

    monkeypatch.setitem(fw.FAMILY_FUNCS, "temporal", boom)
    out = run_experiments(ExperimentArgs(mode="quick", protocols=("chronological",),
                                         models=("Dummy", "M7", "M7_noyear"), seeds=(0,),
                                         out_dir=tmp_path, run_id="t1"))
    runs = pd.read_csv(out / "runs.csv")
    st = runs.set_index("model_name").status
    assert st["Dummy"] == "ok" and st["M7"] == "ok" and st["M7_noyear"] == "failed"
    assert runs[runs.status == "failed"][["mse", "r2"]].isna().all().all()      # failure is NaN, not 0
    err = pd.read_csv(out / "errors.csv")
    assert err.model_name.tolist() == ["M7_noyear"] and "injected failure" in err.error_message[0]
    pred = pd.read_csv(out / "predictions.csv")
    assert set(pred.model_name) == {"Dummy", "M7"}
    # held-out rows are the last 20 % in time order
    assert pd.to_datetime(pred.date).min() >= pd.Timestamp(runs.test_start.iloc[0])
    # metrics in runs.csv recompute exactly from predictions.csv (single evaluation path, MWh)
    for name, g in pred.groupby("model_name"):
        assert np.mean(g.residual ** 2) == pytest.approx(runs.set_index("model_name").loc[name, "mse"])
    cfg = json.loads((out / "experiment_config.json").read_text())
    assert cfg["status"] == "finished" and cfg["dataset"]["sha256"] and "torch" in cfg["versions"]
    assert (out.parent / "index.csv").exists()
    f = Frames.load()
    assert f.dt.is_monotonic_increasing and 1e5 < f.y.mean() < 5e5
