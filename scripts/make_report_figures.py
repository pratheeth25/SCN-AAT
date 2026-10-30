#!/usr/bin/env python3
"""Figures for docs/report.md, drawn from stored results only (no model is trained).

  python scripts/make_report_figures.py          # writes docs/figures/*.png

Reads results/experiments/full_random_chrono_10seeds and full_rolling_3seeds, plus the local dataset
(data/raw/Data-Melbourne_F_fixed.csv) for the data overview figure.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TARGET_COLUMN  # noqa: E402
from src.data import clean_and_prepare, load_csv  # noqa: E402

RUNS = ROOT / "results" / "experiments"
OUT = ROOT / "docs" / "figures"
MODELS = ["M7", "M9", "M9_pso", "M9_pso_ridge", "M9_tscv"]
LABEL = {"M7": "7 rules", "M9": "9 rules", "M9_pso": "9 rules + PSO",
         "M9_pso_ridge": "9 rules + PSO + ridge", "M9_tscv": "9 rules + PSO (time-aware)"}
COLOR = {"M7": "#0072B2", "M9": "#E69F00", "M9_pso": "#009E73", "M9_pso_ridge": "#D55E00", "M9_tscv": "#CC79A7"}
PROTOS = ["random", "chronological", "rolling"]
PTXT = {"random": "Random split", "chronological": "Chronological", "rolling": "Rolling origin"}
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight"})


def load_runs():
    r = pd.concat([pd.read_csv(RUNS / d / "runs.csv")
                   for d in ("full_random_chrono_10seeds", "full_rolling_3seeds")], ignore_index=True)
    return r[(r.status == "ok") & r.model_name.isin(MODELS)]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_data(raw):
    dt = pd.to_datetime(raw[["year", "month", "day"]])
    y = raw[TARGET_COLUMN].to_numpy(float) / 1000
    n = len(y); n_te = int(round(n * 0.2)); n_va = int(round((n - n_te) * 0.2)); n_tr = n - n_te - n_va
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for a, b, c, t in ((0, n_tr, "#0072B2", "train 64 %"), (n_tr, n_tr + n_va, "#E69F00", "validation 16 %"),
                       (n_tr + n_va, n, "#009E73", "test 20 %")):
        ax.axvspan(dt.iloc[a], dt.iloc[b - 1], color=c, alpha=0.10, lw=0)
        ax.text(dt.iloc[a + (b - a) // 2], 405, t, ha="center", va="bottom", color=MUTED, fontsize=9)
    ax.plot(dt, y, color=INK, lw=0.6)
    ax.plot(dt, pd.Series(y).rolling(30, center=True).mean(), color="#0072B2", lw=2, label="30-day mean")
    ax.set_ylabel("energy (MWh/day)"); ax.set_ylim(100, 430); ax.legend(loc="lower left")
    ax.set_title("Daily energy consumption, Melbourne Eastern Treatment Plant (chronological split shaded)",
                 loc="left", fontsize=10)
    save(fig, "fig1_data_and_split.png")


def fig_membership():
    x = np.linspace(-3, 3.5, 600)
    c1, c2, s = 0.0, 0.6, 0.8
    g = lambda c: np.exp(-0.5 * ((x - c) / s) ** 2)  # noqa: E731
    umf = np.where(x < c1, g(c1), np.where(x > c2, g(c2), 1.0))
    lmf = np.where(x <= (c1 + c2) / 2, g(c2), g(c1))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.fill_between(x, lmf, umf, color="#0072B2", alpha=0.18, lw=0, label="footprint of uncertainty (FOU)")
    ax.plot(x, umf, color="#0072B2", lw=2, label="upper MF (UMF)")
    ax.plot(x, lmf, color="#0072B2", lw=2, ls="--", label="lower MF (LMF)")
    for c, t in ((c1, "c1"), (c2, "c2")):
        ax.axvline(c, color=MUTED, lw=0.8, ls=":"); ax.text(c, 1.04, t, ha="center", color=MUTED)
    ax.set_xlabel("standardised input x"); ax.set_ylabel("membership degree"); ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Interval type-2 Gaussian MF with uncertain mean [c1, c2], spread σ", loc="left", fontsize=10)
    save(fig, "fig3_it2_membership.png")


def _box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=MUTED, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=INK, wrap=True)


def _arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    _box(ax, 0.1, 3.5, 1.8, 0.8, "Raw data\n1382 days, 19 inputs", "#f2f2f2")
    _box(ax, 2.3, 3.5, 2.2, 0.8, "Pre-processing\nclean, MI feature selection\n(training rows only)", "#f2f2f2")
    _box(ax, 4.9, 3.5, 2.2, 0.8, "Split protocol\nrandom / chronological /\nrolling origin", "#f2f2f2")
    _box(ax, 0.1, 1.7, 4.1, 1.1, "NON-HYBRID (neuro-fuzzy only)\n7 rules: stratified placement, Adam consequents\n"
         "9 rules: + 2 rules on residual groups", "#dceaf6")
    _box(ax, 5.0, 1.7, 4.4, 1.1, "HYBRID (neuro-fuzzy + PSO)\nPSO searches the 2 new rules (c1, c2, σ)\n"
         "ridge / Adam consequents; time-aware variant", "#dcf1ea")
    _box(ax, 2.2, 0.2, 5.2, 0.9, "Evaluation (test rows used once)\nMSE, RMSE, MAE, R², runtime, paired tests (Holm)",
         "#f2f2f2")
    _box(ax, 7.9, 0.25, 2.0, 0.8, "Stored results\n→ dashboard, report", "#f2f2f2")
    _arrow(ax, (1.9, 3.9), (2.3, 3.9)); _arrow(ax, (4.5, 3.9), (4.9, 3.9))
    _arrow(ax, (6.0, 3.5), (2.4, 2.8)); _arrow(ax, (6.0, 3.5), (7.2, 2.8))
    _arrow(ax, (4.2, 2.25), (5.0, 2.25)); ax.text(4.6, 2.35, "residuals", ha="center", fontsize=7.5, color=MUTED)
    _arrow(ax, (2.4, 1.7), (4.0, 1.1)); _arrow(ax, (7.2, 1.7), (6.0, 1.1))
    _arrow(ax, (7.4, 0.65), (7.9, 0.65))
    save(fig, "fig2_system_architecture.png")


def fig_mse(r):
    s = r.groupby(["protocol", "model_name"]).mse.agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, p in zip(axes, PROTOS):
        g = s[s.protocol == p].set_index("model_name").reindex(MODELS)
        xs = np.arange(len(MODELS))
        ax.bar(xs, g["mean"], yerr=g["std"], color=[COLOR[m] for m in MODELS], width=0.72,
               edgecolor="white", linewidth=2, error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
        base = g.loc["M7", "mean"]
        ax.axhline(base, color=MUTED, lw=0.8, ls="--")
        for i, m in enumerate(MODELS):
            ax.text(i, g.loc[m, "mean"] / 2, f"{g.loc[m, 'mean']:.0f}", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
        ax.set_xticks(xs, ["7", "9", "9+PSO", "9+PSO\n+ridge", "9+PSO\ntime-aware"], fontsize=8)
        ax.set_title(PTXT[p], loc="left", fontsize=10); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("test MSE (MWh²), mean ± sd")
    fig.suptitle("Test MSE by model and protocol (lower is better; dashed line = 7 rules)", x=0.01, ha="left",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_mse_comparison.png")


def fig_seeds(r):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, p in zip(axes, ["random", "chronological"]):
        g = r[r.protocol == p].pivot_table(index="seed", columns="model_name", values="mse")[MODELS]
        xs = np.arange(len(MODELS))
        for _, row in g.iterrows():
            ax.plot(xs, row.values, color=GRID, lw=0.8, zorder=1)
        for i, m in enumerate(MODELS):
            ax.scatter(np.full(len(g), i), g[m], s=26, color=COLOR[m], edgecolor="white", lw=0.8, zorder=2)
            ax.scatter(i, g[m].mean(), s=80, marker="D", color=INK, zorder=3)
        ax.set_xticks(xs, ["7", "9", "9+PSO", "9+PSO\n+ridge", "9+PSO\ntime-aware"], fontsize=8)
        ax.set_title(f"{PTXT[p]}: each seed (grey lines join matched seeds, ◆ = mean)", loc="left", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("test MSE (MWh²)")
    fig.tight_layout()
    save(fig, "fig5_per_seed.png")


def fig_robustness_time(r):
    s = r.groupby(["protocol", "model_name"]).agg(mse=("mse", "mean"), sec=("runtime_seconds", "mean"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    chg = [100 * (s.loc[("chronological", m), "mse"] / s.loc[("random", m), "mse"] - 1) for m in MODELS]
    ys = np.arange(len(MODELS))[::-1]
    a.barh(ys, chg, color=[COLOR[m] for m in MODELS], height=0.62, edgecolor="white", lw=2)
    a.axvline(0, color=MUTED, lw=0.8)
    for y, v in zip(ys, chg):
        a.text(v + (2 if v >= 0 else -2), y, f"{v:+.0f} %", va="center", ha="left" if v >= 0 else "right",
               fontsize=8.5, color=INK)
    a.set_yticks(ys, [LABEL[m] for m in MODELS], fontsize=8.5); a.set_xlim(-25, 80)
    a.set_xlabel("MSE change, random → chronological"); a.grid(axis="y", visible=False)
    a.set_title("Robustness to a time shift (lower is better)", loc="left", fontsize=10)
    w = 0.26
    for k, p in enumerate(PROTOS):
        vals = [s.loc[(p, m), "sec"] for m in MODELS]
        b.barh(ys + (1 - k) * w, vals, height=w, color=COLOR[MODELS[0]], alpha=[1, 0.6, 0.3][k],
               edgecolor="white", lw=1, label=PTXT[p])
    b.set_yticks(ys, [LABEL[m] for m in MODELS], fontsize=8.5); b.set_xlabel("training time per run (s, CPU)")
    b.legend(fontsize=8, loc="upper right"); b.grid(axis="y", visible=False)
    b.set_title("Training time", loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig6_robustness_and_time.png")


def fig_predictions():
    p = pd.read_csv(RUNS / "full_random_chrono_10seeds" / "predictions.csv")
    p = p[(p.protocol == "chronological") & (p.seed == 0)].copy()
    p["date"] = pd.to_datetime(p["date"])
    show = ["M7", "M9_pso", "M9_tscv"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    act = p[p.model_name == "M7"].sort_values("date")
    ax.plot(act.date, act.actual.rolling(7, center=True).mean(), color=INK, lw=2.2, label="actual (7-day mean)")
    for m in show:
        g = p[p.model_name == m].sort_values("date")
        ax.plot(g.date, g.predicted.rolling(7, center=True).mean(), color=COLOR[m], lw=1.6, label=LABEL[m])
    ax.set_ylabel("energy (MWh/day)"); ax.legend(ncol=4, fontsize=8.5, loc="upper left")
    ax.set_title("Chronological test period (2018-05-09 → 2019-06-27), seed 0: actual vs predicted, 7-day means",
                 loc="left", fontsize=10)
    save(fig, "fig7_chrono_predictions.png")


def main():
    r = load_runs()
    raw = clean_and_prepare(load_csv())
    fig_data(raw); fig_membership(); fig_architecture(); fig_mse(r); fig_seeds(r); fig_robustness_time(r)
    fig_predictions()


if __name__ == "__main__":
    main()
