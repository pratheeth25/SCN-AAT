"""XUQ dashboard: 7-rule vs 9-rule vs 9-rule + PSO IT2-ANFIS. Launch (PowerShell, repo root):

    python -m streamlit run streamlit_app.py

Reads existing results only (results/experiments/<run_id>/{runs,predictions}.csv); nothing is trained.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.experiments.compare_models import paired_stats  # noqa: E402
from src.experiments.results_store import build_summary, holm  # noqa: E402
from src.viz import dashboard_data as dd  # noqa: E402

st.set_page_config(page_title="7 vs 9 vs 9 + PSO", layout="wide")

PROTO_ORDER = ["random", "chronological", "rolling"]
PROTO_TEXT = {"random": "Random split (paper protocol)", "chronological": "Chronological (past → future)",
              "rolling": "Rolling origin (5 future blocks)"}
DEFAULT_RUNS = ["full_random_chrono_10seeds", "full_rolling_3seeds"]
PSO_VARIANTS = {"M9_pso": "PSO tunes the 2 new rules",
                "M9_pso_ridge": "PSO + ridge-regularised consequents",
                "M9_tscv": "PSO with time-ordered validation, no `year` input"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


@st.cache_data(show_spinner=False)
def load(root: str, run_ids: tuple, stamp: float):
    runs = dd.load_runs(root, list(run_ids))
    if len(run_ids) > 1 and not runs.empty:
        runs = dd.dedupe_ok(runs)
    return runs, dd.load_table(root, list(run_ids), "predictions")


def stamp_of(root, run_ids) -> float:
    ts = [(Path(root) / r / "runs.csv").stat().st_mtime for r in run_ids if (Path(root) / r / "runs.csv").exists()]
    return max(ts) if ts else 0.0


def fmt(v, nd=1):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{nd}f}"


# ------------------------------- sidebar -------------------------------

st.sidebar.title("7 vs 9 vs 9 + PSO")
pso = st.sidebar.selectbox("9-rule + PSO version", list(PSO_VARIANTS),
                           format_func=lambda m: f"{m}: {PSO_VARIANTS[m]}")
MODELS = {"M7": "7 rules (paper)", "M9": "9 rules", pso: "9 rules + PSO"}
LABEL = {m: f"{t} [{m}]" for m, t in MODELS.items()}
COLOR = dict(zip(LABEL.values(), COLORS))

root = str(dd.default_dir())
available = dd.discover_runs(root)
with st.sidebar.expander("Result folders"):
    root = st.text_input("Experiments folder", root)
    available = dd.discover_runs(root)
    default = [r for r in DEFAULT_RUNS if r in available] or available[:1]
    sel_runs = st.multiselect("Run(s) to read", available, default=default,
                              help="Existing results only. Several runs are pooled (one row per model/protocol/seed/fold).")
    if st.button("Reload"):
        st.cache_data.clear()
        st.rerun()
st.sidebar.caption("MSE in MWh², RMSE / MAE in MWh. No model is trained by this page.")
st.sidebar.caption("Based on Khaled, Mallak, Kaymak & Genga (2026), arXiv:2601.18897, and the original code "
                   "[QusaiKhaled/XUQ](https://github.com/QusaiKhaled/XUQ). Data: Bagherzadeh (2021), "
                   "doi:10.17632/pprkvz3vbd.1.")

if not sel_runs:
    st.info(f"No result folders found in `{root}`. Run `python scripts/run_all_experiments.py --mode full` first.")
    st.stop()

runs_all, preds_all = load(root, tuple(sel_runs), stamp_of(root, sel_runs))
runs = runs_all[runs_all.model_name.isin(MODELS)].copy()
ok = runs[runs.status == "ok"].copy()
if ok.empty:
    st.warning("The selected folders contain no successful runs of M7 / M9 / the chosen PSO model.")
    st.stop()
ok["model"] = ok.model_name.map(LABEL)
protocols = [p for p in PROTO_ORDER if p in set(ok.protocol)]
summary = build_summary(runs)
summary["model"] = summary.model_name.map(LABEL)
order = list(LABEL.values())

# ------------------------------- header -------------------------------

st.title("IT2-ANFIS: 7 rules vs 9 rules vs 9 rules + PSO")
st.markdown(f"- **{LABEL['M7']}**: the paper's 7-rule model (Khaled et al., 2026; original code QusaiKhaled/XUQ).\n"
            f"- **{LABEL['M9']}**: two rules added where the 7-rule model has the largest residuals.\n"
            f"- **{LABEL[pso]}**: {PSO_VARIANTS[pso]}.")
st.caption("Read from: " + ", ".join(f"`{r}`" for r in sel_runs))

metric = st.radio("Metric", list(dd.METRIC_LABELS), format_func=dd.METRIC_LABELS.get, horizontal=True)
lower_better = not dd.HIGHER_IS_BETTER[metric]

# ------------------------------- tiles -------------------------------

for proto in protocols:
    st.subheader(PROTO_TEXT[proto])
    s = summary[summary.protocol == proto].set_index("model_name")
    base = s.loc["M7", f"{metric}_mean"] if "M7" in s.index else np.nan
    best = s[f"{metric}_mean"].idxmin() if lower_better else s[f"{metric}_mean"].idxmax()
    cols = st.columns(3)
    for c, m in zip(cols, MODELS):
        if m not in s.index:
            c.metric(LABEL[m], "no runs")
            continue
        v, sd, n = s.loc[m, f"{metric}_mean"], s.loc[m, f"{metric}_std"], int(s.loc[m, "n_ok"])
        delta = None if m == "M7" or np.isnan(base) else f"{100 * (v - base) / abs(base):+.1f} % vs 7 rules"
        c.metric(LABEL[m] + ("  ★" if m == best else ""), fmt(v, 3 if metric == "r2" else 1), delta,
                 delta_color="inverse" if lower_better else "normal",
                 help=f"mean ± sd = {fmt(v, 3)} ± {fmt(sd, 3)} over {n} runs")
st.caption("★ = best of the three in that protocol. Green delta = better than the 7-rule model.")

# ------------------------------- bar chart + table -------------------------------

st.subheader("Side by side")
fig = px.bar(summary[summary.protocol.isin(protocols)], x="protocol", y=f"{metric}_mean", error_y=f"{metric}_std",
             color="model", barmode="group", color_discrete_map=COLOR,
             category_orders={"protocol": protocols, "model": order}, hover_data=["n_ok"],
             labels={f"{metric}_mean": dd.METRIC_LABELS[metric], "protocol": "", "model": ""})
fig.update_layout(legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, use_container_width=True)

tbl = summary.pivot_table(index="model", columns="protocol", values=f"{metric}_mean").reindex(order)[protocols]
sd = summary.pivot_table(index="model", columns="protocol", values=f"{metric}_std").reindex(order)[protocols]
nd = 3 if metric == "r2" else 1
show = pd.DataFrame({p: [f"{fmt(a, nd)} ± {fmt(b, nd)}" for a, b in zip(tbl[p], sd[p])] for p in protocols},
                    index=tbl.index)
st.dataframe(show, use_container_width=True)
st.caption("Mean ± sd over seeds (random, chronological: 10 seeds; rolling: 3 seeds × 5 folds). Error bars = sd.")

# ------------------------------- paired tests -------------------------------

st.subheader("Is the difference real? Paired tests")
rows = []
for proto in protocols:
    g = ok[ok.protocol == proto]
    key = "fold" if proto == "rolling" else "seed"
    block = []
    for a, b in (("M7", "M9"), ("M7", pso), ("M9", pso)):
        sub = g[g.model_name.isin([a, b])].drop(columns="model").assign(seed=lambda d: d[key])
        sub = sub.rename(columns={"model_name": "model"})
        res = paired_stats(sub[["seed", "model", metric]], metric, a, b)
        if res and res["n_pairs"] >= 2:
            block.append({"protocol": proto, "comparison": f"{MODELS[b]} vs {MODELS[a]}", **res})
    for r, hp in zip(block, holm(np.array([r.get("t_pvalue", np.nan) for r in block]))):
        r["p_holm"] = hp
    rows += block
if rows:
    pt = pd.DataFrame(rows)
    if lower_better:
        pt["change_pct"] = -pt["improvement_pct"]
    else:
        pt["change_pct"] = 100 * (pt["cand_mean"] - pt["ref_mean"]) / pt["ref_mean"].abs()
    pt["verdict"] = np.select(
        [pt.p_holm < 0.05, pt.p_holm.isna()],
        [np.where((pt.change_pct < 0) == lower_better, "better (significant)", "worse (significant)"), "n/a"],
        "no significant difference")
    st.dataframe(pt[["protocol", "comparison", "n_pairs", "ref_mean", "cand_mean", "change_pct", "ci95_low",
                     "ci95_high", "t_pvalue", "p_holm", "verdict"]].round(4), hide_index=True, use_container_width=True)
    st.caption("Paired by seed (rolling: by fold, seed-averaged). `change_pct` = candidate relative to reference "
               "(for errors, negative = lower error). Holm correction over the three comparisons of a protocol.")

# ------------------------------- per-seed spread -------------------------------

st.subheader("Every run")
fig = px.strip(ok[ok.protocol.isin(protocols)], x="protocol", y=metric, color="model", color_discrete_map=COLOR,
               category_orders={"protocol": protocols, "model": order}, hover_data=["seed", "fold"],
               labels={metric: dd.METRIC_LABELS[metric], "protocol": "", "model": ""})
fig.update_traces(marker=dict(size=8))
fig.update_layout(legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, use_container_width=True)

# ------------------------------- predictions -------------------------------

st.subheader("Actual vs predicted on the test days")
pr = preds_all[preds_all.model_name.isin(MODELS)] if len(preds_all) else preds_all
if pr is None or pr.empty:
    st.info("No predictions.csv in the selected folders.")
else:
    c = st.columns(3)
    proto = c[0].selectbox("Protocol", [p for p in protocols if p in set(pr.protocol)], index=0,
                           format_func=PROTO_TEXT.get)
    pp = pr[pr.protocol == proto]
    seed = c[1].selectbox("Seed", sorted(pp.seed.unique()))
    folds = sorted(pp[pp.seed == seed].fold.unique())
    fold = c[2].selectbox("Fold", folds) if proto == "rolling" else folds[0]
    v = pp[(pp.seed == seed) & (pp.fold == fold)].drop_duplicates(["model_name", "date"]).copy()
    v["date"] = pd.to_datetime(v["date"])
    v = v.sort_values("date")
    fig = go.Figure()
    act = v.drop_duplicates("date")
    fig.add_trace(go.Scatter(x=act.date, y=act.actual, name="actual", mode="lines",
                             line=dict(color="#555555", width=1.5)))
    for m in MODELS:
        g = v[v.model_name == m]
        fig.add_trace(go.Scatter(x=g.date, y=g.predicted, name=LABEL[m], mode="lines",
                                 line=dict(color=COLOR[LABEL[m]], width=2)))
    fig.update_layout(yaxis_title="energy (MWh/day)", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    if proto == "random":
        st.caption("Random-split test days are spread over the whole period, so the lines join non-adjacent days.")

# ------------------------------- reading guide -------------------------------

st.info("**How to read this.** The random split mixes past and future days, so it measures interpolation; it is the "
        "paper's protocol. The chronological and rolling protocols test prediction of later, unseen periods. "
        "A version can win on one and lose on the other. Switch the PSO version in the sidebar to compare them.")
