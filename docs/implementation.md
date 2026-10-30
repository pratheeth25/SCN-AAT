# Implementation: 7 rules vs 9 rules vs 9 rules + PSO

> **Attribution.** This project builds on the original code base
> [QusaiKhaled/XUQ](https://github.com/QusaiKhaled/XUQ) by Qusai Khaled. It also builds on the paper by Khaled, Mallak, Kaymak and Genga
> (2026, [arXiv:2601.18897](https://arxiv.org/abs/2601.18897)), which owns the IT2-ANFIS method. The original code also supplies the data
> pipeline and the 7-rule model. Dataset: Bagherzadeh (2021), [doi:10.17632/pprkvz3vbd.1](https://doi.org/10.17632/pprkvz3vbd.1).

How the three compared IT2-ANFIS models are built in this repository, from the raw CSV to the dashboard.
The fuzzy-logic theory behind them is in [fuzz.md](fuzz.md); the numbers are in [results.md](results.md).

| Name in code | Short name | What it is |
|---|---|---|
| `M7` | 7 rules | The 7-rule interval type-2 TSK ANFIS of Khaled et al. (2026), as implemented in the original repository |
| `M9` | 9 rules | `M7` + 2 rules placed where the 7-rule model makes systematic errors |
| `M9_pso` | 9 rules + PSO | `M9`, with the 2 new rules tuned by Particle Swarm Optimisation |
| `M9_pso_ridge` | 9 rules + PSO + ridge | `M9_pso`, with closed-form ridge consequents instead of Adam |
| `M9_tscv` | 9 rules + PSO (time-aware) | `M9_pso_ridge` made forecast-safe: no `year` input, time-ordered PSO fitness and ridge tuning |

---

## 1. Code map

| Step | File | Main functions |
|---|---|---|
| Load and clean data | `src/data/load_data.py`, `src/data/preprocessing.py` | `load_csv`, `clean_and_prepare` |
| Splits, feature selection | `src/experiments/compare_models.py` | `split_rows`, `select_features_train_only` |
| Rolling-origin splits, `no_year` policy | `src/experiments/temporal.py` | `rolling_origin_splits`, `expanding_folds`, `select_features` |
| IT2-ANFIS model | `src/models/it2_anfis.py` | `IT2_TSK_ANFIS`, `_IT2TSKNetwork`, `_ridge_init_consequents` |
| 7-rule training recipe | `src/experiments/compare_models.py` | `fit_baseline7` |
| Residual-targeted rules | `src/optimization/targeted_rules.py` | `find_residual_patterns`, `cluster_fallback`, `init_rules_from_rows`, `build_extended_model` |
| PSO | `src/optimization/pso_rule_optimizer.py` | `PSOConfig`, `ParticleSwarmOptimizer`, `AntecedentCodec` |
| PSO on the 2 new rules | `src/optimization/targeted_rules.py` | `TargetedRuleObjective`, `optimize_new_rules` |
| Ridge λ selection | `src/optimization/rule_count_search.py` | `select_ridge_lambda` |
| M7 → M9 → M9_pso → M9_pso_ridge | `src/experiments/targeted9.py` | `fit_models` |
| M9_tscv | `src/experiments/temporal.py` | `RollingRuleObjective`, `fit_temporal_family` |
| Metrics and units | `src/utils/metrics.py` | `evaluate_predictions`, `convert_units` |
| Experiment runner and result files | `src/experiments/framework.py`, `src/experiments/results_store.py`, `scripts/run_all_experiments.py` | `run_experiments`, `ResultStore`, `build_summary`, `build_paired_tests` |
| Dashboard | `streamlit_app.py`, `src/viz/dashboard_data.py` | — |
| Tests | `tests/` | 72 tests (`python -m pytest -q`) |

---

## 2. Data

- **File:** `data/raw/Data-Melbourne_F_fixed.csv`, from the Melbourne Eastern Treatment Plant.
  It has 1382 daily rows from 2014-01-01 to 2019-06-27, no missing values and no duplicates.
- **Target:** `Energy Consumption`, daily, in kWh.
  Every model is fitted in kWh. Metrics are converted once, by `convert_units`, to MWh: MSE in MWh², RMSE and MAE in MWh.
- **Inputs (19 columns):**
  - flows: `avg_inflow`, `avg_outflow`;
  - water quality: `Am`, `BOD`, `COD`, `TN`;
  - weather: `T`, `TM`, `Tm`, `SLP`, `H`, `PP`, `VV`, `V`, `VM`, `VG`;
  - calendar: `year`, `month`, `day`.
- **Feature selection:** mutual information between each input and the target, computed on the **training rows only**. Inputs with MI > 0 are kept (`select_features_train_only`).
- **Standardisation:**
  - inputs are standardised with the training mean and standard deviation (`IT2_TSK_ANFIS._init_input_scaler`);
  - the target is standardised the same way during training and transformed back at prediction time.

## 3. Evaluation protocols

| Protocol | Rows | Repeats |
|---|---|---|
| **random** (paper protocol) | 64 % train, 16 % validation, 20 % test, drawn at random per seed | 10 seeds |
| **chronological** | first 64 % train, next 16 % validation, last 20 % test (2018-05-09 → 2019-06-27) | 10 seeds (seed changes initialisation only) |
| **rolling** | expanding window: 5 consecutive test blocks over the last 50 % of the series. Each fold trains on everything before its block; the last 20 % of that window is validation. | 3 seeds × 5 folds |

**Leakage rules, used in every model:**
- Training rows fit the scalers, the MI selection, the residual analysis and ridge λ.
- Validation rows drive early stopping and the PSO fitness.
- Test rows are predicted once, after every choice is made.

---

## 4. The common IT2-ANFIS model (`src/models/it2_anfis.py`)

`IT2_TSK_ANFIS(n_rules, X=X_train)` builds an interval type-2, first-order Takagi–Sugeno–Kang network in PyTorch.

**Parameters per rule *j* and input *f*:**
- an uncertain centre interval `[c1_jf, c2_jf]`;
- a spread `σ_jf`.

**Consequent per rule:** a linear model `y_j = b_j + w_j · x`.

**Forward pass (`_IT2TSKNetwork.forward`):**
1. Compute the lower and upper Gaussian memberships of each input, LMF and UMF ([fuzz.md](fuzz.md) §5).
2. Combine them with the product t-norm, in log space, to get the rule firing interval `[f_L, f_U]`. The row-wise maximum is subtracted before exponentiating to avoid underflow.
3. Normalise the firing strengths over the rules.
4. Type reduction gives `y_L = Σ f̄_L,j · y_j` and `y_U = Σ f̄_U,j · y_j`.
5. Defuzzification gives `ŷ = q · y_L + (1 − q) · y_U`, where `q = sigmoid(logit_q)` starts at 0.5 and is trainable.

**Ridge consequents (`_ridge_init_consequents`):**
- With the antecedents fixed, ŷ is linear in the consequents.
- They are therefore solved in closed form: `θ = (DᵀD + λI)⁻¹ Dᵀy`, where `D` stacks `φ_j(x) · [1, x]` and `φ_j = q·f̄_L,j + (1 − q)·f̄_U,j`.
- Every term is penalised, including biases. Without this, a rule that never fires on training data gets an unidentifiable, exploding bias.

---

## 5. Model 1: `M7`, 7 rules (`fit_baseline7`)

**Rule placement, Latin-hypercube style (`_build_network`):**
- Each input's training range is cut into 7 equal segments.
- Rule *r* draws its centre uniformly inside segment *r* (seeded RNG).
- FOU: `c1, c2 = centre ∓ 0.1 × segment width` (`alpha = 0.2` of the segment width, split over both sides).
- Spread: `σ = 0.5 × segment width` (at least 0.05).

**Training:**
- Antecedents are **frozen**; only the consequents and q are learned.
- Optimiser: Adam with batch size 64, up to 100 epochs (`IT2_EPOCHS`).
- Loss: MSE + L1 (0.005) + L2 (0.01) on the consequents.
- Early stopping on validation MSE, patience 30.

**Size:** 3·7·F antecedent values + 7·(F + 1) consequents + q. That averages 375 parameters on the random split (F = 9–15 selected inputs), 100 of them trained.

## 6. Model 2: `M9`, 9 rules (`targeted9.fit_models`)

1. **Residuals of M7 on the training rows:** `r = y − ŷ_M7`.
2. **Find systematic errors (`find_residual_patterns`):**
   - Group the residuals by calendar month, year and weekend/weekday.
   - For each group with at least 20 rows, compute the mean residual and its t-statistic `t = mean / (sd/√n)`.
   - The two groups with the largest |t| ≥ 3 become the new rules. The threshold was fixed before any run.
   - **Fallback (`cluster_fallback`):** if fewer than two groups qualify, k-means (k = 2) on the training rows in the top residual quartile.
   - Example: on the chronological split every seed picked May and August.
3. **Initialise each new rule from its rows (`init_rules_from_rows`):**
   - centre = mean standardised input of the group;
   - σ = the group's standard deviation per input, floored at the baseline's σ;
   - FOU half-width = the baseline's.
   - This guarantees c1 ≤ c2 and σ > 0.
4. **Build the 9-rule model (`build_extended_model`):** the 7 baseline antecedents are copied unchanged, the 2 new rules are appended, and consequents start fresh.
5. **Train** with exactly the M7 recipe: frozen antecedents, Adam + L1/L2, early stopping.
6. **Check:** the code asserts that M7's test predictions are bit-for-bit unchanged after the 9-rule pipeline.

## 7. Model 3: `M9_pso`, 9 rules + PSO

PSO tunes **only the 2 new rules' antecedents**. The 7 original rules stay frozen.

**Encoding (`AntecedentCodec`):**
- Each rule and input is encoded as `(m, h, log σ)`, with `c1 = m − h`, `c2 = m + h`, `σ = exp(log σ)`.
- Therefore c1 ≤ c2 and σ > 0 by construction.

**Bounds:**
- m within the training range ± 10 %;
- h ∈ [0, 1];
- σ ∈ [0.05, 5] (standardised units).

**Search dimension:** 2 rules × 3 × F values (about 100).

**Fitness (`TargetedRuleObjective`), for each particle:**
1. Decode the particle into antecedents.
2. Solve the consequents by ridge on the training rows.
3. Predict the validation rows.
4. Fitness = validation MSE / Var(y_val).

Ridge λ ∈ {0.1, 1, 10, 100} is chosen beforehand by 5-fold CV on the training rows (`select_ridge_lambda`).

**Optimiser (`ParticleSwarmOptimizer`):**
- global-best PSO with 20 particles and up to 40 iterations;
- stops after 10 iterations without improvement;
- inertia falls linearly from 0.9 to 0.4, c₁ = c₂ = 1.5;
- velocity limited to 20 % of each box width, positions clipped to the bounds;
- non-finite fitness counts as +∞.

**Swarm start (`optimize_new_rules`):**
- particle 0 = the residual-based initialisation from §6;
- half the swarm = Gaussian jitter around it;
- the rest = uniform in the box.

**After PSO:** the best antecedents are written into a fresh 9-rule model, whose consequents are trained with the M7 recipe (Adam + L1/L2).

## 8. Variant: `M9_pso_ridge`

The PSO antecedents are the same as in `M9_pso`. The consequents are the ridge solution alone (`fit(..., epochs=0, ridge_lambda=λ)`), with no Adam. This makes the final model match exactly what the PSO fitness evaluated.

## 9. Variant: `M9_tscv`, time-aware 9 rules + PSO (`src/experiments/temporal.py`)

This variant changes three things for forecasting:

1. **`no_year` feature policy.** The raw `year` column is removed before MI selection, because in a past → future split it always lies outside the training range.
2. **Rolling-origin PSO fitness (`RollingRuleObjective`).** The fitness is the mean validation MSE over 3 expanding-window folds inside train + validation, with the consequents re-solved on each fold's past. It replaces the single validation block.
3. **Time-ordered ridge tuning.** λ ∈ {1, 10, 100, 1000} is chosen by `TimeSeriesSplit` on the training rows instead of shuffled K-fold.

The rule construction (residual targeting, 2 new rules, frozen 7) and the PSO settings are those of §6–§8.

---

## 10. Running and storing experiments

`scripts/run_all_experiments.py` runs every model through `src/experiments/framework.py`.

- **Matched runs:** within one protocol, seed and fold, all models see identical rows and features. The 9-rule models reuse that context's fitted M7.
- **Output files** go to `results/experiments/<run_id>/`. All CSVs are append-only and written after each model:

| File | Content |
|---|---|
| `runs.csv` | one row per model / protocol / seed / fold: metrics, periods, rule count, runtime, status, config |
| `predictions.csv` | every test prediction (date, actual, predicted, residual) |
| `summary.csv` | mean / sd per model and protocol |
| `paired_tests.csv` | paired t-test, Wilcoxon, 95 % CI, Cohen's d_z, Holm-adjusted p |
| `errors.csv`, `experiment_config.json` | failures with tracebacks; every setting and seed |

The results used in [results.md](results.md) are `results/experiments/full_random_chrono_10seeds` (random and chronological, 10 seeds) and `results/experiments/full_rolling_3seeds` (rolling, 3 seeds × 5 folds).

## 11. Dashboard (`streamlit_app.py`)

The dashboard reads the CSVs above and trains nothing. It shows M7, M9 and one 9-rule + PSO version, chosen in the sidebar: `M9_pso`, `M9_pso_ridge` or `M9_tscv`. It contains:
- tiles per protocol, with the % change vs 7 rules;
- a grouped bar chart;
- a mean ± sd table;
- paired tests (Holm over the 3 comparisons of each protocol);
- every run as a point;
- actual vs predicted on the test days.

## 12. Commands (PowerShell, from `C:\XUQ`)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt   # once
$env:CUDA_VISIBLE_DEVICES = "-1"                                                       # every new terminal (CPU)
python -m pytest -q                                                                    # 72 tests
python scripts/run_all_experiments.py --mode quick                                     # smoke test, ~4 min
python scripts/run_all_experiments.py --mode full --protocols random chronological `
    --models M7 M9 M9_pso M9_pso_ridge M9_tscv                                         # 10 seeds
python scripts/run_all_experiments.py --mode full --protocols rolling --seeds 3 `
    --models M7 M9 M9_pso M9_pso_ridge M9_tscv                                         # 3 seeds x 5 folds
python -m streamlit run streamlit_app.py                                               # http://localhost:8501
```

Each run writes a new folder and never overwrites an existing one (`--run-id` must be new).
