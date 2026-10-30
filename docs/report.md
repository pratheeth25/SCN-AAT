# Mini Project Report – Hybrid vs Non-Hybrid Systems

## Energy Consumption Prediction for a Wastewater Treatment Plant: 7-Rule and 9-Rule Interval Type-2 Neuro-Fuzzy Models with and without Particle Swarm Optimisation

> **Attribution.** This mini project builds on existing work.
> - **Method:** the IT2-ANFIS method and the 7-rule design are by Khaled, Mallak, Kaymak and Genga (2026) [1].
> - **Code base:** the data pipeline and project structure come from the original repository, https://github.com/QusaiKhaled/XUQ [2].
>   That repository does not include its model files, so the IT2-ANFIS model code was written in this project from the
>   paper's equations. **The 7-rule baseline results may therefore differ from those reported in the paper.**
> - **Dataset:** by Bagherzadeh et al. [3, 4].
>
> This project added the model implementation, the 9-rule and 9-rule + PSO extensions, the experiments and this write-up.

---

## 1. Abstract

Wastewater treatment plants (WWTPs) use 1–3 % of global electricity, so predicting their daily energy demand helps operators plan equipment, costs and renewable-energy use. This project predicts the daily energy consumption of Melbourne's Eastern Treatment Plant (1382 days, 2014–2019) from flow, water-quality and weather measurements. It uses interval type-2 adaptive neuro-fuzzy inference systems (IT2-ANFIS).

**Systems compared:**
- **Non-hybrid systems:** neuro-fuzzy models trained by gradient descent alone.
  - the 7-rule IT2-ANFIS of Khaled et al. (2026), re-implemented from the paper's equations;
  - a 9-rule IT2-ANFIS, with two rules added where the 7-rule model makes systematic errors.
- **Hybrid system:** the 9-rule IT2-ANFIS combined with Particle Swarm Optimisation (PSO). The swarm tunes the two new rules and least squares solves the rule outputs.

**Evaluation:**
- **Protocols:** the base paper's random split, a past → future chronological split, and rolling-origin forecasting.
- **Repeats:** 10 seeds (rolling: 3 seeds × 5 folds).
- **Statistics:** paired tests.

**Results:**
- Adding two rules alone does not improve accuracy.
- With PSO and ridge consequents, the 9-rule model lowers the random-split MSE from 1616 to 1557 MWh² (−3.7 %, Holm-adjusted p = 0.017).
- A time-aware version of the 9-rule + PSO model lowers the chronological MSE from 2044 to 1532 MWh² (−25 %, p < 0.001).

---

## 2. Introduction

### 2.1 Overview of the Problem Statement

A WWTP's daily energy use depends on several groups of variables:
- **hydraulics:** inflow and outflow;
- **pollutant load:** BOD, COD, ammonia and total nitrogen;
- **weather:** temperature, humidity, pressure, rainfall and wind;
- **the calendar:** season and year.

These relationships are nonlinear, noisy and change over time. The task is a regression problem: given one day's 19 measured inputs, predict that day's energy consumption (target in kWh, reported in MWh).

**Data:**
- **Source:** the Melbourne Water Eastern Treatment Plant dataset, `data/raw/Data-Melbourne_F_fixed.csv`.
- **Size:** 1382 daily rows from 2014-01-01 to 2019-06-27, with no missing values.
- **Target:** mean 275 MWh/day, standard deviation 44.6 MWh/day.

**What a useful model must do:**
1. predict accurately on days it has not seen;
2. keep working on *future* periods, not only on days interleaved with the training data;
3. explain *why* it predicts a value and how uncertain it is.

Fuzzy rule-based models address the third point directly, because every prediction is built from readable IF–THEN rules.

### 2.2 Motivation for the Project

Khaled et al. (2026) proposed a 7-rule interval type-2 neuro-fuzzy model for this plant. It gives explainable predictions and an uncertainty interval through the footprint of uncertainty of its membership functions. Its rules, however, are placed by stratified sampling of the input space and then frozen. Nothing checks whether they cover the conditions in which the model makes its largest errors.

This raises three questions:
1. Does the model improve if rules are added **exactly where it fails**?
2. Do those rules work better if their positions are **searched by a swarm optimiser** rather than fixed by hand?
3. Do these changes still help when the model must predict a **future** period?

Answering them compares a non-hybrid neuro-fuzzy design with a hybrid neuro-fuzzy + swarm design on the same data.

### 2.3 Objectives of the Project

1. **Design and implement the models.** Extend the 7-rule IT2-ANFIS to a 9-rule IT2-ANFIS whose two extra rules are placed from residual analysis (non-hybrid). Then build a 9-rule + PSO hybrid in which a particle swarm optimises those rules.
2. **Compare 7 rules, 9 rules and 9 rules + PSO fairly.** Use identical data splits under random, chronological and rolling-origin protocols. Report accuracy, training time, complexity and robustness with paired statistical tests, without using test data for any modelling choice.

---

## 3. Literature Review

### 3.1 Existing Systems and Techniques

**Fuzzy logic and fuzzy rules:**
- **Fuzzy logic** (Zadeh, 1965) represents vague concepts by membership degrees in [0, 1].
- **Fuzzy rule-based systems** combine IF–THEN rules over such memberships. In **Mamdani systems** the THEN part is a fuzzy set. In **Takagi–Sugeno–Kang (TSK) systems** (Takagi & Sugeno, 1985) it is a function of the inputs, usually linear, which makes TSK systems easy to fit to data.

**Neuro-fuzzy learning:**
- **ANFIS** (Jang, 1993) writes a TSK system as a five-layer adaptive network: fuzzification, rule firing, normalisation, consequents and output.
- Its parameters are learned from data. Gradient descent tunes the membership functions, and least squares solves the linear consequents.

**Type-2 fuzzy sets:**
- **Type-2 fuzzy sets** (Mendel & John, 2002) add uncertainty to the membership functions themselves.
- **Interval type-2 (IT2) sets** bound this uncertainty between a lower and an upper membership function. The region between them is the footprint of uncertainty (FOU).
- **Type reduction:** turning an IT2 output into a crisp value requires type reduction, either the iterative Karnik–Mendel algorithm (Karnik & Mendel, 2001) or faster weighted approximations.

**Swarm optimisation:**
- **Particle Swarm Optimisation** (Kennedy & Eberhart, 1995; Shi & Eberhart, 1998) is a gradient-free search in which candidate solutions move under inertia, attraction to their own best position and attraction to the swarm's best.
- It is widely used to tune fuzzy membership functions where gradients are unreliable.

**The base system:** Khaled et al. (2026) built a first-order TSK IT2-ANFIS for this plant.
- **Model:** seven rules with uncertain-mean Gaussian membership functions.
- **Training:** mini-batch gradient descent.
- **Output:** a q-weighted type reduction.
- **Reported accuracy:** a test MSE of 1671.21 MWh² on a random 64/16/20 split.

### 3.2 Limitations of Existing Systems

1. **Fixed, error-blind rule base.** The 7 rules are placed by stratified sampling and frozen. Their positions are never compared with where the model's errors are concentrated, and the rule count of 7 is not justified by the data.
2. **No search over rule positions.** Only the linear consequents are learned. The nonlinear membership parameters, which decide *where* each rule applies, are not optimised.
3. **Random-split evaluation only.** Shuffled splits put training days on both sides of every test day, so they measure interpolation, not forecasting of a future period. Calendar inputs such as `year` look harmless under shuffling but force the linear rule outputs to extrapolate in time.
4. **Single validation block.** One validation period serves model selection. A tuned model can therefore fit that period's peculiarities rather than general behaviour.
5. **No repeated-seed significance tests.** Seed-to-seed variation (sd ≈ 100–160 MWh²) is as large as the differences being claimed.

---

## 4. System Architecture

### 4.1 Overall System Block Diagram

```
 ┌─────────────────────┐   ┌──────────────────────────┐   ┌───────────────────────────────┐
 │ Raw data (CSV)      │──▶│ Pre-processing           │──▶│ Split protocol                │
 │ 1382 days, 19 inputs│   │ clean, rename target,    │   │ random 64/16/20 │ chronological│
 │ target in kWh       │   │ dates; MI feature select │   │ 64/16/20 │ rolling origin ×5   │
 └─────────────────────┘   │ (training rows only)     │   └──────────────┬────────────────┘
                           └──────────────────────────┘                  │ train / val / test
                 ┌──────────────────────────────────────────────────────┴──────────────┐
                 ▼                                                                     ▼
 ┌────────────────────────────────────┐            ┌─────────────────────────────────────────────┐
 │ NON-HYBRID (neuro-fuzzy only)      │            │ HYBRID (neuro-fuzzy + PSO)                  │
 │ 7 rules: stratified placement,     │            │ 7 frozen rules + 2 residual rules           │
 │          Adam-trained consequents  │──residuals▶│ PSO searches the 2 new rules (c1, c2, σ)    │
 │ 9 rules: + 2 rules on residual     │            │ consequents: ridge least squares / Adam     │
 │          groups, Adam consequents  │            │ time-aware variant: no `year`, rolling PSO  │
 └─────────────────┬──────────────────┘            └──────────────────────┬──────────────────────┘
                   └─────────────────────────────┬──────────────────────────┘
                                                 ▼
                     ┌───────────────────────────────────────────────────┐
                     │ Evaluation (test rows used once)                  │
                     │ MSE, RMSE, MAE, R² in MWh; runtime; rule count    │
                     │ paired t-test / 95 % CI / Holm correction         │
                     └──────────────────────────┬────────────────────────┘
                                                ▼
                     ┌───────────────────────────────────────────────────┐
                     │ results/experiments/<run_id>/*.csv  →  Streamlit  │
                     │ dashboard (7 vs 9 vs 9 + PSO) and this report     │
                     └───────────────────────────────────────────────────┘
```

**Common pipeline.** All models receive exactly the same rows, the same features (mutual-information selection fitted on training rows only) and the same seeds. The 9-rule models start from the same fitted 7-rule model.

**Experiment runner.** `scripts/run_all_experiments.py` trains each model for each protocol and seed. It appends one row per run to `runs.csv` and every test prediction to `predictions.csv`.

**Dashboard.** `streamlit_app.py` reads the stored results and trains nothing.

**Leakage rules.**
- Training rows: scalers, feature selection, residual analysis and ridge λ.
- Validation rows: early stopping and PSO fitness.
- Test rows: predicted once, at the end.

---

## 5. Non-Hybrid System Design

### 5.1 Description of the Non-Hybrid Approach

**Definition used in this report:**
- **Non-hybrid** means the neuro-fuzzy model trained by **one learning mechanism only**: gradient descent on the rule consequents, with the membership functions fixed by a deterministic placement rule. No evolutionary or swarm search is involved.
- **Hybrid** adds PSO.

**The two non-hybrid models:**
- **7 rules (M7):** the base paper's IT2-ANFIS, implemented in this project from the paper's equations [1]. The original repository [2] does not include its model code. Seven rules are spread evenly over each input's range and frozen. Each rule's linear output is learned by Adam.
- **9 rules (M9):** M7 plus two rules aimed at its systematic errors. The training residuals of M7 are grouped by calendar month, year and weekend. The two groups with the most significant mean error (for example August and January) each receive a new rule centred on that group's typical input. The 7 original rules are kept unchanged, and all 9 rules' outputs are retrained by Adam.

### 5.2 Algorithms Used

**IT2 membership (per rule j, input f).** Each rule has an uncertain-mean Gaussian with centre interval [c1, c2] and spread σ.
- UMF = 1 inside [c1, c2], with Gaussian tails outside.
- LMF = the lower of the two edge Gaussians.

**Inference:**
1. **Firing interval (product t-norm):** f_L = Π μ_L and f_U = Π μ_U.
2. **Normalisation:** each strength is divided by the sum over all rules.
3. **Rule outputs (first-order TSK):** y_j = b_j + w_j·x.
4. **Type reduction:** y_L = Σ f̄_L,j y_j and y_U = Σ f̄_U,j y_j.
5. **Output:** ŷ = q·y_L + (1 − q)·y_U, with q learned.

**7-rule placement.** Each input's range is cut into 7 segments, and rule r draws its centre in segment r. The FOU half-width is 0.1 of a segment and σ is half a segment.

**Consequent training.** Adam with batch size 64 and at most 100 epochs. The loss is MSE + L1 (0.005) + L2 (0.01), with early stopping on validation MSE (patience 30). The antecedents are frozen.

**9-rule placement:**
1. Compute the residuals r = y − ŷ_M7 on the training rows.
2. For each calendar group with at least 20 rows, compute t = mean(r) / (sd(r)/√n).
3. Take the two groups with the largest |t| ≥ 3. If fewer qualify, use k-means (k = 2) on the largest-error rows.
4. Build each new rule from its group: centre = the group's mean input; σ = the group's sd, floored at the baseline σ; FOU = the baseline's.

### 5.3 System Architecture of Non-Hybrid Model

```
            ┌────────── Layer 1 ──────────┐ ┌─ Layer 2 ─┐ ┌─ Layer 3 ─┐ ┌───── Layer 4 ─────┐ ┌──── Layer 5 ────┐
 x₁ … x_F ─▶│ IT2 Gaussian MFs per rule   │▶│ product   │▶│ normalise │▶│ y_j = b_j + w_j·x  │▶│ q·y_L+(1−q)·y_U │─▶ ŷ
            │ LMF / UMF (c1, c2, σ) FIXED │ │ f_L, f_U  │ │ f̄_L, f̄_U  │ │ learned by Adam    │ │                 │
            └──────────────▲──────────────┘ └───────────┘ └───────────┘ └────────────────────┘ └─────────────────┘
                           │ M7: 7 rules, stratified placement
                           │ M9: same 7 rules + 2 rules placed on residual groups (training rows only)
```

**What is learned:**
- Only layer 4 (the consequents) and q are learned, by gradient descent.
- Layer 1 is set once by a placement rule and never changed.

### 5.4 Implementation Details

| Item | Setting |
|---|---|
| Language / libraries | Python 3.13, PyTorch, NumPy, pandas, scikit-learn |
| Model class | `src/models/it2_anfis.py` (`IT2_TSK_ANFIS`) |
| 7-rule recipe | `src/experiments/compare_models.py` (`fit_baseline7`) |
| Residual rules | `src/optimization/targeted_rules.py` (`find_residual_patterns`, `init_rules_from_rows`, `build_extended_model`) |
| 9-rule pipeline | `src/experiments/targeted9.py` (`fit_models`) |
| Feature selection | `mutual_info_regression` on training rows, keep MI > 0 (9–16 inputs) |
| Splits | random 64/16/20 (10 seeds), chronological 64/16/20 (10 seeds), rolling origin (3 seeds × 5 folds) |
| Metrics | `src/utils/metrics.evaluate_predictions` after the kWh → MWh conversion |
| Safety check | the code asserts that M7's test predictions are unchanged after the 9-rule pipeline |
| Hardware | CPU only (`CUDA_VISIBLE_DEVICES=-1`), Windows 11 |

```powershell
python scripts/run_all_experiments.py --mode full --protocols random chronological --models M7 M9
```

### 5.5 Results of Non-Hybrid System

MSE in MWh², mean ± sd. Source: `results/experiments/full_random_chrono_10seeds`, `full_rolling_3seeds`.

| Model | Random (10 seeds) | Chronological (10 seeds) | Rolling (15 runs) |
|---|---:|---:|---:|
| Paper, IT2-ANFIS 7 rules (reported) | 1671.2 | – | – |
| **7 rules (M7)** | **1616.3 ± 115.6** (R² 0.177) | **2043.5 ± 161.9** | **2639.2 ± 796.2** |
| 9 rules (M9) | 1610.5 ± 136.5 (R² 0.180) | 2351.2 ± 117.4 | 2690.8 ± 620.3 |

**Observations:**
- **The 7-rule model is at the paper's accuracy level:** 1616 vs a reported 1671. It is an independent implementation with frozen rule positions, Adam training and a learned q, on different splits, so a discrepancy with the paper's baseline is expected.
- **Adding two residual-targeted rules without optimisation gives no gain:** −0.4 % on the random split (p = 0.71), +2.0 % on rolling (p = 0.76).
- **It is significantly worse on the future test year:** +15.1 %, Holm p = 0.002. The groups it targets in 2014–2017 (May, August) do not repeat in 2018–19.

---

## 6. Hybrid System Design (Soft Computing Based)

### 6.1 Introduction to Hybrid Soft Computing Models

Soft computing combines techniques that tolerate imprecision and uncertainty, each with its own strength:

| Technique | Strength | Weakness alone |
|---|---|---|
| Fuzzy logic | interpretable rules | no learning |
| Neural-network-style training | fits parameters to data | needs gradients, finds local optima |
| Swarm and evolutionary algorithms | gradient-free global search | have no model of their own |

A hybrid system joins them so each covers another's weakness.

**The hybrid here (9 rules + PSO):**
- **Fuzzy logic** provides the rule structure.
- **PSO** searches the nonlinear membership parameters of the two new rules, which gradient descent left fixed in the non-hybrid models.
- **Least squares (ridge)**, or Adam, fits the linear rule outputs.

This is the classic ANFIS split between nonlinear premise parameters and linear consequent parameters, with PSO in place of gradient descent for the premises.

### 6.2 Techniques Used

| Technique | Role |
|---|---|
| **Interval type-2 fuzzy sets** (uncertain-mean Gaussian) | rule conditions with a footprint of uncertainty |
| **First-order TSK rules** | local linear models y_j = b_j + w_j·x |
| **Residual analysis + k-means fallback** | initial positions of the 2 new rules |
| **Particle Swarm Optimisation** | searches the new rules' centres, FOU widths and spreads |
| **Ridge least squares** | closed-form consequents θ = (DᵀD + λI)⁻¹Dᵀy for every particle and for the final model |
| **Adam + L1/L2** | consequents in the M9_pso variant |
| **Cross-validation** | ridge λ on training rows; time-ordered folds in the time-aware variant |

**PSO update for particle i:**

```
v_i ← w·v_i + c1·r1·(pbest_i − x_i) + c2·r2·(gbest − x_i)
x_i ← clip(x_i + v_i, bounds)
```

- Inertia w falls from 0.9 to 0.4, and c1 = c2 = 1.5.
- Velocity is limited to 20 % of each box width.

**Encoding:** each particle encodes (m, h, log σ) per rule and input, decoded as c1 = m − h, c2 = m + h and σ = e^{log σ}. Every particle is therefore a valid fuzzy set.

### 6.3 Hybrid System Architecture

```
                         ┌──────────────────────────── PSO loop (20 particles, ≤ 40 iterations) ─────────────────────────┐
                         │  particle = parameters of rules 8–9  →  ridge-solve consequents on training rows  →          │
                         │  fitness = validation MSE / Var(y_val)   (time-aware: mean over 3 time-ordered folds)        │
                         └───────────────▲──────────────────────────────────────────────────────┬───────────────────────┘
                                         │ initial swarm: residual-based rules + jitter + uniform │ best rules 8–9
            ┌────────── Layer 1 ─────────┴──┐ ┌─ Layer 2 ─┐ ┌─ Layer 3 ─┐ ┌──── Layer 4 ────┐ ┌──── Layer 5 ────┐
 x₁ … x_F ─▶│ rules 1–7 frozen, 8–9 by PSO  │▶│ product   │▶│ normalise │▶│ ridge (or Adam)  │▶│ q·y_L+(1−q)·y_U │─▶ ŷ
            └───────────────────────────────┘ └───────────┘ └───────────┘ └─────────────────┘ └─────────────────┘
```

**Variants:**

| Variant | Consequents | Fitness | Inputs |
|---|---|---|---|
| **M9_pso** | Adam (as M7) | one validation block | MI-selected, including `year` |
| **M9_pso_ridge** | ridge, exactly as evaluated during PSO | one validation block | same |
| **M9_tscv** (time-aware) | ridge, λ by time-series CV | mean over 3 expanding-window folds | `year` removed |

### 6.4 Implementation Details

| Item | Setting |
|---|---|
| PSO | `src/optimization/pso_rule_optimizer.py` (`ParticleSwarmOptimizer`, `AntecedentCodec`) |
| Targeted search | `src/optimization/targeted_rules.py` (`TargetedRuleObjective`, `optimize_new_rules`) |
| Swarm | 20 particles, ≤ 40 iterations, stop after 10 without improvement; particle 0 = residual-based start, half the swarm = jitter around it, the rest uniform |
| Search dimension | 2 rules × 3 × F (≈ 50–100 values) |
| Bounds | centre within the training range ± 10 %, FOU half-width ∈ [0, 1], σ ∈ [0.05, 5] (standardised units) |
| Ridge λ | {0.1, 1, 10, 100} by 5-fold CV on training rows; {1, 10, 100, 1000} by time-series CV for M9_tscv |
| Pipelines | `src/experiments/targeted9.py` (M9_pso, M9_pso_ridge), `src/experiments/temporal.py` (M9_tscv) |
| Reproducibility | fixed seeds; tests check that the PSO output is identical for the same seed |
| Tests | 72 unit tests (`python -m pytest -q`): MF constraints, PSO, leakage, metrics |

```powershell
python scripts/run_all_experiments.py --mode full --protocols random chronological --models M7 M9 M9_pso M9_pso_ridge M9_tscv
python scripts/run_all_experiments.py --mode full --protocols rolling --seeds 3 --models M7 M9 M9_pso M9_pso_ridge M9_tscv
```

### 6.5 Results of Hybrid System

MSE in MWh², mean ± sd.

| Model | Random (10 seeds) | Chronological (10 seeds) | Rolling (15 runs) |
|---|---:|---:|---:|
| 9 rules + PSO (M9_pso) | 1569.9 ± 104.5 | 2578.8 ± 299.5 | 2585.6 ± 687.8 |
| **9 rules + PSO + ridge (M9_pso_ridge)** | **1556.9 ± 98.6** (R² 0.206) | 2231.7 ± 340.9 | 2411.6 ± 607.8 |
| **9 rules + PSO, time-aware (M9_tscv)** | 1651.7 ± 124.5 | **1531.7 ± 34.3** | **2281.0 ± 718.5** |

**Observations:**
- **PSO + ridge gives the only significant gain on the paper's protocol:** −3.7 % vs 7 rules (Holm p = 0.017) and −3.3 % vs 9 rules (Holm p = 0.035), better on 9 of 10 seeds.
- **PSO with Adam consequents** gives −2.9 % (Holm p = 0.19, not significant).
- **The time-aware variant is the best model for forecasting:** −25.0 % vs 7 rules and −34.9 % vs 9 rules on the chronological split (both p < 0.0001), and −13.6 % on rolling (p = 0.11, only 5 folds).

---

## 7. Comparative Analysis of Hybrid and Non-Hybrid Systems

### 7.1 Performance Metrics

| Metric | Formula | Unit | Meaning |
|---|---|---|---|
| MSE | (1/n) Σ (yᵢ − ŷᵢ)² | MWh² | main criterion, as in the base paper |
| RMSE | √MSE | MWh | typical error size |
| MAE | (1/n) Σ |yᵢ − ŷᵢ| | MWh | robust to outliers |
| R² | 1 − SSE / SST | – | share of test variance explained; < 0 means worse than a constant at the test mean |
| Runtime | wall-clock time per run | s | training cost |
| Paired tests | paired t-test, 95 % CI of the difference, Holm correction | – | whether a difference survives seed-to-seed noise |

### 7.2 Accuracy Comparison

| Model | Type | Random MSE | Chronological MSE | Rolling MSE |
|---|---|---:|---:|---:|
| 7 rules (M7) | non-hybrid | 1616.3 | 2043.5 | 2639.2 |
| 9 rules (M9) | non-hybrid | 1610.5 | 2351.2 | 2690.8 |
| 9 rules + PSO (M9_pso) | hybrid | 1569.9 | 2578.8 | 2585.6 |
| 9 rules + PSO + ridge (M9_pso_ridge) | hybrid | **1556.9** | 2231.7 | 2411.6 |
| 9 rules + PSO, time-aware (M9_tscv) | hybrid | 1651.7 | **1531.7** | **2281.0** |

Paired MSE tests vs 7 rules (Holm-adjusted p within each protocol):

| Protocol | 9 rules | 9 rules + PSO | 9 rules + PSO + ridge | 9 rules + PSO, time-aware |
|---|---|---|---|---|
| random | −0.4 % (0.71) | −2.9 % (0.19) | **−3.7 % (0.017)** | +2.2 % (0.20) |
| chronological | +15.1 % (0.002) | +26.2 % (0.002) | +9.2 % (0.32) | **−25.0 % (< 0.0001)** |
| rolling | +2.0 % (0.76) | −2.0 % (1.0) | −8.6 % (0.40) | −13.6 % (0.32) |

### 7.3 Computational Complexity

N = training rows (≈ 880 random), F = inputs (9–16), K = rules, P = particles, I = PSO iterations.

| Model | Training | Prediction per sample | Parameters (random split) total / optimised |
|---|---|---|---|
| 7 rules | O(epochs · N · K · F), K = 7 | O(K · F) | 375 / 100 |
| 9 rules | the same, K = 9, plus residual grouping O(N) | O(K · F) | 482 / 128 |
| 9 rules + PSO | P·(I + 1) fitness evaluations, each O(N·K·F + (K(F+1))³) for the ridge solve | O(K · F) | 482 / 207 |
| 9 rules + PSO + ridge | the same PSO cost, then one ridge solve | O(K · F) | 482 / 206 |

- **Prediction cost:** all models are equally cheap, a few hundred multiply–adds per day.
- **Training cost:** the hybrid models add up to 20 × 41 = 820 ridge solves, and they optimise about twice as many parameters as the 7-rule model.

### 7.4 Training Time Comparison

Mean seconds per run on CPU. The 9-rule times include training the 7-rule model they start from.

| Model | Random | Chronological | Rolling |
|---|---:|---:|---:|
| 7 rules | 1.87 | 1.11 | 1.54 |
| 9 rules | 5.20 | 2.28 | 3.66 |
| 9 rules + PSO | 10.07 | 6.94 | 11.24 |
| 9 rules + PSO + ridge | 7.92 | 5.72 | 9.37 |
| 9 rules + PSO, time-aware | 19.20 | 13.98 | 18.85 |

**Relative cost:**
- The hybrid models cost 4–12× the 7-rule training time.
- The time-aware variant is the most expensive, because each particle is scored on 3 folds.
- In absolute terms every model trains in seconds on 1382 days.

### 7.5 Robustness and Adaptability

**Robustness to time shift** (random → chronological MSE):

| Model | Change |
|---|---:|
| 7 rules | +26 % |
| 9 rules | +46 % |
| 9 rules + PSO | +64 % |
| 9 rules + PSO + ridge | +43 % |
| 9 rules + PSO, time-aware | **−7 %** |

Why the models behave this way:
- **9 rules and the paper-protocol hybrids degrade.** Their new rules target 2014–2017 calendar patterns. PSO also fits the single validation year, and the linear rule outputs extrapolate along `year`.
- **The time-aware hybrid is stable.** Removing `year` and scoring PSO on time-ordered folds makes it the most stable model.

**Robustness to initialisation** (sd of chronological MSE over seeds):

| Model | sd |
|---|---:|
| 7 rules | 162 |
| 9 rules | 117 |
| 9 rules + PSO | 299 |
| 9 rules + PSO + ridge | 341 |
| 9 rules + PSO, time-aware | **34** |

**Adaptability:**
- **All models:** new rules are added without touching the existing ones. The 7 original rules are bit-for-bit unchanged, which the code checks.
- **Hybrid models:** they also move the new rules to where they reduce error. The non-hybrid 9-rule model cannot.

### 7.6 Strengths and Weaknesses of Each Approach

| | Non-hybrid (7 rules, 9 rules) | Hybrid (9 rules + PSO) |
|---|---|---|
| **Strengths** | simple; fast training (1–5 s); deterministic placement; easy to explain; the 7-rule model is the most robust of the paper-protocol models on future data | the only significant accuracy gain on the paper's protocol (−3.7 %); rule positions optimised without gradients; the time-aware variant gives the best forecasting accuracy (−25 %) and the lowest seed variance |
| **Weaknesses** | rule positions never optimised; adding rules alone gives no gain and can hurt on future data | 4–12× slower training; can overfit a single validation block (+26 % on future data for M9_pso); needs a time-aware fitness to forecast well |

---

## 8. Results and Discussion

### 8.1 Output Analysis

1. **7 → 9 rules (non-hybrid).** Placing two rules on statistically significant residual groups is not enough by itself.
   - Random split: −0.4 % (p = 0.71).
   - Chronological split: 15 % worse, because the calendar patterns found in 2014–2017 do not repeat in 2018–19.
2. **9 rules → 9 rules + PSO (hybrid).** Letting the swarm move the two new rules, and solving their outputs exactly by ridge, gives the one statistically supported gain on the paper's protocol.
   - MSE 1616 → 1557 MWh², RMSE 40.18 → 39.44 MWh, R² 0.177 → 0.206.
   - The gain is real but small: about 0.7 MWh of RMSE on a 40 MWh scale.
   - With Adam consequents instead of ridge, the gain shrinks to −2.9 % and is not significant. The final model's consequents differ from those PSO evaluated.
3. **Generalisation to the future.**
   - **Paper protocol:** PSO lowers the error on the single validation block but raises it on the following year (+9 % to +26 %).
   - **Time-aware design** (no `year`, PSO scored on time-ordered folds, time-ordered ridge tuning): the 9-rule + PSO model reaches 1532 MWh² on the future test year, 25 % below the 7-rule model, with the lowest seed variance (sd 34).
4. **Which model to use.**
   - Interpolation (paper protocol): **9 rules + PSO + ridge**.
   - Forecasting future days: **9 rules + PSO, time-aware**.
5. **Uncertainty output.** All models give an interval [y_L, y_U] from the footprint of uncertainty.
   - It measures *structural* uncertainty and covers fewer than 10 % of test values.
   - It must be calibrated (for example split-conformal prediction, about 90 % coverage on the random split) before being used as a prediction interval.

**Dashboard.** `python -m streamlit run streamlit_app.py` shows all of these comparisons from the stored result files.

---

## 9. Conclusion

### 9.1 Summary of Work

**Work done:**
- The 7-rule interval type-2 neuro-fuzzy model of Khaled et al. (2026) was implemented from the paper's equations, because the original repository [2] does not include its model files. It reaches the paper's accuracy level (1616 vs 1671 MWh²), though the baseline numbers are not identical to the paper's.
- It was extended to a 9-rule model, with two rules placed by residual analysis (non-hybrid).
- A 9-rule + PSO hybrid was built, in which a particle swarm optimises the new rules and ridge least squares solves the rule outputs.
- All models were compared on identical splits under random, chronological and rolling-origin protocols with paired tests.

**Findings:**
- **Adding rules alone does not help.**
- **The hybrid improves accuracy on the paper's protocol** by 3.7 % (Holm p = 0.017).
- **A time-aware hybrid improves forecasting of the future test year by 25 %** (p < 0.001).

The optimiser and the evaluation design matter more than the number of rules.

### 9.2 Future Scope

1. **Seasonal rule inputs:** cyclic season inputs or seasonal antecedents, to capture calendar effects without extrapolating along `year`.
2. **Calibrated uncertainty:** train with an interval-score loss, or calibrate with conformal prediction on a separate calibration block.
3. **Wider search:** other population-based optimisers (genetic algorithm, differential evolution), and multi-objective search over accuracy, rule count and interpretability.
4. **Rule pruning and merging:** for more compact, readable rule bases.
5. **More data:** additional plants, multi-step-ahead forecasting, and lagged inputs.

---

## 10. Applications

- **WWTP operations:** day-ahead energy budgeting; scheduling aeration and pumping in low-tariff hours; aligning energy-intensive tasks with on-site biogas or solar generation.
- **Energy management and sustainability reporting:** baseline consumption for carbon accounting and for detecting abnormal energy use.
- **Explainable decision support:** each prediction can be traced to the rules that fired ("high inflow, cold season") and to an uncertainty interval, which helps operators trust the forecast.
- **Other utilities and process plants:** the same 9-rule + PSO approach applies to drinking-water treatment, district heating, building energy and industrial load forecasting.

---

## Acknowledgements

- The IT2-ANFIS method, the 7-rule model and the original code base are the work of Qusai Khaled, Bahjat Mallak,
  Uzay Kaymak and Laura Genga (Eindhoven University of Technology and Haskoning) [1, 2]. This project uses and
  extends their published code (data pipeline and project structure); all rights to that code remain with its author.
  The model implementation was written in this project from the paper, because the original repository does not include it.
- The Melbourne Eastern Treatment Plant dataset was published by Faramarz Bagherzadeh and co-authors [3, 4].

## 11. References

1. Q. Khaled, B. Mallak, U. Kaymak and L. Genga, "Explainable Uncertainty Quantification for Wastewater Treatment Energy Prediction via Interval Type-2 Neuro-Fuzzy System," arXiv:2601.18897, 2026.
2. Q. Khaled, XUQ: Interval Type-2 ANFIS for Uncertainty Quantification (code repository), https://github.com/QusaiKhaled/XUQ, 2026.
3. F. Bagherzadeh, "Full Scale Wastewater Treatment Plant Data," Mendeley Data, V1, doi:10.17632/pprkvz3vbd.1, 2021.
4. F. Bagherzadeh, A. S. Nouri, M.-J. Mehrani and S. Thennadil, "Prediction of energy consumption and evaluation of affecting factors in a full-scale WWTP using a machine learning approach," *Process Safety and Environmental Protection*, 154, 458–466, 2021, doi:10.1016/j.psep.2021.08.040.
5. L. A. Zadeh, "Fuzzy sets," *Information and Control*, 8(3), 338–353, 1965.
6. T. Takagi and M. Sugeno, "Fuzzy identification of systems and its applications to modeling and control," *IEEE Trans. Systems, Man, and Cybernetics*, 15(1), 116–132, 1985.
7. J.-S. R. Jang, "ANFIS: adaptive-network-based fuzzy inference system," *IEEE Trans. Systems, Man, and Cybernetics*, 23(3), 665–685, 1993.
8. J. M. Mendel and R. I. John, "Type-2 fuzzy sets made simple," *IEEE Trans. Fuzzy Systems*, 10(2), 117–127, 2002.
9. N. N. Karnik and J. M. Mendel, "Centroid of a type-2 fuzzy set," *Information Sciences*, 132, 195–220, 2001.
10. J. Kennedy and R. Eberhart, "Particle swarm optimization," *Proc. IEEE Int. Conf. Neural Networks*, 1942–1948, 1995.
11. Y. Shi and R. Eberhart, "A modified particle swarm optimizer," *Proc. IEEE Int. Conf. Evolutionary Computation*, 69–73, 1998.
12. A. E. Hoerl and R. W. Kennard, "Ridge regression: biased estimation for nonorthogonal problems," *Technometrics*, 12(1), 55–67, 1970.
13. D. P. Kingma and J. Ba, "Adam: a method for stochastic optimization," *ICLR*, 2015.
14. S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics*, 6(2), 65–70, 1979.
