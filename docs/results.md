# Results: 7 rules vs 9 rules vs 9 rules + PSO

> **Attribution.** The IT2-ANFIS method is by Khaled, Mallak, Kaymak and Genga (2026,
> [arXiv:2601.18897](https://arxiv.org/abs/2601.18897)), and the original code base is
> [QusaiKhaled/XUQ](https://github.com/QusaiKhaled/XUQ). Dataset: Bagherzadeh (2021), [doi:10.17632/pprkvz3vbd.1](https://doi.org/10.17632/pprkvz3vbd.1).
>
> **Baseline discrepancy.** The original repository does not include its model files, so all models here, including the
> 7-rule baseline, are this project's own implementation of the paper's equations. Baseline numbers can therefore differ
> from the paper's; see §6.

Every number below is read from existing result files:

- `results/experiments/full_random_chrono_10seeds/runs.csv`: random and chronological protocols, 10 seeds;
- `results/experiments/full_rolling_3seeds/runs.csv`: rolling origin, 3 seeds × 5 folds.

The same numbers appear in the dashboard (`python -m streamlit run streamlit_app.py`). A second, independent full run reproduced them exactly.

**Units:** MSE in MWh², RMSE and MAE in MWh. Lower is better for MSE, RMSE and MAE; higher is better for R².
**Models:** see [implementation.md](implementation.md).

---

## 1. Short answer

| Question | Answer |
|---|---|
| Do 9 rules beat 7 rules? | **Not without PSO.** M9 is no better on the random split (−0.4 %, p = 0.71). It is significantly worse on the chronological split (+15 %). |
| Does 9 rules + PSO beat 7 rules on the paper's protocol (random split)? | **Yes, modestly, with PSO + ridge:** 1616 → 1557 MWh² (−3.7 %, Holm p = 0.017). PSO with Adam consequents gives −2.9 %, which is not significant. |
| Does it forecast the future better? | **The paper-protocol versions do not.** M9_pso is +26 % worse on the chronological test year, and M9_pso_ridge is +9 % (not significant). **The time-aware version M9_tscv does:** −25 % on the chronological split (p < 0.001) and −14 % on rolling folds (not significant with 5 folds). |
| Best choice | Interpolation (paper protocol): **M9_pso_ridge**. Forecasting: **M9_tscv**. |

## 2. Mean results (mean ± sd over runs)

### 2.1 Random split (paper protocol, 10 seeds)

| Model | MSE | RMSE | MAE | R² | vs 7 rules |
|---|---:|---:|---:|---:|---:|
| 7 rules (M7) | 1616.3 ± 115.6 | 40.18 | 31.01 | 0.177 | – |
| 9 rules (M9) | 1610.5 ± 136.5 | 40.10 | 30.92 | 0.180 | −0.4 % |
| 9 rules + PSO (M9_pso) | 1569.9 ± 104.5 | 39.60 | 30.74 | 0.200 | −2.9 % |
| **9 rules + PSO + ridge (M9_pso_ridge)** | **1556.9 ± 98.6** | **39.44** | **30.54** | **0.206** | **−3.7 %** |
| 9 rules + PSO, time-aware (M9_tscv) | 1651.7 ± 124.5 | 40.61 | 31.67 | 0.159 | +2.2 % |
| *Paper, IT2-ANFIS 7 rules (Table 2)* | *1671.21* | *41.10* | *32.07* | – | – |

### 2.2 Chronological split (train 2014–2017, test 2018-05-09 → 2019-06-27, 10 seeds)

| Model | MSE | RMSE | MAE | R² | vs 7 rules |
|---|---:|---:|---:|---:|---:|
| 7 rules (M7) | 2043.5 ± 161.9 | 45.17 | 34.75 | −0.375 | – |
| 9 rules (M9) | 2351.2 ± 117.4 | 48.48 | 38.22 | −0.582 | +15.1 % |
| 9 rules + PSO (M9_pso) | 2578.8 ± 299.5 | 50.70 | 40.05 | −0.735 | +26.2 % |
| 9 rules + PSO + ridge (M9_pso_ridge) | 2231.7 ± 340.9 | 47.12 | 36.97 | −0.501 | +9.2 % |
| **9 rules + PSO, time-aware (M9_tscv)** | **1531.7 ± 34.3** | **39.13** | **30.82** | **−0.030** | **−25.0 %** |

### 2.3 Rolling origin (5 future blocks × 3 seeds = 15 runs)

| Model | MSE | RMSE | MAE | R² | vs 7 rules |
|---|---:|---:|---:|---:|---:|
| 7 rules (M7) | 2639.2 ± 796.2 | 50.75 | 40.68 | −0.550 | – |
| 9 rules (M9) | 2690.8 ± 620.3 | 51.51 | 41.82 | −0.595 | +2.0 % |
| 9 rules + PSO (M9_pso) | 2585.6 ± 687.8 | 50.40 | 40.69 | −0.527 | −2.0 % |
| 9 rules + PSO + ridge (M9_pso_ridge) | 2411.6 ± 607.8 | 48.71 | 39.41 | −0.430 | −8.6 % |
| **9 rules + PSO, time-aware (M9_tscv)** | **2281.0 ± 718.5** | **47.15** | **38.50** | **−0.348** | **−13.6 %** |

## 3. Are the differences real? Paired tests on MSE

**How the tests were run:**
- Pairs are matched by seed for the random and chronological splits, and by fold (seed-averaged) for rolling.
- Tests: paired t-test and 95 % CI of the difference.
- Holm correction over the three comparisons of each protocol.

| Protocol | Comparison | Change | 95 % CI of Δ (MWh²) | p | Holm p | Verdict |
|---|---|---:|---|---:|---:|---|
| random | 9 rules vs 7 | −0.4 % | [−40.7, +29.1] | 0.71 | 0.71 | no difference |
| random | 9 + PSO (M9_pso) vs 7 | −2.9 % | [−96.3, +3.5] | 0.065 | 0.19 | no significant difference |
| random | 9 + PSO + ridge vs 7 | **−3.7 %** | [−96.7, −22.1] | 0.006 | **0.017** | **better** |
| random | 9 + PSO + ridge vs 9 | −3.3 % | [−95.3, −11.8] | 0.018 | 0.035 | better |
| chronological | 9 rules vs 7 | +15.1 % | [+174.6, +440.8] | 0.0005 | 0.002 | worse |
| chronological | 9 + PSO (M9_pso) vs 7 | +26.2 % | [+299.0, +771.6] | 0.0006 | 0.002 | worse |
| chronological | 9 + PSO + ridge vs 7 | +9.2 % | [−89.6, +466.0] | 0.16 | 0.32 | no significant difference |
| chronological | 9 + PSO time-aware vs 7 | **−25.0 %** | [−619.8, −403.9] | < 0.0001 | < 0.0001 | **better** |
| rolling | 9 + PSO + ridge vs 7 | −8.6 % | [−642, +187] | 0.20 | 0.40 | no significant difference |
| rolling | 9 + PSO time-aware vs 7 | −13.6 % | [−837, +121] | 0.11 | 0.32 | no significant difference |

## 4. Cost

Runtimes are means per run on CPU. The 9-rule runtimes include training the 7-rule model, because the new rules are built from its residuals.

| Model | Rules | Parameters total / optimised (random split) | Runtime random / chrono / rolling (s) |
|---|---:|---|---|
| 7 rules | 7 | 375 / 100 | 1.9 / 1.1 / 1.5 |
| 9 rules | 9 | 482 / 128 | 5.2 / 2.3 / 3.7 |
| 9 rules + PSO | 9 | 482 / 207 | 10.1 / 6.9 / 11.2 |
| 9 rules + PSO + ridge | 9 | 482 / 206 | 7.9 / 5.7 / 9.4 |
| 9 rules + PSO, time-aware | 9 | – | 19.2 / 14.0 / 18.8 |

**Search size:** PSO uses up to 20 particles × 41 swarm evaluations, i.e. at most 820 fitness evaluations.

## 5. Interpretation

1. **Two extra rules alone do not help.** The 7 rules already cover the input space. The residual-placed rules (for example "August" and "January" on the random split, "May" and "August" on the chronological split) are trained with the same Adam recipe and add parameters without adding accuracy.
2. **PSO is what makes 9 rules useful on the paper's protocol.** Tuning only the two new rules moves them to where they reduce validation error. Ridge consequents then give the final model exactly the consequents that PSO evaluated, which is why M9_pso_ridge is the one significant gain (−3.7 %, better on 9 of 10 seeds).
3. **Why the paper-protocol versions fail on the future.**
   - **Validation overfit:** PSO lowers the validation error of the single 2017 block, but the test year 2018–19 behaves differently.
   - **Rules that do not carry over:** the calendar patterns found in 2014–2017 (May, August) do not repeat in 2018–19.
   - **`year` extrapolation:** the raw `year` input lies outside the training range, so the linear rule consequents extrapolate along it.
4. **The time-aware version fixes that.** M9_tscv removes `year`, scores PSO candidates on 3 time-ordered folds and tunes ridge λ in time order. It becomes the best of the five on both forecasting protocols.
   - **The cost:** about 2 % accuracy on the random split, because `year` is a useful interpolation feature there.
   - **Where the gain comes from:** a 7-rule model with the same feature and ridge fixes reaches 1540.6 MWh² on the chronological split. Most of the gain is therefore due to those fixes, not to the two extra rules.
5. **R² stays negative on the chronological split.** The test period's mean (279 MWh) and spread differ from the training period's, so a model that is accurate on the past is still offset on the future. M9_tscv comes closest to zero (−0.03).

## 6. Limitations

- **The 7-rule baseline is this project's own implementation.** The original repository does not publish its model files, so the model was written from the paper's equations. It differs from the paper in four ways:
  - the rule positions are frozen, not trained;
  - Adam replaces adaptive mini-batch gradient descent;
  - L1 / L2 are 0.005 / 0.01, not 0.05 / 0.001;
  - q is learned instead of fixed at 0.5.

  The splits and seeds also differ from the paper's runs. Its random-split MSE (1616.3) is therefore 3 % below the paper's reported 1671.2. Do not read it as a reproduction of the paper's table. The 7 vs 9 vs 9 + PSO comparisons are internally consistent, because all models share the same implementation, splits and seeds.
- The data come from one plant. The chronological holdout is one 14-month period, and rolling has only 5 folds.
- **Seeds in the chronological protocol change only the initialisation**, so the sd there measures optimiser noise.
- MAPE (reported in the paper) was not computed.
