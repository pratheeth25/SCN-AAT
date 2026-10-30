# IT2-ANFIS for WWTP Energy Prediction: 7 Rules vs 9 Rules vs 9 Rules + PSO

> **Attribution: this is not original work.** This repository is a fork of
> **[QusaiKhaled/XUQ](https://github.com/QusaiKhaled/XUQ)** by Qusai Khaled. It accompanies the paper
> *Explainable Uncertainty Quantification for Wastewater Treatment Energy Prediction via Interval Type-2
> Neuro-Fuzzy System* (Khaled, Mallak, Kaymak and Genga, 2026, [arXiv:2601.18897](https://arxiv.org/abs/2601.18897)).
> The original authors own the IT2-ANFIS method, the original code base and the research idea. The dataset belongs to
> its creators (see [Dataset](#dataset)). This fork only adds experiments on top of their work; see [Credits](#credits).

Interval type-2 adaptive neuro-fuzzy inference systems (IT2-ANFIS) that predict the daily energy consumption of
Melbourne's Eastern Treatment Plant. The project compares three models:

| Model | Description |
|---|---|
| **7 rules** (`M7`) | The 7-rule IT2-ANFIS of Khaled et al. (2026), from the original repository |
| **9 rules** (`M9`) | `M7` plus 2 rules placed where `M7` makes systematic errors |
| **9 rules + PSO** (`M9_pso`, `M9_pso_ridge`, `M9_tscv`) | The 2 new rules tuned by Particle Swarm Optimisation; `M9_tscv` is the time-aware variant for forecasting |

Results (MSE in MWh², lower is better):

| Protocol | 7 rules | 9 rules | Best 9 rules + PSO |
|---|---:|---:|---:|
| Random split (paper protocol) | 1616 | 1610 | **1557** (`M9_pso_ridge`, −3.7 %, Holm p = 0.017) |
| Chronological (train past, test future) | 2044 | 2351 | **1532** (`M9_tscv`, −25 %, p < 0.001) |

## Documentation

| File | Content |
|---|---|
| [docs/implementation.md](docs/implementation.md) | how the three models are implemented, plus step-by-step commands |
| [docs/fuzz.md](docs/fuzz.md) | every fuzzy-logic and soft-computing concept used |
| [docs/results.md](docs/results.md) | full results and paired significance tests |
| [docs/report.md](docs/report.md) | mini-project report (hybrid vs non-hybrid systems) |

## Repository layout

```
├── streamlit_app.py          dashboard: 7 vs 9 vs 9 + PSO (reads stored results, trains nothing)
├── run_pipeline.py           single-run CLI: load → clean → feature selection → train IT2-ANFIS
├── scripts/
│   ├── run_all_experiments.py   runs models × protocols × seeds, writes results/experiments/<run_id>/
│   └── summarize_runs.py        pools run folders and prints summary / paired-test tables
├── src/
│   ├── models/it2_anfis.py      IT2 TSK ANFIS (PyTorch)
│   ├── optimization/            PSO, rule encoding, residual-targeted rules, ridge λ selection
│   ├── experiments/             7 → 9 → 9 + PSO pipelines, runner, result storage
│   ├── data/, features/         loading, cleaning, mutual-information feature selection
│   ├── utils/metrics.py         metrics and kWh → MWh conversion
│   └── viz/                     dashboard data helpers, membership-function plots
├── tests/                    pytest suite
├── results/experiments/      stored runs used by the docs and the dashboard
├── notebooks/                exploratory notebooks
└── docs/                     the four documents above
```

## Dataset

This project uses the **Full-Scale Wastewater Treatment Plant Data** for the Melbourne Eastern Treatment Plant: 1382 days from 2014 to 2019, with 19 inputs and the daily energy consumption in kWh.

- **Source:** [Mendeley Data](https://data.mendeley.com/datasets/pprkvz3vbd/1).
  Citation: Bagherzadeh, F. (2021), "Full Scale Wastewater Treatment Plant Data", Mendeley Data, V1, [doi:10.17632/pprkvz3vbd.1](https://doi.org/10.17632/pprkvz3vbd.1).
- **Setup:** download the CSV and save it as `data/raw/Data-Melbourne_F_fixed.csv`. Data files are not stored in this repository.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1                 # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
$env:CUDA_VISIBLE_DEVICES = "-1"             # run on CPU (faster for this data size)

python -m pytest -q                          # tests
python -m streamlit run streamlit_app.py     # dashboard at http://localhost:8501 (uses stored results)
```

On Linux or macOS, activate with `source .venv/bin/activate` and use `CUDA_VISIBLE_DEVICES=-1` before each command.

Re-run the experiments (each run writes a new folder under `results/experiments/`):

```powershell
python scripts/run_all_experiments.py --mode quick                       # smoke test, a few minutes
python scripts/run_all_experiments.py --mode full --protocols random chronological `
    --models M7 M9 M9_pso M9_pso_ridge M9_tscv                           # 10 seeds
python scripts/run_all_experiments.py --mode full --protocols rolling --seeds 3 `
    --models M7 M9 M9_pso M9_pso_ridge M9_tscv                           # 3 seeds × 5 folds
python scripts/run_all_experiments.py --list-models                      # all registered models
```

The full set of options and the output-file formats are in [docs/implementation.md](docs/implementation.md).

## Credits

| What | Credit |
|---|---|
| IT2-ANFIS method, research idea, paper | **Qusai Khaled, Bahjat Mallak, Uzay Kaymak, Laura Genga**: Eindhoven University of Technology and Haskoning |
| Original code base: IT2-ANFIS model, data loading and cleaning, mutual-information feature selection, baselines, notebooks, tests, CI | **Qusai Khaled**, [github.com/QusaiKhaled/XUQ](https://github.com/QusaiKhaled/XUQ) |
| Dataset | **Faramarz Bagherzadeh** et al., Melbourne Water Eastern Treatment Plant data ([Mendeley Data](https://doi.org/10.17632/pprkvz3vbd.1); [PSEP 2021](https://doi.org/10.1016/j.psep.2021.08.040)) |
| Underlying techniques | fuzzy sets (Zadeh), TSK rules (Takagi and Sugeno), ANFIS (Jang), type-2 fuzzy sets (Mendel and John), PSO (Kennedy and Eberhart). Full references are in [docs/fuzz.md](docs/fuzz.md). |

**What this fork adds** on top of the original work:
- the 9-rule and 9-rule + PSO extensions;
- the multi-protocol experiment runner;
- stored results;
- the comparison dashboard;
- the documents in `docs/`.

These additions build directly on the authors' model and code, and are not presented as independent research.

**License:** the original repository does not state a license. All rights to the original code remain with its author. Please contact the original authors before reusing their code beyond reference and study.

## Citation

If you use anything from this repository, cite the original paper and repository:

> Q. Khaled, B. Mallak, U. Kaymak and L. Genga, "Explainable Uncertainty Quantification for Wastewater Treatment
> Energy Prediction via Interval Type-2 Neuro-Fuzzy System," arXiv:2601.18897, 2026.
> [doi:10.48550/arXiv.2601.18897](https://doi.org/10.48550/arXiv.2601.18897).
> Code: [github.com/QusaiKhaled/XUQ](https://github.com/QusaiKhaled/XUQ)

```bibtex
@article{khaled2026explainable,
  title={Explainable Uncertainty Quantification for Wastewater Treatment Energy Prediction via Interval Type-2 Neuro-Fuzzy System},
  author={Khaled, Qusai and Mallak, Bahjat and Kaymak, Uzay and Genga, Laura},
  journal={arXiv preprint arXiv:2601.18897},
  year={2026}
}

@misc{khaled2026xuq,
  author={Khaled, Qusai},
  title={XUQ: Interval Type-2 ANFIS for Uncertainty Quantification},
  howpublished={\url{https://github.com/QusaiKhaled/XUQ}},
  year={2026}
}

@misc{bagherzadeh2021data,
  author={Bagherzadeh, Faramarz},
  title={Full Scale Wastewater Treatment Plant Data},
  howpublished={Mendeley Data, V1, doi:10.17632/pprkvz3vbd.1},
  year={2021}
}
```
