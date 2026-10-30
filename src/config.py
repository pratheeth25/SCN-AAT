"""Default configuration and constants for the IT2-ANFIS pipeline."""

from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data
DEFAULT_CSV_NAME = "Data-Melbourne_F_fixed.csv"
TARGET_COLUMN = "Energy Consumption"
DROP_COLUMNS = ["Unnamed: 0"]
# The CSV carries no unit metadata. The values (116k-398k/day) are kWh; the paper
# reports the same quantity in MWh. Models fit in TARGET_UNIT, results are
# reported in REPORT_UNIT through an explicit conversion (utils.metrics.convert_units).
TARGET_UNIT = "kWh"
REPORT_UNIT = "MWh"

# Feature selection
MI_RANDOM_STATE = 0
MIN_MI_THRESHOLD = 0.0  # Keep features with MI > this (0 = drop only exactly zero)

# Train/val/test
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # of training set
RANDOM_STATE = 42

# IT2-ANFIS defaults
DEFAULT_N_RULES = 7
IT2_M1 = 1.5
IT2_M2 = 3.0
IT2_MIN_SIGMA = 0.05
IT2_EPOCHS = 100
IT2_PATIENCE = 30
IT2_BATCH_SIZE = 64
IT2_FREEZE_ANTECEDENTS = True
IT2_LAMBDA_L1 = 0.005
IT2_LAMBDA_L2 = 0.01
IT2_Q = 0.5

# Rule sweep
RULE_SWEEP_MIN = 1
RULE_SWEEP_MAX = 50
RULE_SWEEP_N_SEEDS = 10

# ---------------------------------------------------------------------------
# PSO rule / membership optimisation (src/optimization, src/experiments/compare_models.py)
# ---------------------------------------------------------------------------
RULE_COUNT_CANDIDATES = list(range(3, 16))        # K in {3..15}; baseline K = DEFAULT_N_RULES
PSO_LAMBDA_RULE = 0.005        # primary complexity weight (fixed a priori): one extra rule
                               # must cut validation MSE by 0.5 % of Var(y_val)
PSO_LAMBDA_SENSITIVITY = [0.0, 0.0025, 0.005, 0.01, 0.02]
PSO_RIDGE_GRID = [0.1, 1.0, 10.0, 100.0]          # consequent L2, chosen by CV on train rows
INTERVAL_ALPHA = 0.10          # nominal 90 % intervals for interval score / conformal
EXPERIMENT_SEEDS = list(range(10))
SPLIT_PROTOCOLS = ["random", "chronological"]
PSO_BUDGETS = {
    # stage1 = per-K enumeration budget, stage2 = refinement budget (B and C share it)
    "quick": {"stage1": {"n_particles": 8, "max_iter": 5, "patience": 3},
              "stage2": {"n_particles": 10, "max_iter": 10, "patience": 5},
              "seeds": [0, 1], "k_values": [3, 7, 11], "finetune_epochs": 20},
    "full": {"stage1": {"n_particles": 16, "max_iter": 20, "patience": 6},
             "stage2": {"n_particles": 24, "max_iter": 60, "patience": 12},
             "seeds": EXPERIMENT_SEEDS, "k_values": RULE_COUNT_CANDIDATES, "finetune_epochs": 100},
}

# ---------------------------------------------------------------------------
# Temporal generalisation (src/experiments/temporal.py, framework.py)
# ---------------------------------------------------------------------------
TEMPORAL_RIDGE_GRID = [1.0, 10.0, 100.0, 1000.0]   # chosen by expanding-window CV on train rows
ROLLING_INITIAL = 0.5          # first rolling-origin training window = 50 % of the series
ROLLING_VAL_FRAC = 0.2         # last 20 % of every training window = validation
ROLLING_FITNESS_FOLDS = 3      # expanding-window folds inside train+val for the PSO fitness
FEATURE_POLICIES = ("raw", "no_year")
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
