"""Parameter sweep: rules 1..K with multiple random seeds."""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from ..config import (
    TEST_SIZE,
    VAL_SIZE,
    RULE_SWEEP_MIN,
    RULE_SWEEP_MAX,
    RULE_SWEEP_N_SEEDS,
)
from ..models.it2_anfis import IT2_TSK_ANFIS
from ..models.anfis_first import FirstOrderANFIS


def _run_sweep_anfis_first(X_pool, y_pool, n_rules, seed, X_test, y_test):
    # Train/val come ONLY from the pool (train+val rows); X_test is never seen.
    model = FirstOrderANFIS(n_rules=n_rules, random_state=seed)
    model.fit(X_pool, y_pool)
    return mean_squared_error(y_test, model.predict(X_test))


def _run_sweep_it2(X_pool, y_pool, n_rules, seed, X_test, y_test):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_pool, y_pool, test_size=VAL_SIZE, random_state=seed
    )
    torch.manual_seed(seed)
    model = IT2_TSK_ANFIS(n_rules=n_rules, X=X_tr, verbose=False, random_state=seed)
    model.fit(
        X_tr, y_tr, X_val, y_val,
        epochs=50, patience=20, batch_size=64,
        freeze_antecedents=True, verbose=False,
    )
    return mean_squared_error(y_test, model.predict(X_test))


def sweep_rules(
    X,
    y,
    model_type="it2",
    n_rules_min=RULE_SWEEP_MIN,
    n_rules_max=RULE_SWEEP_MAX,
    n_seeds=RULE_SWEEP_N_SEEDS,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    base_seed=42,
):
    """
    Sweep number of rules from n_rules_min to n_rules_max; for each, run n_seeds trials.
    model_type: 'it2' or 'anfis_first'.
    Returns dict with keys: n_rules_list, mse_mean, mse_std, mse_per_run (list of lists).
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
    # One fixed held-out test set; every (n_rules, seed) run trains on the remaining rows only.
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=base_seed
    )

    n_rules_list = list(range(n_rules_min, n_rules_max + 1))
    mse_per_run = []
    runner = _run_sweep_it2 if model_type == "it2" else _run_sweep_anfis_first
    seeds = [base_seed + i * 100 for i in range(n_seeds)]

    for n_rules in n_rules_list:
        row = []
        for seed in seeds:
            mse = runner(X_pool, y_pool, n_rules, seed, X_test, y_test)
            row.append(mse)
        mse_per_run.append(row)

    mse_mean = np.array([np.mean(row) for row in mse_per_run])
    mse_std = np.array([np.std(row) for row in mse_per_run])
    return {
        "n_rules_list": n_rules_list,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "mse_per_run": mse_per_run,
        "seeds": seeds,
    }
