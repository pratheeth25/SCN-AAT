"""Single-run training harness (config-driven).

The IT2_TSK_ANFIS model handles its own input and target standardisation
internally, so this module only owns the data splitting and model instantiation.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from ..config import (
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    DEFAULT_N_RULES,
    IT2_EPOCHS,
    IT2_PATIENCE,
    IT2_BATCH_SIZE,
    IT2_FREEZE_ANTECEDENTS,
    IT2_LAMBDA_L1,
    IT2_LAMBDA_L2,
)
from ..models.it2_anfis import IT2_TSK_ANFIS


def run_it2_training(
    X,
    y,
    n_rules=None,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    epochs=IT2_EPOCHS,
    patience=IT2_PATIENCE,
    batch_size=IT2_BATCH_SIZE,
    freeze_antecedents=IT2_FREEZE_ANTECEDENTS,
    lambda_l1=IT2_LAMBDA_L1,
    lambda_l2=IT2_LAMBDA_L2,
    verbose=True,
):
    """
    Split data, train IT2-ANFIS, return model and splits.

    The returned model's predict() and predict_interval() methods output
    values in the *original* target units (KWh), because IT2_TSK_ANFIS
    standardises y internally and inverse-transforms on the way out.

    Returns
    -------
    model : IT2_TSK_ANFIS
    splits : tuple  (X_tr, X_val, X_test, y_tr, y_val, y_test)
        All arrays are in original (un-scaled) units.
    """
    n_rules = n_rules or DEFAULT_N_RULES
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    model = IT2_TSK_ANFIS(n_rules=n_rules, X=X_tr, verbose=verbose)
    model.fit(
        X_tr, y_tr, X_val, y_val,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        freeze_antecedents=freeze_antecedents,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        verbose=verbose,
    )
    return model, (X_tr, X_val, X_test, y_tr, y_val, y_test)
