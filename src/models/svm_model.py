"""Support Vector Regression baseline."""

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_svm(X_train, y_train, X_val=None, y_val=None, *, kernel="rbf", tune=False, cv=5, n_jobs=-1, **kwargs):
    """Fit scaled SVR and return (model, metrics). Validation data is optional.

    The target is standardised (SVR's C / epsilon assume a unit-scale y; on raw kWh
    values with C=1 the model collapses to a constant), and predictions are
    inverse-transformed automatically by TransformedTargetRegressor.
    With tune=True, C / gamma / epsilon are chosen by CV on the training data
    (``cv``: 5 folds by default; pass a TimeSeriesSplit for time-ordered rows).
    """
    base = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(kernel=kernel, **kwargs)),
        transformer=StandardScaler(),
    )
    if tune:
        grid = {
            "regressor__svr__C": [1, 3, 10, 30],
            "regressor__svr__gamma": [0.01, 0.03, 0.1],
            "regressor__svr__epsilon": [0.1, 0.3],
        }
        model = GridSearchCV(base, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=n_jobs)
    else:
        model = base
    model.fit(X_train, y_train)
    metrics = {}
    if X_val is not None and y_val is not None:
        pred = model.predict(X_val)
        metrics = {
            "r2": float(r2_score(y_val, pred)),
            "mse": float(mean_squared_error(y_val, pred)),
            "rmse": float(mean_squared_error(y_val, pred) ** 0.5),
            "mae": float(mean_absolute_error(y_val, pred)),
        }
    return model, metrics
