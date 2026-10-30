"""Random Forest regression baseline."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_rf(X_train, y_train, X_val=None, y_val=None, *, random_state=42, **kwargs):
    """Fit a Random Forest and return (model, metrics).

    Compatible with common NumPy arrays or pandas DataFrames.
    Validation data is optional. Extra keyword arguments are passed to
    RandomForestRegressor, e.g. n_estimators=300.
    """
    model = RandomForestRegressor(
        n_estimators=kwargs.pop("n_estimators", 300),
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )
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
