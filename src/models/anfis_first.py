"""A compact first-order Sugeno ANFIS baseline.

Gaussian rule firing strengths are combined with local linear consequents.
Parameters are estimated with ridge regression on normalized rule-weighted
inputs. This is a practical baseline, not an exact reproduction of a paper.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class FirstOrderANFIS:
    def __init__(self, n_rules=7, random_state=42, ridge_alpha=1.0):
        self.n_rules = int(n_rules)
        self.random_state = random_state
        self.ridge_alpha = float(ridge_alpha)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and have the same number of rows as y")
        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0
        Z = (X - self.x_mean_) / self.x_scale_
        k = max(1, min(self.n_rules, len(Z)))
        self.kmeans_ = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(Z)
        self.centers_ = self.kmeans_.cluster_centers_
        distances = np.linalg.norm(Z[:, None, :] - self.centers_[None, :, :], axis=2)
        self.sigma_ = np.maximum(np.median(distances, axis=0), 0.25)
        W = self._weights(Z)
        # Local linear consequents: one weighted linear model per rule.
        self.models_ = []
        for j in range(k):
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(Z, y, sample_weight=np.maximum(W[:, j], 1e-8))
            self.models_.append(model)
        return self

    def _weights(self, Z):
        dist2 = ((Z[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        logw = -0.5 * dist2 / (self.sigma_[None, :] ** 2)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Z = (X - self.x_mean_) / self.x_scale_
        W = self._weights(Z)
        local_preds = np.column_stack([m.predict(Z) for m in self.models_])
        return np.sum(W * local_preds, axis=1)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return {
            "r2": float(r2_score(y, pred)),
            "mse": float(mean_squared_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
            "mae": float(mean_absolute_error(y, pred)),
        }
