"""A compact zero-order Sugeno ANFIS baseline.

Each fuzzy rule has a constant consequent. Rule firing strengths are
computed from Gaussian membership functions and normalized before the
weighted sum is returned. This is a self-contained baseline implementation,
not a claim of exact reproduction of any paper's implementation.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ZeroOrderANFIS:
    def __init__(self, n_rules=7, random_state=42, ridge_alpha=1e-3):
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
        # Ridge-stabilized fitting of constant consequents via weighted least squares.
        eye = np.eye(k)
        self.consequents_ = np.linalg.solve(W.T @ W + self.ridge_alpha * eye, W.T @ y)
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
        return self._weights(Z) @ self.consequents_

    def evaluate(self, X, y):
        pred = self.predict(X)
        return {
            "r2": float(r2_score(y, pred)),
            "mse": float(mean_squared_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
            "mae": float(mean_absolute_error(y, pred)),
        }
