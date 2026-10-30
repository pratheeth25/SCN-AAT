"""Interval Type-2 TSK ANFIS for regression and uncertainty quantification.

This is a reconstructed implementation following the mathematical formulation
described in arXiv:2601.18897 (Khaled et al., 2026). It is NOT a verbatim copy
of the authors' private source code and may differ in engineering details.

Reconstructed mathematical formulation
---------------------------------------
Each rule j = 1 … R has an interval type-2 Gaussian antecedent for each
feature f = 1 … F, defined by an *uncertain mean interval* [c1_jf, c2_jf] and
a shared standard deviation sigma_jf.

Lower membership function (LMF):
    mu_L(x_f) = exp(-0.5 * ((x_f - c2_jf) / sigma_jf)^2)  if x_f <= c_mid
                exp(-0.5 * ((x_f - c1_jf) / sigma_jf)^2)  if x_f >  c_mid
    where c_mid = (c1_jf + c2_jf) / 2.

Upper membership function (UMF):
    mu_U(x_f) = exp(-0.5 * ((x_f - c1_jf) / sigma_jf)^2)  if x_f < c1_jf
                1.0                                          if c1_jf <= x_f <= c2_jf
                exp(-0.5 * ((x_f - c2_jf) / sigma_jf)^2)  if x_f > c2_jf

Rule firing strengths (product t-norm, computed in log-space for stability):
    mu_j_L(x) = prod_f mu_L_jf(x_f)
    mu_j_U(x) = prod_f mu_U_jf(x_f)

Normalised firing strengths:
    f_bar_L_j = mu_j_L / sum_k mu_k_L
    f_bar_U_j = mu_j_U / sum_k mu_k_U

First-order TSK consequent for rule j:
    y_j = w_j^T x + b_j

Type-reduced outputs:
    y_lower = sum_j f_bar_L_j * y_j
    y_upper = sum_j f_bar_U_j * y_j

Defuzzified point prediction (weighted average with learned q in (0,1)):
    y_hat = q * y_lower + (1 - q) * y_upper

Target standardisation
-----------------------
The raw target (energy in KWh, ~275 000 mean) has a scale that makes gradient
descent on raw values very sensitive to initialisation. We standardise y to
zero-mean / unit-variance during training and inverse-transform for prediction,
so all metrics are reported in the original KWh units.

Public interface
----------------
    model = IT2_TSK_ANFIS(n_rules=7, X=X_train)
    model.fit(X_train, y_train, X_val, y_val, epochs=100, patience=30)
    y_pred = model.predict(X_test)           # KWh (original units)
    lo, hi = model.predict_interval(X_test)  # KWh (original units)
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal PyTorch network
# ---------------------------------------------------------------------------

class _IT2TSKNetwork(nn.Module):
    """
    Vectorised IT2-TSK network.

    All computations are done on normalised inputs Z (zero-mean/unit-variance)
    and normalised targets yn.  The wrapper class inverts the target
    standardisation at prediction time.

    Parameters
    ----------
    c1 : Tensor  shape (R, F)  lower uncertain mean bounds (normalised input space)
    c2 : Tensor  shape (R, F)  upper uncertain mean bounds
    log_sigma : Tensor  shape (R, F)  log of Gaussian spread (> 0 after exp)
    q : float  initial defuzzification weight in (0, 1)
    """

    def __init__(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        log_sigma: torch.Tensor,
        q: float = 0.5,
        n_inputs: int | None = None,
        ant_idx: np.ndarray | None = None,
    ):
        super().__init__()
        R, F_ant = c1.shape
        # Consequents see all inputs; antecedents only see the columns in ant_idx.
        F = F_ant if n_inputs is None else n_inputs
        self.n_rules = R
        self.n_features = F
        idx = np.arange(F) if ant_idx is None else np.asarray(ant_idx, dtype=np.int64)
        self.register_buffer("ant_idx", torch.tensor(idx, dtype=torch.long))

        # Antecedent parameters
        self.c1 = nn.Parameter(c1.clone())
        self.c2 = nn.Parameter(c2.clone())
        self.log_sigma = nn.Parameter(log_sigma.clone())

        # First-order TSK consequent: [bias | w_1 … w_F] per rule
        self.consequents = nn.Parameter(torch.zeros(R, F + 1))
        nn.init.normal_(self.consequents, mean=0.0, std=0.01)

        # Learnable defuzzification weight q (sigmoid-transformed to stay in (0,1))
        # Initialise logit such that sigmoid(logit) ~ q
        logit_q = float(np.log(q / (1.0 - q + 1e-9)))
        self.logit_q = nn.Parameter(torch.tensor(logit_q))

    # ------------------------------------------------------------------
    def _compute_memberships(self, Z: torch.Tensor):
        """
        Compute lower and upper IT2 Gaussian membership functions.

        Z : (N, F) normalised input
        Returns: mu_L (N, R), mu_U (N, R)  — products over features
        """
        sigma = torch.exp(self.log_sigma).clamp(min=0.03, max=50.0)  # (R, F_ant)
        Z = Z[:, self.ant_idx]

        # Broadcast: Z -> (N, 1, F), params -> (1, R, F)
        x   = Z.unsqueeze(1)                    # (N, 1, F)
        c1  = self.c1.unsqueeze(0)              # (1, R, F)
        c2  = self.c2.unsqueeze(0)              # (1, R, F)
        sig = sigma.unsqueeze(0)                # (1, R, F)

        # Enforce ordering: c1 <= c2 via soft abs
        c1_ord = torch.minimum(c1, c2)
        c2_ord = torch.maximum(c1, c2)
        c_mid  = 0.5 * (c1_ord + c2_ord)       # (1, R, F)

        # --- Lower membership (LMF) ---
        # Uses the "inner" Gaussian centred at the far bound
        left  = torch.exp(-0.5 * ((x - c2_ord) / sig) ** 2)
        right = torch.exp(-0.5 * ((x - c1_ord) / sig) ** 2)
        mu_l_per_feat = torch.where(x <= c_mid, left, right)   # (N, R, F)

        # --- Upper membership (UMF) ---
        # = 1.0 inside [c1, c2]; Gaussian tail outside
        left_u  = torch.exp(-0.5 * ((x - c1_ord) / sig) ** 2)
        right_u = torch.exp(-0.5 * ((x - c2_ord) / sig) ** 2)
        mu_u_per_feat = torch.where(
            x < c1_ord,
            left_u,
            torch.where(x <= c2_ord, torch.ones_like(x), right_u),
        )                                                        # (N, R, F)

        # Product t-norm in log-space for numerical stability
        log_mu_l = torch.log(mu_l_per_feat.clamp(min=1e-30)).sum(dim=2)  # (N, R)
        log_mu_u = torch.log(mu_u_per_feat.clamp(min=1e-30)).sum(dim=2)  # (N, R)

        # Subtract row-wise max before exp to prevent underflow
        log_mu_l = log_mu_l - log_mu_l.max(dim=1, keepdim=True).values
        log_mu_u = log_mu_u - log_mu_u.max(dim=1, keepdim=True).values

        mu_L = torch.exp(log_mu_l).clamp(min=1e-12)  # (N, R)
        mu_U = torch.exp(log_mu_u).clamp(min=1e-12)  # (N, R)
        return mu_L, mu_U

    # ------------------------------------------------------------------
    def firing_weights(self, Z: torch.Tensor):
        """Normalised firing strengths; returns (f_bar_L, f_bar_U) each (N, R)."""
        mu_L, mu_U = self._compute_memberships(Z)
        f_L = mu_L / mu_L.sum(dim=1, keepdim=True).clamp(min=1e-12)
        f_U = mu_U / mu_U.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return f_L, f_U

    # ------------------------------------------------------------------
    def forward(self, Z: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        Z : (N, F) normalised inputs

        Returns
        -------
        y_hat   : (N, 1) defuzzified prediction (normalised target space)
        y_lower : (N, 1) lower type-reduced estimate
        y_upper : (N, 1) upper type-reduced estimate
        """
        f_L, f_U = self.firing_weights(Z)                # (N, R) each

        # First-order TSK consequent outputs
        design = torch.cat(
            [torch.ones(Z.shape[0], 1, device=Z.device), Z], dim=1
        )                                                # (N, F+1)
        rule_out = design @ self.consequents.T           # (N, R)

        y_lower = (f_L * rule_out).sum(dim=1, keepdim=True)  # (N, 1)
        y_upper = (f_U * rule_out).sum(dim=1, keepdim=True)  # (N, 1)

        q = torch.sigmoid(self.logit_q)
        y_hat = q * y_lower + (1.0 - q) * y_upper

        return y_hat, y_lower, y_upper


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class IT2_TSK_ANFIS:
    """
    Interval Type-2 Takagi-Sugeno-Kang ANFIS for regression.

    Targets are standardised internally before training and inverse-transformed
    at prediction time, so all public-facing outputs are in the original units.

    Parameters
    ----------
    n_rules : int
        Number of fuzzy rules.
    X : ndarray, optional
        Training data used for initialisation (centres, sigmas, input scaler).
    verbose : bool
        Print epoch summaries.
    q : float
        Initial defuzzification weight in (0, 1).
    random_state : int
        Seed for reproducible initialisation.
    """

    def __init__(
        self,
        n_rules: int = 7,
        X: np.ndarray | None = None,
        verbose: bool = True,
        q: float = 0.5,
        random_state: int = 42,
        init: str = "lhs",
        sigma_scale: float | None = None,
        antecedent_idx=None,
        device: str | None = None,
    ):
        self.antecedent_idx = None if antecedent_idx is None else np.asarray(antecedent_idx, dtype=int)
        self.init = init
        self.sigma_scale = sigma_scale
        self.n_rules = int(n_rules)
        self.verbose = bool(verbose)
        self.q = float(q)
        self.random_state = int(random_state)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._net: _IT2TSKNetwork | None = None
        self.history_: dict = {"train_mse": [], "val_mse": []}
        # Input scaler (fitted on training data)
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        # Target scaler (fitted on training y)
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0
        if X is not None:
            self._init_input_scaler(X)
            self._build_network(X)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_input_scaler(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.x_mean_  = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-8] = 1.0

    def _build_network(self, X: np.ndarray):
        """Initialise antecedent parameters using Latin-Hypercube-style stratified sampling."""
        X = np.asarray(X, dtype=np.float64)
        Z_all = self._normalize_X(X)
        self.n_features_in_ = Z_all.shape[1]
        ai = np.arange(Z_all.shape[1]) if self.antecedent_idx is None else self.antecedent_idx
        Z = Z_all[:, ai]
        R = self.n_rules
        F = Z.shape[1]

        rng = np.random.default_rng(self.random_state)

        # Stratified (LHS-like) centre initialisation
        x_min = Z.min(axis=0)
        x_max = Z.max(axis=0)
        x_rng = x_max - x_min
        x_rng[x_rng < 1e-8] = 1.0

        centers = np.zeros((R, F))
        for r in range(R):
            for f in range(F):
                seg_start = x_min[f] + r * x_rng[f] / R
                seg_end   = x_min[f] + (r + 1) * x_rng[f] / R
                centers[r, f] = rng.uniform(seg_start, seg_end)

        # Partition width used for FOU
        partition_w = x_rng / R
        alpha = 0.2  # FOU half-width as fraction of partition width

        c1 = centers - alpha * partition_w[None, :] / 2.0
        c2 = centers + alpha * partition_w[None, :] / 2.0

        # sigma = half a partition width, at least 0.05
        sigma = np.clip(partition_w[None, :] * 0.5, 0.05, None).repeat(R, axis=0)

        if self.init == "kmeans":
            # Data-driven centres (deviation from the paper's LHS): rules sit where
            # the training data actually lies, so firing is not one-hot in high-D.
            from sklearn.cluster import KMeans
            centers = KMeans(R, n_init=5, random_state=self.random_state).fit(Z).cluster_centers_
            c1 = centers - alpha * partition_w[None, :] / 2.0
            c2 = centers + alpha * partition_w[None, :] / 2.0
            sigma = np.full((R, F), 1.0)
        if self.sigma_scale is not None:
            sigma = sigma * float(self.sigma_scale)

        self._net = _IT2TSKNetwork(
            c1=torch.tensor(c1,  dtype=torch.float32),
            c2=torch.tensor(c2,  dtype=torch.float32),
            log_sigma=torch.tensor(np.log(sigma), dtype=torch.float32),
            q=self.q,
            n_inputs=self.n_features_in_,
            ant_idx=ai,
        ).to(self.device)

    # ------------------------------------------------------------------
    # Scaling helpers
    # ------------------------------------------------------------------

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        return (X.astype(np.float64) - self.x_mean_) / self.x_scale_

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y.astype(np.float64) - self.y_mean_) / self.y_scale_

    def _denormalize_y(self, yn: np.ndarray) -> np.ndarray:
        return yn * self.y_scale_ + self.y_mean_

    def _to_tensor_Z(self, X: np.ndarray) -> torch.Tensor:
        Z = self._normalize_X(X)
        return torch.tensor(Z, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs: int = 100,
        patience: int = 30,
        batch_size: int = 64,
        freeze_antecedents: bool = True,
        lambda_l1: float = 0.005,
        lambda_l2: float = 0.01,
        verbose=None,
        eta_cons: float = 0.01,
        eta_ant: float = 0.001,
        ridge_lambda: float | None = None,
        **kwargs,
    ) -> "IT2_TSK_ANFIS":
        """
        Train the model.

        Targets are standardised inside this method; the network is trained on
        zero-mean/unit-variance y.  All external outputs (predict, etc.) are in
        original units.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        assert X.ndim == 2 and len(X) == len(y), "X must be 2-D and len(X)==len(y)"
        show = self.verbose if verbose is None else bool(verbose)

        # Fit/re-fit input scaler if not done during __init__
        if self.x_mean_ is None:
            self._init_input_scaler(X)
            self._build_network(X)

        assert X.shape[1] == self.n_features_in_, (
            f"Feature count mismatch: expected {self.n_features_in_}, got {X.shape[1]}"
        )

        # Fit target scaler on training y ONLY
        self.y_mean_  = float(y.mean())
        self.y_scale_ = float(y.std())
        if self.y_scale_ < 1e-8:
            self.y_scale_ = 1.0

        yn  = self._normalize_y(y)
        Zt  = self._to_tensor_Z(X)
        ynt = torch.tensor(yn, dtype=torch.float32, device=self.device).unsqueeze(1)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64).ravel()
            Zv    = self._to_tensor_Z(X_val)
            ynv   = torch.tensor(
                self._normalize_y(y_val), dtype=torch.float32, device=self.device
            ).unsqueeze(1)

        # Optimiser groups
        antecedent_params = [self._net.c1, self._net.c2, self._net.log_sigma]
        consequent_params  = [self._net.consequents, self._net.logit_q]
        for p in antecedent_params:
            p.requires_grad_(not freeze_antecedents)

        optim_groups = [{"params": consequent_params, "lr": eta_cons}]
        if not freeze_antecedents:
            optim_groups.append({"params": antecedent_params, "lr": eta_ant})
        optimizer = torch.optim.Adam(optim_groups, weight_decay=0.0)

        best_val_loss = float("inf")
        if ridge_lambda is not None:
            self._ridge_init_consequents(Zt, ynt, float(ridge_lambda))
            if has_val:
                with torch.no_grad():
                    best_val_loss = float(torch.mean((self._net(Zv)[0] - ynv) ** 2).cpu())
        best_state    = copy.deepcopy(self._net.state_dict())
        best_epoch    = 0
        stale         = 0
        self.history_ = {"train_mse": [], "val_mse": []}
        N  = len(X)
        bs = max(1, int(batch_size))

        for epoch in range(1, int(epochs) + 1):
            self._net.train()
            perm = torch.randperm(N, device=self.device)
            for idx in perm.split(bs):
                zb, yb = Zt[idx], ynt[idx]
                optimizer.zero_grad(set_to_none=True)
                pred, _, _ = self._net(zb)
                loss = torch.mean((pred - yb) ** 2)
                # Regularisation on consequent weights only
                w = self._net.consequents
                loss = loss + float(lambda_l1) * w.abs().mean()
                loss = loss + float(lambda_l2) * w.pow(2).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=5.0)
                optimizer.step()
                # Enforce c1 <= c2 after each update
                with torch.no_grad():
                    self._net.c1.data, self._net.c2.data = (
                        torch.minimum(self._net.c1, self._net.c2),
                        torch.maximum(self._net.c1, self._net.c2),
                    )

            # Evaluation
            self._net.eval()
            with torch.no_grad():
                tr_pred, _, _ = self._net(Zt)
                tr_mse = float(torch.mean((tr_pred - ynt) ** 2).cpu())
                if has_val:
                    vl_pred, _, _ = self._net(Zv)
                    vl_mse = float(torch.mean((vl_pred - ynv) ** 2).cpu())
                else:
                    vl_mse = tr_mse

            self.history_["train_mse"].append(tr_mse)
            self.history_["val_mse"].append(vl_mse)

            if show and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                print(
                    f"Epoch {epoch:3d} | Train MSE: {tr_mse:.6f} | Val MSE: {vl_mse:.6f}"
                )

            # Check finite
            if not (np.isfinite(tr_mse) and np.isfinite(vl_mse)):
                print(f"[WARNING] Non-finite loss at epoch {epoch}. Stopping.")
                break

            if vl_mse < best_val_loss - 1e-8:
                best_val_loss = vl_mse
                best_state    = copy.deepcopy(self._net.state_dict())
                best_epoch    = epoch
                stale         = 0
            else:
                stale += 1

            if stale >= int(patience):
                if show:
                    print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

        self._net.load_state_dict(best_state)
        self.best_epoch_   = best_epoch
        self.best_val_mse_ = best_val_loss
        self._net.eval()
        return self

    def get_antecedents(self) -> dict:
        """Current antecedent parameters (normalised input space) as numpy arrays."""
        net = self._net
        with torch.no_grad():
            c1 = torch.minimum(net.c1, net.c2).cpu().numpy().astype(np.float64)
            c2 = torch.maximum(net.c1, net.c2).cpu().numpy().astype(np.float64)
            sigma = torch.exp(net.log_sigma).cpu().numpy().astype(np.float64)
        return {"c1": c1, "c2": c2, "sigma": sigma}

    def set_antecedents(self, c1, c2, sigma) -> None:
        """Overwrite antecedents; validates shape, c1 <= c2 and sigma > 0."""
        net = self._net
        c1, c2, sigma = (np.asarray(a, dtype=np.float64) for a in (c1, c2, sigma))
        if not (c1.shape == c2.shape == sigma.shape == tuple(net.c1.shape)):
            raise ValueError(f"antecedent shape mismatch, expected {tuple(net.c1.shape)}")
        if np.any(c1 > c2) or np.any(sigma <= 0) or not np.all(np.isfinite(np.r_[c1.ravel(), c2.ravel(), sigma.ravel()])):
            raise ValueError("invalid antecedents: need finite values, c1 <= c2 and sigma > 0")
        with torch.no_grad():
            net.c1.copy_(torch.tensor(c1, dtype=torch.float32))
            net.c2.copy_(torch.tensor(c2, dtype=torch.float32))
            net.log_sigma.copy_(torch.tensor(np.log(sigma), dtype=torch.float32))

    def _ridge_init_consequents(self, Zt, ynt, lam):
        """Closed-form ridge solution for the TSK consequents (classic ANFIS hybrid step).

        With the antecedents fixed, y_hat is linear in the consequents:
        y_hat = sum_j phi_j(x) * [1, x] @ theta_j, phi_j = q*fL_j + (1-q)*fU_j
        (the same phi as Eq. 16 of the paper). All terms, biases included, are penalised.
        """
        net = self._net
        with torch.no_grad():
            f_L, f_U = net.firing_weights(Zt)
            q = torch.sigmoid(net.logit_q)
            phi = q * f_L + (1.0 - q) * f_U                              # (N, R)
            design = torch.cat([torch.ones(Zt.shape[0], 1, device=Zt.device), Zt], dim=1)
            D = (phi.unsqueeze(2) * design.unsqueeze(1)).reshape(Zt.shape[0], -1).double()
            P = torch.eye(D.shape[1], dtype=torch.float64, device=D.device) * lam
            # Biases are penalised too: a rule with ~zero firing on every training row
            # has an all-zero design column, so an unpenalised bias is unidentifiable and
            # can explode; it then dominates any test point whose (max-shifted) firing
            # favours that rule. With standardised y this shrinks such rules to the mean.
            target = ynt.double()
            theta = torch.linalg.solve(D.T @ D + P + 1e-8 * torch.eye(D.shape[1], dtype=torch.float64, device=D.device), D.T @ target)
            net.consequents.copy_(theta.reshape(net.n_rules, -1).float())

    def predict(self, X) -> np.ndarray:
        """Point predictions in original target units."""
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")
        self._net.eval()
        with torch.no_grad():
            yn_pred, _, _ = self._net(self._to_tensor_Z(np.asarray(X, dtype=np.float64)))
        return self._denormalize_y(yn_pred.cpu().numpy().ravel())

    def predict_interval(self, X):
        """
        Prediction interval in original target units.

        Returns
        -------
        lower : ndarray  lower bounds
        upper : ndarray  upper bounds
        """
        if self._net is None:
            raise RuntimeError("Call fit() before predict_interval().")
        self._net.eval()
        with torch.no_grad():
            _, yl, yu = self._net(self._to_tensor_Z(np.asarray(X, dtype=np.float64)))
        lo = self._denormalize_y(yl.cpu().numpy().ravel())
        hi = self._denormalize_y(yu.cpu().numpy().ravel())
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def explain_prediction(self, X_input) -> dict:
        """Per-rule firing strengths and output contributions for a single sample."""
        x = np.asarray(X_input, dtype=np.float64).reshape(1, -1)
        self._net.eval()
        with torch.no_grad():
            Z = self._to_tensor_Z(x)
            pred, lo, hi = self._net(Z)
            f_L, f_U     = self._net.firing_weights(Z)
        return {
            "prediction":       float(self._denormalize_y(pred.cpu().numpy().ravel())[0]),
            "lower_output":     float(self._denormalize_y(lo.cpu().numpy().ravel())[0]),
            "upper_output":     float(self._denormalize_y(hi.cpu().numpy().ravel())[0]),
            "interval_width":   float(abs(hi.item() - lo.item()) * self.y_scale_),
            "rule_weights_lower": f_L.cpu().numpy().ravel(),
            "rule_weights_upper": f_U.cpu().numpy().ravel(),
        }


# Backwards-compatibility aliases for older callers that import the names
ANFISFirst = None  # populated in anfis_first.py
ANFISZero  = None  # populated in anfis_zero.py
