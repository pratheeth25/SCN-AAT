"""Save/load models and serialization helpers."""
from __future__ import annotations

import pickle
from pathlib import Path


def save_model(model, path):
    """Save a model (or dict of arrays) to pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """Load a model from pickle."""
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def save_it2(model, path, meta: dict | None = None):
    """Save a fitted IT2_TSK_ANFIS as a portable checkpoint (weights, scalers, constructor args, meta)."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net = model._net
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "n_rules": model.n_rules, "n_features": net.n_features,
                "antecedent_idx": None if model.antecedent_idx is None else model.antecedent_idx.tolist(),
                "x_mean": model.x_mean_, "x_scale": model.x_scale_,
                "y_mean": model.y_mean_, "y_scale": model.y_scale_, "meta": meta or {}}, path)


def load_it2(path, device: str = "cpu"):
    """Rebuild an IT2_TSK_ANFIS saved by ``save_it2``. Returns (model, meta)."""
    import numpy as np
    import torch
    from ..models.it2_anfis import IT2_TSK_ANFIS
    ck = torch.load(Path(path), map_location=device, weights_only=False)
    m = IT2_TSK_ANFIS(ck["n_rules"], X=np.zeros((2, ck["n_features"])), verbose=False,
                      antecedent_idx=ck["antecedent_idx"], device=device)
    m._net.load_state_dict(ck["state_dict"])
    m.x_mean_, m.x_scale_ = np.asarray(ck["x_mean"]), np.asarray(ck["x_scale"])
    m.y_mean_, m.y_scale_ = float(ck["y_mean"]), float(ck["y_scale"])
    return m, ck["meta"]
