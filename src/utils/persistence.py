"""Save/load models and serialization helpers."""

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
