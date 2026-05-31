import json
from pathlib import Path

import numpy as np
import torch

from app.ml.transformer import FeatureTransformer
from app.models.training import TrainingConfig


def load_model(
    checkpoint_dir: Path,
    n_features: int,
    config: TrainingConfig,
) -> FeatureTransformer:
    model = FeatureTransformer(
        n_features=n_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )
    state_dict = torch.load(
        checkpoint_dir / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_scaler_params(path: Path, mean: np.ndarray, scale: np.ndarray) -> None:
    """Persist StandardScaler mean_ and scale_ to JSON."""
    data = {
        "mean": np.asarray(mean).astype(np.float64).tolist(),
        "scale": np.asarray(scale).astype(np.float64).tolist(),
    }
    path.write_text(json.dumps(data))


def load_scaler_params(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load StandardScaler mean_ and scale_ from JSON.

    Returns (mean, scale) as float32 numpy arrays.
    """
    data = json.loads(path.read_text())
    return (
        np.array(data["mean"], dtype=np.float32),
        np.array(data["scale"], dtype=np.float32),
    )


def load_fold_model(
    checkpoint_dir: Path,
    n_features: int,
    config: TrainingConfig,
    fold_idx: int,
) -> FeatureTransformer:
    path = checkpoint_dir / f"model_fold{fold_idx}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Fold {fold_idx} model not found: {path}")
    model = FeatureTransformer(
        n_features=n_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_fold_scaler(
    checkpoint_dir: Path,
    fold_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = checkpoint_dir / f"scaler_fold{fold_idx}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fold {fold_idx} scaler not found: {path}")
    return load_scaler_params(path)
