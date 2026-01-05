from pathlib import Path

import numpy as np
import torch

from app.ml.transformer import FeatureTransformer
from app.models.training import TrainingConfig


def extract_attention_weights(
    checkpoint_dir: Path,
    features: np.ndarray,
) -> np.ndarray:
    """Extract N x N attention weight matrix from a trained model checkpoint.

    Returns averaged attention weights across all samples: shape (n_features, n_features).
    """
    config = TrainingConfig.model_validate_json(
        (checkpoint_dir / "config.json").read_text()
    )
    model = FeatureTransformer(
        n_features=features.shape[1],
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

    with torch.no_grad():
        X_tensor = torch.tensor(features, dtype=torch.float32)
        _, attn_weights = model(X_tensor)
        avg_attn = attn_weights.mean(dim=0).numpy()

    return avg_attn


def extract_top_pairs(
    matrix: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
    threshold: float | None = None,
) -> tuple[list[dict], float]:
    """Extract top-K component pairs sorted by attention weight."""
    n = len(feature_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            weight = float((matrix[i][j] + matrix[j][i]) / 2)
            pairs.append({
                "source": feature_names[i],
                "target": feature_names[j],
                "weight": weight,
            })

    pairs.sort(key=lambda p: p["weight"], reverse=True)

    if threshold is None:
        threshold = float(np.median([p["weight"] for p in pairs]))

    for p in pairs[:top_k]:
        p["classification"] = (
            "synergistic" if p["weight"] > threshold else "antagonistic"
        )

    return pairs[:top_k], threshold
