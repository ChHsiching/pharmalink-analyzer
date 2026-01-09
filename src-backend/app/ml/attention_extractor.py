from pathlib import Path

import numpy as np
import torch

from app.ml.checkpoint_loader import load_model
from app.models.training import TrainingConfig


def extract_attention_weights(
    checkpoint_dir: Path,
    features: np.ndarray,
) -> np.ndarray:
    config = TrainingConfig.model_validate_json(
        (checkpoint_dir / "config.json").read_text()
    )
    model = load_model(checkpoint_dir, features.shape[1], config)

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
