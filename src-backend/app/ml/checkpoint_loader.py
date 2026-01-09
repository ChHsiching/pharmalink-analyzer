from pathlib import Path

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
