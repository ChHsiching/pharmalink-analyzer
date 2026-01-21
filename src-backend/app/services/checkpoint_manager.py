"""CheckpointManager — checkpoint persistence extracted from TrainingService."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler

from app.ml.checkpoint_loader import save_scaler_params
from app.models.training import TrainingConfig, TrainingProgress


class CheckpointManager:
    """Handles checkpoint serialization: model weights, config, scaler, metrics."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self._checkpoint_dir = checkpoint_dir

    def save(
        self,
        task_id: str,
        config: TrainingConfig,
        model_state: dict[str, torch.Tensor],
        val_loss: float,
        scaler: StandardScaler,
        progress: list[TrainingProgress],
    ) -> Path:
        """Save a training checkpoint to disk. Returns the checkpoint directory path."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_id = f"{config.dataset_id}_{task_id}"
        ckpt_path = self._checkpoint_dir / ckpt_id
        ckpt_path.mkdir(exist_ok=True)

        torch.save(model_state, ckpt_path / "model.pt")
        (ckpt_path / "config.json").write_text(config.model_dump_json())
        save_scaler_params(
            ckpt_path / "scaler_params.json", scaler.mean_, scaler.scale_,
        )

        metrics_data: dict = {"final_val_loss": val_loss}
        if progress:
            metrics_data["history"] = [p.model_dump() for p in progress]
        (ckpt_path / "metrics.json").write_text(json.dumps(metrics_data))

        return ckpt_path
