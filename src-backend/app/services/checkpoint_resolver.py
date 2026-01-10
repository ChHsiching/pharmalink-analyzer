import json

import numpy as np

from app.exceptions import CheckpointNotFoundError, DatasetNotFoundError
from app.models.training import CheckpointInfo, TrainingConfig


class CheckpointResolver:
    def __init__(self, checkpoint_dir, data_loader):
        self._checkpoint_dir = checkpoint_dir
        self._data_loader = data_loader

    def resolve(self, model_id: str) -> tuple:
        cp_dir = self._checkpoint_dir / model_id
        if not cp_dir.is_dir():
            raise CheckpointNotFoundError(model_id)
        config_path = cp_dir / "config.json"
        if not config_path.exists():
            raise CheckpointNotFoundError(model_id)
        config = TrainingConfig.model_validate_json(config_path.read_text())
        return cp_dir, config

    def get_features(self, dataset_id: str) -> tuple[np.ndarray, list[str]]:
        dataset = self._data_loader.get_dataset(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        df = self._data_loader.get_dataframe(dataset_id)
        if df is None:
            raise DatasetNotFoundError(dataset_id)
        feature_names = dataset.feature_names
        X = df[feature_names].to_numpy().astype(np.float32)
        return X, feature_names

    def get_features_with_target(self, dataset_id: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        dataset = self._data_loader.get_dataset(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        df = self._data_loader.get_dataframe(dataset_id)
        if df is None:
            raise DatasetNotFoundError(dataset_id)
        feature_names = dataset.feature_names
        X = df[feature_names].to_numpy().astype(np.float32)
        y = df[dataset.target].to_numpy().astype(np.float32)
        return X, y, feature_names

    def list_checkpoints(self) -> list[CheckpointInfo]:
        if not self._checkpoint_dir.exists():
            return []
        result = []
        for d in sorted(self._checkpoint_dir.iterdir()):
            if d.is_dir() and (d / "config.json").exists():
                config = TrainingConfig.model_validate_json((d / "config.json").read_text())
                final_loss = 0.0
                metrics_path = d / "metrics.json"
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text())
                    final_loss = metrics.get("final_val_loss", 0.0)
                result.append(CheckpointInfo(
                    id=d.name,
                    dataset_id=config.dataset_id,
                    config=config,
                    final_loss=final_loss,
                ))
        return result
