import numpy as np
from fastapi import HTTPException

from app.models.training import TrainingConfig


class CheckpointResolver:
    def __init__(self, checkpoint_dir, data_loader):
        self._checkpoint_dir = checkpoint_dir
        self._data_loader = data_loader

    def resolve(self, model_id: str) -> tuple:
        cp_dir = self._checkpoint_dir / model_id
        if not cp_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {model_id}")
        config_path = cp_dir / "config.json"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config not found: {model_id}")
        config = TrainingConfig.model_validate_json(config_path.read_text())
        return cp_dir, config

    def get_features(self, dataset_id: str) -> tuple[np.ndarray, list[str]]:
        dataset = self._data_loader.get_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        df = self._data_loader.get_dataframe(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset data not found: {dataset_id}")
        feature_names = dataset.feature_names
        X = df[feature_names].to_numpy().astype(np.float32)
        return X, feature_names

    def get_features_with_target(self, dataset_id: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        dataset = self._data_loader.get_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        df = self._data_loader.get_dataframe(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset data not found: {dataset_id}")
        feature_names = dataset.feature_names
        X = df[feature_names].to_numpy().astype(np.float32)
        y = df[dataset.target].to_numpy().astype(np.float32)
        return X, y, feature_names
