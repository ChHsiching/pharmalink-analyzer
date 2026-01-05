# src-backend/app/services/analysis.py
import numpy as np
from fastapi import HTTPException

from app.config import CHECKPOINT_DIR
from app.ml.attention_extractor import extract_attention_weights, extract_top_pairs
from app.models.analysis import (
    AttentionMatrixResponse,
    AttentionPair,
    HeatmapDataResponse,
    NetworkEdge,
    NetworkGraphResponse,
    NetworkNode,
)
from app.services.data_loader import data_loader as _default_data_loader


class AnalysisService:
    def __init__(self, data_loader=None, checkpoint_dir=None):
        self._data_loader = data_loader or _default_data_loader
        self._checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR

    def _resolve_checkpoint(self, model_id: str):
        cp_dir = self._checkpoint_dir / model_id
        if not cp_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {model_id}")
        config_path = cp_dir / "config.json"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config not found: {model_id}")

        from app.models.training import TrainingConfig
        config = TrainingConfig.model_validate_json(config_path.read_text())
        return cp_dir, config

    def _get_features(self, dataset_id: str):
        dataset = self._data_loader.get_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        df = self._data_loader.get_dataframe(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset data not found: {dataset_id}")
        feature_names = dataset.feature_names
        X = df[feature_names].to_numpy().astype(np.float32)
        return X, feature_names

    def get_attention_matrix(self, model_id: str, top_k: int = 10, threshold: float | None = None):
        cp_dir, config = self._resolve_checkpoint(model_id)
        X, feature_names = self._get_features(config.dataset_id)
        matrix = extract_attention_weights(cp_dir, X)
        pairs_raw, actual_threshold = extract_top_pairs(matrix, feature_names, top_k, threshold)
        pairs = [AttentionPair(**p) for p in pairs_raw]
        return AttentionMatrixResponse(
            model_id=model_id,
            feature_names=feature_names,
            matrix=matrix.tolist(),
            pairs=pairs,
            threshold=actual_threshold,
        )

    def get_heatmap_data(self, model_id: str):
        cp_dir, config = self._resolve_checkpoint(model_id)
        X, feature_names = self._get_features(config.dataset_id)
        matrix = extract_attention_weights(cp_dir, X)
        return HeatmapDataResponse(
            model_id=model_id,
            feature_names=feature_names,
            values=matrix.tolist(),
            min_value=float(matrix.min()),
            max_value=float(matrix.max()),
        )

    def get_network_graph(self, model_id: str, threshold: float | None = None):
        cp_dir, config = self._resolve_checkpoint(model_id)
        X, feature_names = self._get_features(config.dataset_id)
        matrix = extract_attention_weights(cp_dir, X)
        n = len(feature_names)

        if threshold is None:
            upper = [float(matrix[i][j]) for i in range(n) for j in range(i + 1, n)]
            threshold = float(np.mean(upper))

        nodes = [NetworkNode(id=name, name=name) for name in feature_names]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                weight = float((matrix[i][j] + matrix[j][i]) / 2)
                classification = "synergistic" if weight >= threshold else "antagonistic"
                edges.append(NetworkEdge(
                    source=feature_names[i],
                    target=feature_names[j],
                    weight=weight,
                    classification=classification,
                ))

        return NetworkGraphResponse(
            model_id=model_id,
            nodes=nodes,
            edges=edges,
            threshold=threshold,
        )


analysis_service = AnalysisService()
