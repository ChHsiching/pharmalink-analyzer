# src-backend/app/services/analysis.py
import numpy as np

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
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.data_loader import data_loader as _default_data_loader


class AnalysisService:
    def __init__(self, resolver: CheckpointResolver):
        self._resolver = resolver

    def get_attention_matrix(self, model_id: str, top_k: int = 10, threshold: float | None = None):
        cp_dir, config = self._resolver.resolve(model_id)
        X, feature_names = self._resolver.get_features(config.dataset_id)
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
        cp_dir, config = self._resolver.resolve(model_id)
        X, feature_names = self._resolver.get_features(config.dataset_id)
        matrix = extract_attention_weights(cp_dir, X)
        return HeatmapDataResponse(
            model_id=model_id,
            feature_names=feature_names,
            values=matrix.tolist(),
            min_value=float(matrix.min()),
            max_value=float(matrix.max()),
        )

    def get_network_graph(self, model_id: str, threshold: float | None = None):
        cp_dir, config = self._resolver.resolve(model_id)
        X, feature_names = self._resolver.get_features(config.dataset_id)
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
