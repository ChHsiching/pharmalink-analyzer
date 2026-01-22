import numpy as np

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


class AnalysisService:
    def __init__(self, resolver: CheckpointResolver):
        self._resolver = resolver

    def _extract_attention(self, model_id: str) -> tuple[np.ndarray, list[str]]:
        cp_dir, config = self._resolver.resolve(model_id)
        X, feature_names = self._resolver.get_features(config.dataset_id)
        matrix = extract_attention_weights(cp_dir, X, config)
        return matrix, feature_names

    def get_attention_matrix(self, model_id: str, top_k: int = 10, threshold_percentile: int | None = None):
        matrix, feature_names = self._extract_attention(model_id)
        pairs_raw, returned_pct = extract_top_pairs(matrix, feature_names, top_k, threshold_percentile)
        pairs = [AttentionPair(**p) for p in pairs_raw]
        return AttentionMatrixResponse(
            model_id=model_id,
            feature_names=feature_names,
            matrix=matrix.tolist(),
            pairs=pairs,
            threshold=returned_pct,
        )

    def get_heatmap_data(self, model_id: str):
        matrix, feature_names = self._extract_attention(model_id)
        return HeatmapDataResponse(
            model_id=model_id,
            feature_names=feature_names,
            values=matrix.tolist(),
            min_value=float(matrix.min()),
            max_value=float(matrix.max()),
        )

    def get_network_graph(self, model_id: str, threshold_percentile: int | None = None):
        matrix, feature_names = self._extract_attention(model_id)
        n = len(feature_names)

        if threshold_percentile is None:
            threshold_percentile = 50

        pair_weights = [float((matrix[i][j] + matrix[j][i]) / 2)
                        for i in range(n) for j in range(i + 1, n)]
        abs_threshold = float(np.percentile(pair_weights, threshold_percentile))

        nodes = [NetworkNode(id=name, name=name) for name in feature_names]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                weight = float((matrix[i][j] + matrix[j][i]) / 2)
                classification = "synergistic" if weight >= abs_threshold else "antagonistic"
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
            threshold=float(threshold_percentile),
        )
