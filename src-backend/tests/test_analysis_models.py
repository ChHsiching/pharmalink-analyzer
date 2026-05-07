# src-backend/tests/test_analysis_models.py
from app.models.analysis import (
    AttentionPair,
    AttentionMatrixResponse,
    HeatmapDataResponse,
    NetworkNode,
    NetworkEdge,
    NetworkGraphResponse,
)


def test_attention_pair_fields():
    pair = AttentionPair(
        source="comp_a", target="comp_b",
        weight=0.85, classification="synergistic",
    )
    assert pair.source == "comp_a"
    assert pair.classification == "synergistic"


def test_attention_matrix_response_structure():
    resp = AttentionMatrixResponse(
        model_id="test_ckpt",
        feature_names=["a", "b", "c"],
        matrix=[[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]],
        pairs=[
            AttentionPair(source="a", target="b", weight=0.35, classification="synergistic"),
        ],
        threshold=0.3,
    )
    assert len(resp.matrix) == 3
    assert len(resp.pairs) == 1


def test_heatmap_data_response_fields():
    resp = HeatmapDataResponse(
        model_id="ckpt1",
        feature_names=["x", "y"],
        values=[[0.6, 0.4], [0.5, 0.5]],
        min_value=0.4,
        max_value=0.6,
    )
    assert resp.min_value == 0.4
    assert resp.max_value == 0.6


def test_network_graph_response_structure():
    resp = NetworkGraphResponse(
        model_id="ckpt1",
        nodes=[NetworkNode(id="a", name="a")],
        edges=[
            NetworkEdge(source="a", target="b", weight=0.7, classification="synergistic"),
        ],
        threshold=0.5,
    )
    assert len(resp.nodes) == 1
    assert len(resp.edges) == 1
