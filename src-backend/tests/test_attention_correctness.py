"""Correctness tests for attention extraction: extract_top_pairs behavior."""

import numpy as np

from app.ml.attention_extractor import extract_top_pairs


def test_returns_correct_count():
    matrix = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.5], [0.1, 0.5, 0.0]])
    pairs, _ = extract_top_pairs(matrix, ["A", "B", "C"], top_k=2)
    assert len(pairs) == 2


def test_sorted_by_weight():
    matrix = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.5], [0.1, 0.5, 0.0]])
    pairs, _ = extract_top_pairs(matrix, ["A", "B", "C"], top_k=3)
    weights = [p["weight"] for p in pairs]
    assert weights == sorted(weights, reverse=True)


def test_classification_by_percentile():
    matrix = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.5], [0.1, 0.5, 0.0]])
    pairs, _ = extract_top_pairs(matrix, ["A", "B", "C"], top_k=3)
    classifications = [p["classification"] for p in pairs]
    assert classifications[0] == "synergistic"
    assert classifications[-1] == "antagonistic"


def test_symmetric_averaging():
    matrix = np.array([[0.0, 0.8], [0.4, 0.0]])
    pairs, _ = extract_top_pairs(matrix, ["A", "B"], top_k=1)
    assert abs(pairs[0]["weight"] - 0.6) < 1e-10


def test_custom_percentile():
    matrix = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.5], [0.1, 0.5, 0.0]])
    pairs, _ = extract_top_pairs(matrix, ["A", "B", "C"], top_k=3, threshold_percentile=90)
    synergistic_count = sum(1 for p in pairs if p["classification"] == "synergistic")
    assert synergistic_count <= 1
