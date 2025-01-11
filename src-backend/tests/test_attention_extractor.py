import numpy as np
import pytest

from app.ml.attention_extractor import extract_attention_weights, extract_top_pairs


@pytest.fixture
def trained_checkpoint(make_checkpoint):
    env = make_checkpoint()
    return env["cp_dir"], env["X"], env["feature_names"], env["config"]


def test_extract_attention_weights_shape(trained_checkpoint):
    cp_dir, X, feature_names, config = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X, config)
    assert matrix.shape == (len(feature_names), len(feature_names))
    assert isinstance(matrix, np.ndarray)


def test_extract_attention_weights_positive(trained_checkpoint):
    cp_dir, X, _, config = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X, config)
    assert (matrix >= 0).all()


def test_extract_attention_weights_row_sums(trained_checkpoint):
    """Head-averaged softmax weights are approximately but not exactly 1.0."""
    cp_dir, X, _, config = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X, config)
    row_sums = matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=0.2)


def test_extract_top_pairs_count(trained_checkpoint):
    cp_dir, X, feature_names, config = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X, config)
    pairs, _ = extract_top_pairs(matrix, feature_names, top_k=3)
    assert len(pairs) == 3


def test_extract_top_pairs_sorted_descending(trained_checkpoint):
    cp_dir, X, feature_names, config = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X, config)
    pairs, _ = extract_top_pairs(matrix, feature_names, top_k=5)
    weights = [p["weight"] for p in pairs]
    assert weights == sorted(weights, reverse=True)


def test_extract_top_pairs_default_percentile_is_50(trained_checkpoint):
    cp_dir, X, feature_names, config = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X, config)
    _, returned_pct = extract_top_pairs(matrix, feature_names, top_k=3)
    assert returned_pct == 50.0


def test_extract_top_pairs_percentile_0_all_synergistic(trained_checkpoint):
    cp_dir, X, feature_names, config = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X, config)
    pairs, pct = extract_top_pairs(matrix, feature_names, top_k=3, threshold_percentile=0)
    assert pct == 0.0
    assert all(p["classification"] == "synergistic" for p in pairs)


def test_extract_top_pairs_percentile_100_mostly_antagonistic(trained_checkpoint):
    cp_dir, X, feature_names, config = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X, config)
    pairs, pct = extract_top_pairs(matrix, feature_names, top_k=3, threshold_percentile=100)
    assert pct == 100.0
    synergistic = [p for p in pairs if p["classification"] == "synergistic"]
    assert len(synergistic) <= 1


def test_extract_top_pairs_synergistic_weights_exceed_antagonistic(trained_checkpoint):
    cp_dir, X, feature_names, config = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X, config)
    pairs, _ = extract_top_pairs(matrix, feature_names, top_k=5, threshold_percentile=50)
    synergistic = [p for p in pairs if p["classification"] == "synergistic"]
    antagonistic = [p for p in pairs if p["classification"] == "antagonistic"]
    if synergistic and antagonistic:
        assert min(p["weight"] for p in synergistic) >= max(p["weight"] for p in antagonistic)
