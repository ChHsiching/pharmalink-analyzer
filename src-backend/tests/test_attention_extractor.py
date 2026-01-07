# src-backend/tests/test_attention_extractor.py
import numpy as np
import pytest
import torch
import torch.nn as nn

from app.ml.attention_extractor import extract_attention_weights, extract_top_pairs
from app.ml.transformer import FeatureTransformer
from app.models.training import TrainingConfig


@pytest.fixture
def trained_checkpoint(tmp_path):
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    feature_names = [f"comp_{i}" for i in range(n_features)]

    config = TrainingConfig(
        dataset_id="test_ds",
        d_model=16, n_heads=2, n_layers=1, dropout=0.0,
    )
    model = FeatureTransformer(
        n_features=n_features,
        d_model=config.d_model, n_heads=config.n_heads,
        n_layers=config.n_layers, dropout=config.dropout,
    )
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(20):
        pred, _ = model(X_t)
        loss = loss_fn(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cp_dir = tmp_path / "test_ds_task1"
    cp_dir.mkdir()
    torch.save(model.state_dict(), cp_dir / "model.pt")
    (cp_dir / "config.json").write_text(config.model_dump_json())
    (cp_dir / "metrics.json").write_text('{"final_val_loss": 0.1}')

    return cp_dir, X, feature_names


def test_extract_attention_weights_shape(trained_checkpoint):
    cp_dir, X, feature_names = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X)
    assert matrix.shape == (len(feature_names), len(feature_names))
    assert isinstance(matrix, np.ndarray)


def test_extract_attention_weights_positive(trained_checkpoint):
    cp_dir, X, _ = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X)
    assert (matrix >= 0).all()


def test_extract_attention_weights_row_sums(trained_checkpoint):
    cp_dir, X, _ = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X)
    row_sums = matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_extract_top_pairs_count(trained_checkpoint):
    cp_dir, X, feature_names = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X)
    pairs, _ = extract_top_pairs(matrix, feature_names, top_k=3)
    assert len(pairs) == 3


def test_extract_top_pairs_sorted_descending(trained_checkpoint):
    cp_dir, X, feature_names = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X)
    pairs, _ = extract_top_pairs(matrix, feature_names, top_k=5)
    weights = [p["weight"] for p in pairs]
    assert weights == sorted(weights, reverse=True)


def test_extract_top_pairs_classification(trained_checkpoint):
    cp_dir, X, feature_names = trained_checkpoint
    matrix = extract_attention_weights(cp_dir, X)
    pairs, threshold = extract_top_pairs(matrix, feature_names, top_k=5)
    for pair in pairs:
        expected = "synergistic" if pair["weight"] > threshold else "antagonistic"
        assert pair["classification"] == expected
