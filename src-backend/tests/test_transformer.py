import torch
import pytest

from app.ml.transformer import FeatureTransformer


def test_forward_pass_output_shape():
    model = FeatureTransformer(n_features=20, d_model=64, n_heads=4, n_layers=2)
    x = torch.randn(8, 20)
    pred, attn = model(x)
    assert pred.shape == (8,)
    assert attn.shape == (8, 20, 20)


def test_attention_weights_are_normalized():
    torch.manual_seed(42)
    model = FeatureTransformer(n_features=10, d_model=32, n_heads=2, n_layers=1)
    x = torch.randn(4, 10)
    _, attn = model(x)
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_model_can_overfit_small_data():
    torch.manual_seed(42)
    model = FeatureTransformer(n_features=5, d_model=32, n_heads=2, n_layers=2, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()

    x = torch.randn(10, 5)
    y = (x ** 2).sum(dim=1)

    for _ in range(500):
        optimizer.zero_grad()
        pred, _ = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred, _ = model(x)
        final_loss = criterion(pred, y).item()

    assert final_loss < 0.5, f"Model failed to converge, loss={final_loss}"


def test_different_feature_counts():
    for n in [20, 21]:
        model = FeatureTransformer(n_features=n, d_model=32, n_heads=2, n_layers=1)
        x = torch.randn(4, n)
        pred, attn = model(x)
        assert pred.shape == (4,)
        assert attn.shape == (4, n, n)


def test_single_sample_batch():
    model = FeatureTransformer(n_features=10, d_model=32, n_heads=2, n_layers=1)
    x = torch.randn(1, 10)
    pred, attn = model(x)
    assert pred.shape == (1,)
    assert attn.shape == (1, 10, 10)
