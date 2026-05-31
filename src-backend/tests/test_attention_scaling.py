"""Verify attention extraction uses saved scaler to normalize features.

Bug: extract_attention_weights() passed raw features to a model trained on
StandardScaler-normalized data, collapsing the attention distribution.

Reproduction strategy: use heterogeneously-scaled data (feature 0 ~5-25,
feature 1 ~0.02-0.05) so StandardScaler has non-trivial params. Without
the fix, raw features go to the model directly; with the fix, they are
normalized first. The consistency test compares function output against
manual scale + model inference — it fails before fix, passes after.
"""
import numpy as np
import pytest
import torch

from app.ml.attention_extractor import extract_attention_weights
from app.ml.checkpoint_loader import load_scaler_params, load_model


def _make_heteroscale_data(
    rng: np.random.Generator, n_samples: int, n_features: int,
) -> np.ndarray:
    """Generate data with heterogeneous feature scales (like real Leaf100HDL).

    Feature 0: ~5-25 range (like CA, MA, CRA)
    Feature 1: ~0.02-0.05 range (like GUA, QUE)
    Features 2+: standard normal
    """
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    X[:, 0] = X[:, 0] * 5 + 15
    X[:, 1] = X[:, 1] * 0.01 + 0.03
    return X


@pytest.fixture
def scaled_checkpoint(make_checkpoint):
    """Checkpoint trained on heterogeneously-scaled data.

    Default make_checkpoint uses randn (mean≈0, std≈1), which makes
    StandardScaler near-identity. We need non-trivial scaler params to
    properly test that scaling is applied.
    """
    rng = np.random.default_rng(42)
    X = _make_heteroscale_data(rng, 10, 5)
    y = rng.standard_normal(10).astype(np.float32)
    return make_checkpoint(X=X, y=y)


def test_attention_matches_manual_scaling(scaled_checkpoint):
    """After fix, function output must equal manual scale + model inference."""
    env = scaled_checkpoint
    cp_dir, config, n_feat = env["cp_dir"], env["config"], env["n_features"]

    rng = np.random.default_rng(99)
    X_raw = _make_heteroscale_data(rng, 20, n_feat)

    # Expected: manual scale + direct model inference
    mean, scale = load_scaler_params(cp_dir / "scaler_params.json")
    X_scaled = (X_raw - mean) / scale
    model = load_model(cp_dir, n_feat, config)
    with torch.no_grad():
        _, weights = model(torch.tensor(X_scaled, dtype=torch.float32))
    expected = weights.mean(dim=0).numpy()

    # Actual: function should handle scaling internally
    actual = extract_attention_weights(cp_dir, X_raw, config)

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_attention_not_uniform_on_heteroscale_data(scaled_checkpoint):
    """After fix, attention on heterogeneously-scaled data must be non-uniform."""
    env = scaled_checkpoint
    cp_dir, config, n_feat = env["cp_dir"], env["config"], env["n_features"]

    rng = np.random.default_rng(77)
    X = _make_heteroscale_data(rng, 30, n_feat)
    attn = extract_attention_weights(cp_dir, X, config)

    assert np.var(attn) > 1e-8, (
        f"Attention near-uniform (var={np.var(attn):.2e}), "
        "features likely not scaled before inference"
    )
