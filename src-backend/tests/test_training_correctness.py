"""Correctness tests for training pipeline: loss convergence, attention validity, augmentation."""

import numpy as np
import pytest
import torch

from app.ml.transformer import FeatureTransformer
from app.ml.augmentation import augment
from app.models.training import AugmentationConfig


class TestFeatureTransformer:
    def test_forward_output_shape(self):
        model = FeatureTransformer(n_features=5, d_model=32, n_heads=2, n_layers=1)
        X = torch.randn(10, 5)
        pred, attn = model(X)
        assert pred.shape == (10,)
        assert attn.shape == (10, 5, 5)

    def test_attention_weights_non_negative(self):
        model = FeatureTransformer(n_features=4, d_model=16, n_heads=2, n_layers=1)
        X = torch.randn(20, 4)
        _, attn = model(X)
        assert (attn >= -1e-6).all()

    def test_weights_change_after_training(self):
        model = FeatureTransformer(n_features=5, d_model=32, n_heads=2, n_layers=1)
        X = torch.randn(20, 5)
        y = torch.randn(20)
        initial_params = {n: p.clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        model.train()
        pred, _ = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        changed = any(not torch.equal(initial_params[n], p) for n, p in model.named_parameters())
        assert changed


class TestAugmentation:
    def test_augment_increases_samples(self):
        X = np.random.randn(30, 5)
        y = np.random.randn(30)
        config = AugmentationConfig(enabled=True, gaussian_noise_sigma=0.1, bootstrap_enabled=True)
        X_aug, y_aug = augment(X, y, config)
        assert X_aug.shape[0] > X.shape[0]
        assert X_aug.shape[1] == X.shape[1]

    def test_augment_disabled_returns_original(self):
        X = np.random.randn(30, 5)
        y = np.random.randn(30)
        config = AugmentationConfig(enabled=False)
        X_aug, y_aug = augment(X, y, config)
        np.testing.assert_array_equal(X_aug, X)
        np.testing.assert_array_equal(y_aug, y)
