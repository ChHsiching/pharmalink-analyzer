import numpy as np
import pytest

from app.models.training import AugmentationConfig
from app.ml.augmentation import augment


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 5)).astype(np.float32)
    y = rng.standard_normal(20).astype(np.float32)
    return X, y


def test_augment_both_enabled_triples_samples(sample_data):
    X, y = sample_data
    config = AugmentationConfig(enabled=True, gaussian_noise_sigma=0.05, bootstrap_enabled=True)
    X_aug, y_aug = augment(X, y, config)
    assert X_aug.shape[0] == 60
    assert y_aug.shape[0] == 60
    assert X_aug.shape[1] == 5


def test_augment_noise_only(sample_data):
    X, y = sample_data
    config = AugmentationConfig(enabled=True, gaussian_noise_sigma=0.05, bootstrap_enabled=False)
    X_aug, y_aug = augment(X, y, config)
    assert X_aug.shape[0] == 40


def test_augment_bootstrap_only(sample_data):
    X, y = sample_data
    config = AugmentationConfig(enabled=True, gaussian_noise_sigma=0.0, bootstrap_enabled=True)
    X_aug, y_aug = augment(X, y, config)
    assert X_aug.shape[0] == 40


def test_augment_noise_distribution(sample_data):
    X, y = sample_data
    config = AugmentationConfig(enabled=True, gaussian_noise_sigma=0.1, bootstrap_enabled=False)
    np.random.seed(123)
    X_aug, _ = augment(X, y, config)
    noise_added = X_aug[20:] - X
    assert abs(noise_added.std() - 0.1) < 0.05


def test_augment_disabled_returns_original(sample_data):
    X, y = sample_data
    config = AugmentationConfig(enabled=False)
    X_aug, y_aug = augment(X, y, config)
    assert X_aug.shape[0] == 20
    np.testing.assert_array_equal(X_aug, X)


def test_augment_preserves_feature_count(sample_data):
    X, y = sample_data
    config = AugmentationConfig()
    X_aug, _ = augment(X, y, config)
    assert X_aug.shape[1] == X.shape[1]
