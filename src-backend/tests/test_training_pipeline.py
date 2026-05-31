"""Tests for TrainingPipeline — validates pure ML training orchestration."""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from app.models.training import TrainingConfig, AugmentationConfig, TrainingProgress
from app.services.training_pipeline import TrainingPipeline, TrainingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5)).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] * 0.5 + rng.standard_normal(30) * 0.1).astype(np.float32)
    return X, y


@pytest.fixture
def small_config():
    return TrainingConfig(
        dataset_id="test",
        d_model=16, n_heads=2, n_layers=1,
        epochs=3, k_folds=2,
        early_stopping_patience=10,
        dropout=0.0,
        augmentation=AugmentationConfig(enabled=False),
    )


@pytest.fixture
def pipeline():
    return TrainingPipeline()


# ---------------------------------------------------------------------------
# TrainingResult dataclass
# ---------------------------------------------------------------------------

class TestTrainingResult:
    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        assert is_dataclass(TrainingResult)

    def test_holds_result(self):
        result = TrainingResult(
            model_state={"w": torch.tensor([1.0])},
            scaler=StandardScaler(),
            best_loss=0.1,
        )
        assert result.best_loss == 0.1
        assert not result.stopped


# ---------------------------------------------------------------------------
# Pipeline train() — happy path
# ---------------------------------------------------------------------------

class TestPipelineTrain:
    def test_returns_training_result(self, pipeline, small_data, small_config):
        X, y = small_data
        result = pipeline.train(X, y, 5, small_config)
        assert isinstance(result, TrainingResult)
        assert result.model_state is not None
        assert result.scaler is not None
        assert result.best_loss < float("inf")
        assert not result.stopped

    def test_calls_on_progress(self, pipeline, small_data, small_config):
        X, y = small_data
        progress_calls = []
        result = pipeline.train(X, y, 5, small_config, on_progress=progress_calls.append)
        assert len(progress_calls) > 0
        assert all(isinstance(p, TrainingProgress) for p in progress_calls)
        assert progress_calls[-1].fold == 2

    def test_augmentation_enabled(self, pipeline, small_data):
        X, y = small_data
        config = TrainingConfig(
            dataset_id="test",
            d_model=16, n_heads=2, n_layers=1,
            epochs=3, k_folds=2,
            dropout=0.0,
            augmentation=AugmentationConfig(enabled=True),
        )
        result = pipeline.train(X, y, 5, config)
        assert result.model_state is not None

    def test_early_stopping_patience(self, small_data):
        X, y = small_data
        config = TrainingConfig(
            dataset_id="test",
            d_model=16, n_heads=2, n_layers=1,
            epochs=100, k_folds=2,
            early_stopping_patience=2,
            dropout=0.0,
            augmentation=AugmentationConfig(enabled=False),
        )
        pipeline = TrainingPipeline()
        progress_calls = []
        result = pipeline.train(X, y, 5, config, on_progress=progress_calls.append)
        assert len(progress_calls) < 200


# ---------------------------------------------------------------------------
# Pipeline train() — is_stopped callback
# ---------------------------------------------------------------------------

class TestPipelineStopped:
    def test_stops_immediately(self, pipeline, small_data, small_config):
        X, y = small_data
        result = pipeline.train(
            X, y, 5, small_config,
            is_stopped=lambda: True,
        )
        assert result.stopped
        assert result.model_state is None

    def test_stops_after_first_progress(self, pipeline, small_data, small_config):
        X, y = small_data
        call_count = [0]

        def on_progress(p):
            call_count[0] += 1

        result = pipeline.train(
            X, y, 5, small_config,
            on_progress=on_progress,
            is_stopped=lambda: call_count[0] >= 1,
        )
        assert result.stopped


def test_train_returns_all_fold_states():
    """TrainingResult should contain model states and scalers for all folds."""
    pipeline = TrainingPipeline()
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5)).astype(np.float32)
    y = rng.standard_normal(30).astype(np.float32)
    config = TrainingConfig(
        dataset_id="test", d_model=16, n_heads=2, n_layers=1,
        epochs=5, k_folds=3,
    )
    result = pipeline.train(X, y, n_features=5, config=config)

    assert len(result.fold_states) == 3
    assert len(result.fold_scalers) == 3
    assert result.best_fold_idx is not None
    assert 0 <= result.best_fold_idx < 3
    for state in result.fold_states:
        assert isinstance(state, dict)
        assert len(state) > 0
    for scaler in result.fold_scalers:
        assert isinstance(scaler, StandardScaler)
