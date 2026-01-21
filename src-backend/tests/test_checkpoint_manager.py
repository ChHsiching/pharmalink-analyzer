"""Tests for CheckpointManager — validates checkpoint persistence."""

import json

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from app.models.training import TrainingConfig, TrainingProgress
from app.services.checkpoint_manager import CheckpointManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def checkpoint_dir(tmp_path):
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture
def manager(checkpoint_dir):
    return CheckpointManager(checkpoint_dir)


@pytest.fixture
def model_state():
    return {"w": torch.tensor([1.0, 2.0])}


@pytest.fixture
def scaler():
    s = StandardScaler()
    s.fit(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
    return s


@pytest.fixture
def progress():
    return [
        TrainingProgress(epoch=1, fold=1, train_loss=0.5, val_loss=0.6, r2=0.5, status="training"),
        TrainingProgress(epoch=2, fold=1, train_loss=0.3, val_loss=0.4, r2=0.7, status="training"),
    ]


@pytest.fixture
def config():
    return TrainingConfig(dataset_id="ds1")


# ---------------------------------------------------------------------------
# Save — file creation
# ---------------------------------------------------------------------------

class TestCheckpointManagerSave:
    def test_creates_checkpoint_directory(self, manager, checkpoint_dir, config, model_state, scaler, progress):
        ckpt_path = manager.save("task1", config, model_state, 0.1, scaler, progress)
        assert ckpt_path.is_dir()
        assert ckpt_path.name == "ds1_task1"

    def test_saves_model_file(self, manager, config, model_state, scaler, progress):
        ckpt_path = manager.save("task1", config, model_state, 0.1, scaler, progress)
        assert (ckpt_path / "model.pt").exists()
        loaded = torch.load(ckpt_path / "model.pt", map_location="cpu", weights_only=True)
        assert "w" in loaded

    def test_saves_config_file(self, manager, config, model_state, scaler, progress):
        ckpt_path = manager.save("task1", config, model_state, 0.1, scaler, progress)
        assert (ckpt_path / "config.json").exists()
        saved_config = json.loads((ckpt_path / "config.json").read_text())
        assert saved_config["dataset_id"] == "ds1"

    def test_saves_scaler_params(self, manager, config, model_state, scaler, progress):
        ckpt_path = manager.save("task1", config, model_state, 0.1, scaler, progress)
        assert (ckpt_path / "scaler_params.json").exists()
        scaler_data = json.loads((ckpt_path / "scaler_params.json").read_text())
        assert "mean" in scaler_data
        assert "scale" in scaler_data
        assert len(scaler_data["mean"]) == 2
        assert all(s > 0 for s in scaler_data["scale"])

    def test_saves_metrics_with_history(self, manager, config, model_state, scaler, progress):
        ckpt_path = manager.save("task1", config, model_state, 0.1, scaler, progress)
        metrics = json.loads((ckpt_path / "metrics.json").read_text())
        assert metrics["final_val_loss"] == 0.1
        assert len(metrics["history"]) == 2
        assert metrics["history"][0]["epoch"] == 1

    def test_saves_metrics_without_history(self, manager, config, model_state, scaler):
        ckpt_path = manager.save("task1", config, model_state, 0.1, scaler, [])
        metrics = json.loads((ckpt_path / "metrics.json").read_text())
        assert metrics["final_val_loss"] == 0.1
        assert "history" not in metrics

    def test_returns_checkpoint_path(self, manager, checkpoint_dir, config, model_state, scaler):
        result = manager.save("task1", config, model_state, 0.1, scaler, [])
        assert result == checkpoint_dir / "ds1_task1"
