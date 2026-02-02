"""Tests for TrainingService — validates facade orchestration with mock pipeline."""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from app.models.training import TrainingConfig, AugmentationConfig, TrainingProgress
from app.services.data_loader import DataLoader
from app.services.training import TrainingService
from app.services.training_pipeline import TrainingResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_training_result(**overrides):
    defaults = {
        "model_state": {"w": torch.tensor([1.0])},
        "scaler": StandardScaler(),
        "best_loss": 0.05,
        "stopped": False,
    }
    defaults.update(overrides)
    return TrainingResult(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_dir(tmp_path):
    rng = np.random.default_rng(42)
    n, cols = 30, 5
    data = rng.standard_normal((n, cols + 1)).astype(np.float32)
    header = "QA,CGA,CA,PIS,AST,TC\n"
    rows = "\n".join(",".join(f"{v:.4f}" for v in row) for row in data) + "\n"
    (tmp_path / "fruit-test-tc.csv").write_text(header + rows)
    return tmp_path


@pytest.fixture
def loader(csv_dir):
    dl = DataLoader(data_dir=csv_dir)
    dl.load_preset_datasets()
    return dl


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.train.return_value = _fake_training_result()
    return pipeline


@pytest.fixture
def mock_checkpoint_manager():
    return MagicMock()


@pytest.fixture
def service(tmp_path, loader, mock_pipeline, mock_checkpoint_manager):
    return TrainingService(
        checkpoint_dir=tmp_path / "ckpts",
        data_loader=loader,
        pipeline=mock_pipeline,
        checkpoint_manager=mock_checkpoint_manager,
    )


# ---------------------------------------------------------------------------
# Initial status
# ---------------------------------------------------------------------------

class TestInitialStatus:
    def test_initial_status_is_idle(self, service):
        status = service.get_status()
        assert status.status == "idle"


# ---------------------------------------------------------------------------
# Start training
# ---------------------------------------------------------------------------

class TestStartTraining:
    @pytest.mark.asyncio
    async def test_completes_with_mock_pipeline(self, service, mock_pipeline, mock_checkpoint_manager):
        config = TrainingConfig(
            dataset_id="fruit-test-tc",
            d_model=32, n_heads=2, n_layers=1,
            epochs=5, k_folds=2,
            augmentation=AugmentationConfig(enabled=False),
        )
        task_id = await service.start_training(config)
        assert task_id

        for _ in range(30):
            await asyncio.sleep(0.1)
            if service.get_status().status != "running":
                break

        status = service.get_status()
        assert status.status == "completed"
        mock_pipeline.train.assert_called_once()
        mock_checkpoint_manager.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_stopped_early_skips_save(self, service, mock_pipeline, mock_checkpoint_manager):
        mock_pipeline.train.return_value = _fake_training_result(stopped=True)

        config = TrainingConfig(
            dataset_id="fruit-test-tc",
            d_model=32, n_heads=2, n_layers=1,
            epochs=5, k_folds=2,
            augmentation=AugmentationConfig(enabled=False),
        )
        await service.start_training(config)

        for _ in range(30):
            await asyncio.sleep(0.1)
            if service.get_status().status != "running":
                break

        assert service.get_status().status == "stopped"
        mock_checkpoint_manager.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_cannot_start_while_running(self, service, mock_pipeline):
        def slow_train(*args, **kwargs):
            time.sleep(2)
            return _fake_training_result()

        mock_pipeline.train.side_effect = slow_train

        config = TrainingConfig(
            dataset_id="fruit-test-tc",
            d_model=32, n_heads=2, n_layers=1,
            epochs=200, k_folds=5,
            augmentation=AugmentationConfig(enabled=False),
        )
        await service.start_training(config)
        await asyncio.sleep(0.2)
        with pytest.raises(ValueError, match="already in progress"):
            await service.start_training(config)
        service.request_stop()

    @pytest.mark.asyncio
    async def test_dataset_not_found_raises(self, tmp_path):
        svc = TrainingService(checkpoint_dir=tmp_path / "ckpts")
        config = TrainingConfig(dataset_id="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            await svc.start_training(config)

    @pytest.mark.asyncio
    async def test_pipeline_error_finishes_with_error(self, service, mock_pipeline, mock_checkpoint_manager):
        mock_pipeline.train.side_effect = RuntimeError("CUDA OOM")

        config = TrainingConfig(
            dataset_id="fruit-test-tc",
            d_model=32, n_heads=2, n_layers=1,
            epochs=3, k_folds=2,
            augmentation=AugmentationConfig(enabled=False),
        )
        await service.start_training(config)

        for _ in range(30):
            await asyncio.sleep(0.1)
            if service.get_status().status != "running":
                break

        assert service.get_status().status == "error"
        mock_checkpoint_manager.save.assert_not_called()


# ---------------------------------------------------------------------------
# Stop training
# ---------------------------------------------------------------------------

class TestStopTraining:
    @pytest.mark.asyncio
    async def test_request_stop(self, service, mock_pipeline):
        def train_with_stop_check(*args, **kwargs):
            is_stopped = kwargs.get("is_stopped", lambda: False)
            time.sleep(0.5)
            if is_stopped():
                return _fake_training_result(stopped=True)
            return _fake_training_result()

        mock_pipeline.train.side_effect = train_with_stop_check

        config = TrainingConfig(
            dataset_id="fruit-test-tc",
            d_model=32, n_heads=2, n_layers=1,
            epochs=200, k_folds=5,
            augmentation=AugmentationConfig(enabled=False),
        )
        await service.start_training(config)
        await asyncio.sleep(0.3)
        service.request_stop()

        for _ in range(30):
            await asyncio.sleep(0.1)
            if service.get_status().status != "running":
                break

        assert service.get_status().status == "stopped"


# ---------------------------------------------------------------------------
# Progress queue fan-out
# ---------------------------------------------------------------------------

class TestProgressQueues:
    def test_add_and_remove_queue(self, service):
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        service.add_progress_queue(q1)
        service.add_progress_queue(q2)
        assert len(service._progress_queues) == 2

        service.remove_progress_queue(q1)
        assert len(service._progress_queues) == 1

        service.remove_progress_queue(q2)
        assert len(service._progress_queues) == 0

    def test_remove_nonexistent_queue_no_error(self, service):
        q = asyncio.Queue()
        service.remove_progress_queue(q)

    def test_multiple_queues_all_receive_progress(self, service, mock_pipeline):
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        service.add_progress_queue(q1)
        service.add_progress_queue(q2)
        service._loop = asyncio.new_event_loop()

        progress = TrainingProgress(
            epoch=1, fold=1, train_loss=0.5, val_loss=0.6, r2=0.5, status="training",
        )
        service._push_progress(progress)

        service._loop.close()

        assert service._progress[-1] == progress


class TestTaskIdFormat:
    @pytest.mark.asyncio
    async def test_task_id_is_14_digit_timestamp(self, service):
        config = TrainingConfig(
            dataset_id="fruit-test-tc",
            d_model=32, n_heads=2, n_layers=1,
            epochs=5, k_folds=2,
            augmentation=AugmentationConfig(enabled=False),
        )
        import re

        task_id = await service.start_training(config)
        assert re.match(r"^\d{14}$", task_id), f"task_id '{task_id}' is not a 14-digit timestamp"
