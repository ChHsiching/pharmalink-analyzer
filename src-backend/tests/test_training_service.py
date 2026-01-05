import asyncio
from pathlib import Path

import numpy as np
import pytest

from app.models.training import TrainingConfig, AugmentationConfig
from app.services.data_loader import DataLoader
from app.services.training import TrainingService


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
def service(tmp_path, loader):
    return TrainingService(
        checkpoint_dir=tmp_path / "ckpts",
        data_loader=loader,
    )


def test_initial_status_is_idle(service):
    status = service.get_status()
    assert status.status == "idle"


@pytest.mark.asyncio
async def test_start_training_completes(service):
    config = TrainingConfig(
        dataset_id="fruit-test-tc",
        d_model=32, n_heads=2, n_layers=1,
        epochs=5, k_folds=2,
        early_stopping_patience=10,
        dropout=0.0,
        augmentation=AugmentationConfig(enabled=False),
    )
    task_id = await service.start_training(config)
    assert task_id
    assert service.get_status().status == "running"

    for _ in range(60):
        await asyncio.sleep(0.5)
        if service.get_status().status != "running":
            break

    status = service.get_status()
    assert status.status == "completed"
    assert len(status.progress) > 0
    assert status.progress[-1].fold == 2


@pytest.mark.asyncio
async def test_start_training_saves_checkpoint(service):
    config = TrainingConfig(
        dataset_id="fruit-test-tc",
        d_model=32, n_heads=2, n_layers=1,
        epochs=3, k_folds=2,
        dropout=0.0,
        augmentation=AugmentationConfig(enabled=False),
    )
    await service.start_training(config)

    for _ in range(60):
        await asyncio.sleep(0.5)
        if service.get_status().status != "running":
            break

    checkpoints = service.list_checkpoints()
    assert len(checkpoints) == 1
    assert checkpoints[0].dataset_id == "fruit-test-tc"


@pytest.mark.asyncio
async def test_stop_training(service):
    config = TrainingConfig(
        dataset_id="fruit-test-tc",
        d_model=32, n_heads=2, n_layers=1,
        epochs=200, k_folds=5,
        augmentation=AugmentationConfig(enabled=False),
    )
    await service.start_training(config)
    await asyncio.sleep(0.5)
    service.request_stop()

    for _ in range(30):
        await asyncio.sleep(0.5)
        if service.get_status().status != "running":
            break

    assert service.get_status().status == "stopped"


@pytest.mark.asyncio
async def test_cannot_start_while_running(service):
    config = TrainingConfig(
        dataset_id="fruit-test-tc",
        d_model=32, n_heads=2, n_layers=1,
        epochs=200, k_folds=5,
        augmentation=AugmentationConfig(enabled=False),
    )
    await service.start_training(config)
    with pytest.raises(ValueError, match="already in progress"):
        await service.start_training(config)
    service.request_stop()


@pytest.mark.asyncio
async def test_dataset_not_found_raises(tmp_path):
    svc = TrainingService(checkpoint_dir=tmp_path / "ckpts")
    config = TrainingConfig(dataset_id="nonexistent")
    with pytest.raises(ValueError, match="not found"):
        await svc.start_training(config)
