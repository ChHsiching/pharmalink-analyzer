import asyncio
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.dependencies import get_checkpoint_resolver, get_training_service
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.data_loader import DataLoader, data_loader
from app.services.training import TrainingService


@pytest.fixture
def csv_dir(tmp_path):
    rng = np.random.default_rng(42)
    n, cols = 20, 3
    data = rng.standard_normal((n, cols + 1)).astype(np.float32)
    header = "QA,CGA,CA,TC\n"
    rows = "\n".join(",".join(f"{v:.4f}" for v in row) for row in data) + "\n"
    (tmp_path / "fruit-test-tc.csv").write_text(header + rows)
    return tmp_path


@pytest_asyncio.fixture
async def client(csv_dir, tmp_path):
    data_loader._datasets.clear()
    dl = DataLoader(data_dir=csv_dir)
    dl.load_preset_datasets()
    for k, v in dl._datasets.items():
        data_loader._datasets[k] = v

    test_training = TrainingService(checkpoint_dir=tmp_path / "ckpts", data_loader=data_loader)
    test_training._status = "idle"
    test_training._progress = []

    test_resolver = CheckpointResolver(
        checkpoint_dir=tmp_path / "ckpts", data_loader=data_loader,
    )

    app.dependency_overrides[get_training_service] = lambda: test_training
    app.dependency_overrides[get_checkpoint_resolver] = lambda: test_resolver

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    del app.dependency_overrides[get_training_service]
    del app.dependency_overrides[get_checkpoint_resolver]


@pytest.mark.asyncio
async def test_start_training_success(client):
    resp = await client.post("/api/v1/models/train", json={
        "dataset_id": "fruit-test-tc",
        "d_model": 32, "n_heads": 2, "n_layers": 1,
        "epochs": 3, "k_folds": 2,
        "augmentation": {"enabled": False},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert data["status"] == "started"

    final_status = None
    for _ in range(40):
        await asyncio.sleep(0.5)
        status_resp = await client.get("/api/v1/models/train/status")
        final_status = status_resp.json()["status"]
        if final_status != "running":
            break

    assert final_status == "completed", f"Training did not complete: {final_status}"


@pytest.mark.asyncio
async def test_start_training_dataset_not_found(client):
    resp = await client.post("/api/v1/models/train", json={
        "dataset_id": "nonexistent",
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_status_idle(client):
    resp = await client.get("/api/v1/models/train/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "idle"


@pytest.mark.asyncio
async def test_stop_training_when_not_running(client):
    resp = await client.post("/api/v1/models/train/stop")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_list_checkpoints_empty(client):
    resp = await client.get("/api/v1/models/checkpoints")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_load_checkpoint_not_found(client):
    resp = await client.post("/api/v1/models/checkpoints/load?checkpoint_id=nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_start_training_no_target_variable(client, csv_dir):
    """Training should return 422 when dataset has no target variable selected."""
    # Load a dataset, then replace it with one that has an empty target
    rng = np.random.default_rng(99)
    n, cols = 5, 3
    data = rng.standard_normal((n, cols + 1)).astype(np.float32)
    header = "A,B,C,D\n"
    rows = "\n".join(",".join(f"{v:.4f}" for v in row) for row in data) + "\n"
    (csv_dir / "no-target.csv").write_text(header + rows)

    from app.services.data_loader import DataLoader as DL
    dl = DataLoader(data_dir=csv_dir)
    dl.load_preset_datasets()
    # Manually set target to empty string to simulate "not selected"
    meta, df = dl._datasets["no-target"]
    dl._datasets["no-target"] = (meta.model_copy(update={"target": ""}), df)

    # Register in global data_loader so TrainingService sees it
    data_loader._datasets["no-target"] = dl._datasets["no-target"]

    # Re-create training service with updated data_loader
    svc = TrainingService(checkpoint_dir=csv_dir / "ckpts2", data_loader=data_loader)
    svc._status = "idle"
    svc._progress = []
    app.dependency_overrides[get_training_service] = lambda: svc

    resp = await client.post("/api/v1/models/train", json={
        "dataset_id": "no-target",
    })
    assert resp.status_code == 422
    assert "target variable" in resp.json()["detail"]

    # Cleanup
    del data_loader._datasets["no-target"]
