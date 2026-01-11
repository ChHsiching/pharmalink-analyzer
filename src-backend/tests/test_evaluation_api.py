import json

import numpy as np
import pytest
import torch
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.dependencies import get_evaluation_service
from app.services.evaluation import EvaluationService
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.data_loader import DataLoader


@pytest.fixture
def evaluation_env(tmp_path, make_checkpoint):
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 30
    n_features = 5
    rng = np.random.default_rng(42)
    header = ",".join([f"f{i}" for i in range(n_features)] + ["target"])
    rows_data = rng.standard_normal((n_samples, n_features + 1)).astype(np.float32)
    rows_data[:, n_features] = (
        rows_data[:, 0] * 2
        + rows_data[:, 1] * 0.5
        + rng.standard_normal(n_samples) * 0.1
    )
    rows = "\n".join(",".join(f"{v:.4f}" for v in row) for row in rows_data)
    csv_path = tmp_path / "test_eval.csv"
    csv_path.write_text(header + "\n" + rows)

    dl = DataLoader()
    meta, df = dl._load_csv(csv_path)
    dl._datasets[meta.id] = (meta, df)

    X = df[meta.feature_names].to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float32)

    metrics_content = json.dumps({
        "final_val_loss": 0.1,
        "history": [
            {"epoch": 1, "fold": 1, "train_loss": 0.5, "val_loss": 0.6, "r2": 0.5, "status": "training"},
            {"epoch": 2, "fold": 1, "train_loss": 0.3, "val_loss": 0.4, "r2": 0.7, "status": "training"},
        ],
    })

    env = make_checkpoint(
        X=X, y=y,
        n_features=n_features,
        training_epochs=30,
        checkpoint_name=f"checkpoints/{meta.id}_task1",
        dataset_id=meta.id,
        k_folds=3,
        metrics_content=metrics_content,
    )

    test_service = EvaluationService(
        resolver=CheckpointResolver(tmp_path / "checkpoints", dl),
    )
    app.dependency_overrides[get_evaluation_service] = lambda: test_service
    yield {"checkpoint_id": env["cp_dir"].name}
    del app.dependency_overrides[get_evaluation_service]


@pytest.mark.asyncio
async def test_metrics_not_found():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/v1/evaluation/metrics/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_loss_curve(evaluation_env):
    info = evaluation_env
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/v1/evaluation/loss-curve/{info['checkpoint_id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == info["checkpoint_id"]
    assert len(data["folds"]) == 1
    assert len(data["folds"][0]["points"]) == 2


@pytest.mark.asyncio
async def test_get_metrics(evaluation_env):
    info = evaluation_env
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/v1/evaluation/metrics/{info['checkpoint_id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == info["checkpoint_id"]
    assert len(data["folds"]) == 3
    assert "aggregate" in data
    assert "r2_mean" in data["aggregate"]


@pytest.mark.asyncio
async def test_get_predictions(evaluation_env):
    info = evaluation_env
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/v1/evaluation/predictions/{info['checkpoint_id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == info["checkpoint_id"]
    assert len(data["predictions"]) > 0
    assert "actual" in data["predictions"][0]
    assert "predicted" in data["predictions"][0]


@pytest.mark.asyncio
async def test_get_residuals(evaluation_env):
    info = evaluation_env
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/v1/evaluation/residuals/{info['checkpoint_id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == info["checkpoint_id"]
    assert len(data["residuals"]) > 0
    assert len(data["bins"]) > 0
    assert "mean" in data
    assert "std" in data
