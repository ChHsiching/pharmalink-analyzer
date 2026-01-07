# tests/test_evaluation_api.py
import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.ml.transformer import FeatureTransformer
from app.models.training import TrainingConfig
from app.services import evaluation as evaluation_module
from app.services.data_loader import DataLoader


@pytest.fixture
def evaluation_env(tmp_path):
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

    feature_names = meta.feature_names
    X = df[feature_names].to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float32)

    config = TrainingConfig(
        dataset_id=meta.id,
        d_model=16,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        k_folds=3,
        epochs=5,
    )
    model = FeatureTransformer(
        n_features=n_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(30):
        pred, _ = model(X_t)
        loss = loss_fn(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cp_base = tmp_path / "checkpoints"
    cp_dir = cp_base / f"{meta.id}_task1"
    cp_dir.mkdir(parents=True)
    torch.save(model.state_dict(), cp_dir / "model.pt")
    (cp_dir / "config.json").write_text(config.model_dump_json())
    (cp_dir / "metrics.json").write_text(
        json.dumps(
            {
                "final_val_loss": 0.1,
                "history": [
                    {
                        "epoch": 1,
                        "fold": 1,
                        "train_loss": 0.5,
                        "val_loss": 0.6,
                        "r2": 0.5,
                        "status": "training",
                    },
                    {
                        "epoch": 2,
                        "fold": 1,
                        "train_loss": 0.3,
                        "val_loss": 0.4,
                        "r2": 0.7,
                        "status": "training",
                    },
                ],
            }
        )
    )

    evaluation_module.evaluation_service = evaluation_module.EvaluationService(
        data_loader=dl,
        checkpoint_dir=cp_base,
    )
    return {"checkpoint_id": cp_dir.name}


@pytest.mark.asyncio
async def test_metrics_not_found():
    """Tracer bullet: 404 response for nonexistent checkpoint."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/v1/evaluation/metrics/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_loss_curve(evaluation_env):
    """Loss curve reads from metrics.json history."""
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
    """Metrics endpoint returns fold metrics and aggregate statistics."""
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
    """Predictions endpoint returns actual vs predicted pairs."""
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
    """Residuals endpoint returns residuals, bins, mean, and std."""
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
