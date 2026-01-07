# src-backend/tests/test_analysis_api.py
import numpy as np
import pytest
import pytest_asyncio
import torch
import torch.nn as nn
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.ml.transformer import FeatureTransformer
from app.models.training import TrainingConfig
from app.services import analysis as analysis_module
from app.services.data_loader import DataLoader


@pytest.fixture
def analysis_env(tmp_path):
    np.random.seed(42)
    torch.manual_seed(42)

    header = "f0,f1,f2,f3,f4,target"
    rng = np.random.default_rng(42)
    rows = "\n".join(
        ",".join(f"{v:.4f}" for v in row)
        for row in rng.standard_normal((20, 6)).astype(np.float32)
    )
    csv_path = tmp_path / "test_analysis.csv"
    csv_path.write_text(header + "\n" + rows)

    dl = DataLoader()
    meta, df = dl._load_csv(csv_path)
    dl._datasets[meta.id] = (meta, df)

    feature_names = meta.feature_names
    X = df[feature_names].to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float32)

    config = TrainingConfig(
        dataset_id=meta.id,
        d_model=16, n_heads=2, n_layers=1, dropout=0.0,
    )
    model = FeatureTransformer(
        n_features=len(feature_names),
        d_model=config.d_model, n_heads=config.n_heads,
        n_layers=config.n_layers, dropout=config.dropout,
    )
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(20):
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
    (cp_dir / "metrics.json").write_text('{"final_val_loss": 0.1}')

    analysis_module.analysis_service = analysis_module.AnalysisService(
        data_loader=dl, checkpoint_dir=cp_base,
    )
    return {"checkpoint_id": cp_dir.name, "feature_count": len(feature_names)}


@pytest.mark.asyncio
async def test_get_attention_matrix(analysis_env):
    info = analysis_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/api/v1/analysis/attention/{info['checkpoint_id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == info["checkpoint_id"]
    assert len(data["feature_names"]) == info["feature_count"]
    assert len(data["matrix"]) == info["feature_count"]
    assert len(data["matrix"][0]) == info["feature_count"]
    assert len(data["pairs"]) == 10
    assert "threshold" in data


@pytest.mark.asyncio
async def test_get_attention_matrix_not_found():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/analysis/attention/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_heatmap(analysis_env):
    info = analysis_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/api/v1/analysis/attention/{info['checkpoint_id']}/heatmap")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == info["checkpoint_id"]
    assert len(data["values"]) == info["feature_count"]
    assert data["min_value"] <= data["max_value"]


@pytest.mark.asyncio
async def test_get_network_graph(analysis_env):
    info = analysis_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/api/v1/analysis/network/{info['checkpoint_id']}")
    assert resp.status_code == 200
    data = resp.json()
    n = info["feature_count"]
    assert len(data["nodes"]) == n
    assert len(data["edges"]) == n * (n - 1) // 2
    assert data["threshold"] > 0


@pytest.mark.asyncio
async def test_get_network_with_threshold(analysis_env):
    info = analysis_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/api/v1/analysis/network/{info['checkpoint_id']}",
            params={"threshold": 0.5},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold"] == 0.5
    for edge in data["edges"]:
        if edge["weight"] >= 0.5:
            assert edge["classification"] == "synergistic"
        else:
            assert edge["classification"] == "antagonistic"


@pytest.mark.asyncio
async def test_heatmap_not_found():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/analysis/attention/nonexistent/heatmap")
    assert resp.status_code == 404
