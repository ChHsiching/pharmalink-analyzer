# src-backend/tests/test_analysis_api.py
import numpy as np
import pytest
import pytest_asyncio
import torch
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services import analysis as analysis_module
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.data_loader import DataLoader


@pytest.fixture
def analysis_env(tmp_path, make_checkpoint):
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

    X = df[meta.feature_names].to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float32)

    env = make_checkpoint(
        X=X, y=y,
        n_features=len(meta.feature_names),
        training_epochs=20,
        checkpoint_name=f"checkpoints/{meta.id}_task1",
        dataset_id=meta.id,
    )

    analysis_module.analysis_service = analysis_module.AnalysisService(
        resolver=CheckpointResolver(tmp_path / "checkpoints", dl),
    )
    return {"checkpoint_id": env["cp_dir"].name, "feature_count": env["n_features"]}


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
