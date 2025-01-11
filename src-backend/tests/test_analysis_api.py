import numpy as np
import pytest
import pytest_asyncio
import torch
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.dependencies import get_analysis_service
from app.services.analysis import AnalysisService
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

    test_service = AnalysisService(
        resolver=CheckpointResolver(tmp_path / "checkpoints", dl),
    )
    app.dependency_overrides[get_analysis_service] = lambda: test_service
    yield {"checkpoint_id": env["cp_dir"].name, "feature_count": env["n_features"]}
    del app.dependency_overrides[get_analysis_service]


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
            params={"threshold": 50},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold"] == 50.0
    synergistic = [e for e in data["edges"] if e["classification"] == "synergistic"]
    antagonistic = [e for e in data["edges"] if e["classification"] == "antagonistic"]
    assert len(synergistic) > 0
    assert len(antagonistic) > 0


@pytest.mark.asyncio
async def test_get_network_percentile_zero_all_synergistic(analysis_env):
    info = analysis_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/api/v1/analysis/network/{info['checkpoint_id']}",
            params={"threshold": 0},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert all(e["classification"] == "synergistic" for e in data["edges"])


@pytest.mark.asyncio
async def test_get_network_default_threshold_is_50(analysis_env):
    info = analysis_env
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/api/v1/analysis/network/{info['checkpoint_id']}")
    assert resp.status_code == 200
    assert resp.json()["threshold"] == 50.0


@pytest.mark.asyncio
async def test_heatmap_not_found():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/analysis/attention/nonexistent/heatmap")
    assert resp.status_code == 404
