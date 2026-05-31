import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "datasets_loaded" in data
    assert "dataset_count" in data
    assert isinstance(data["datasets_loaded"], bool)
    assert isinstance(data["dataset_count"], int)


@pytest.mark.asyncio
async def test_health_no_datasets(client, monkeypatch):
    """AC4: datasets_loaded is false when no datasets loaded."""
    monkeypatch.setattr(
        "app.api.health.data_loader.list_datasets", lambda: []
    )
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["datasets_loaded"] is False
    assert data["dataset_count"] == 0
