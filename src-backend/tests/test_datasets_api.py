import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.data_loader import data_loader

TEST_CSV = b"QA,CGA,CA,TC\n1.0,2.0,3.0,10.0\n4.0,5.0,6.0,20.0\n7.0,8.0,9.0,30.0\n"


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture(autouse=True)
async def reset_datasets():
    data_loader._datasets.clear()
    meta = data_loader.add_dataset("test-tc.csv", TEST_CSV, name="Test TC")
    # Simulate a preset dataset for delete-protection tests
    preset_meta = meta.model_copy(update={"is_preset": True})
    df = data_loader._datasets[meta.id][1]
    data_loader._datasets[meta.id] = (preset_meta, df)
    yield
    data_loader._datasets.clear()


@pytest.mark.asyncio
async def test_list_datasets(client):
    response = await client.get("/api/v1/datasets")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "Test TC"
    assert data[0]["target"] == "TC"
    assert data[0]["plant_part"] == "Fruit"


@pytest.mark.asyncio
async def test_upload_dataset(client):
    response = await client.post(
        "/api/v1/datasets",
        files={"file": ("custom.csv", b"X,Y,T\n1.0,2.0,10.0\n3.0,4.0,20.0\n", "text/csv")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "custom"
    assert data["is_preset"] is False
    assert data["n_samples"] == 2


@pytest.mark.asyncio
async def test_upload_non_csv_rejected(client):
    response = await client.post(
        "/api/v1/datasets",
        files={"file": ("test.txt", b"not a csv", "text/plain")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_empty_file_rejected(client):
    response = await client.post(
        "/api/v1/datasets",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_dataset_detail(client):
    response = await client.get("/api/v1/datasets/test-tc")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-tc"
    assert len(data["preview"]) == 3
    assert data["target"] == "TC"
    assert data["n_features"] == 3


@pytest.mark.asyncio
async def test_get_dataset_not_found(client):
    response = await client.get("/api/v1/datasets/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_dataset_stats(client):
    response = await client.get("/api/v1/datasets/test-tc/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"] == "test-tc"
    assert len(data["stats"]) == 4
    tc_stat = next(s for s in data["stats"] if s["column"] == "TC")
    assert tc_stat["mean"] == 20.0
    assert tc_stat["min"] == 10.0


@pytest.mark.asyncio
async def test_get_stats_not_found(client):
    response = await client.get("/api/v1/datasets/nonexistent/stats")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_custom_dataset(client):
    upload_resp = await client.post(
        "/api/v1/datasets",
        files={"file": ("deleteme.csv", b"A,B,C\n1,2,3\n", "text/csv")},
    )
    custom_id = upload_resp.json()["id"]

    response = await client.delete(f"/api/v1/datasets/{custom_id}")
    assert response.status_code == 204

    verify = await client.get(f"/api/v1/datasets/{custom_id}")
    assert verify.status_code == 404


@pytest.mark.asyncio
async def test_delete_preset_forbidden(client):
    response = await client.delete("/api/v1/datasets/test-tc")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_delete_not_found(client):
    response = await client.delete("/api/v1/datasets/nonexistent")
    assert response.status_code == 404
