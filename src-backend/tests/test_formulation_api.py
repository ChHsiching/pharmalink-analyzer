"""Tests for formulation API endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock

from app.dependencies import get_formulation_service
from app.main import app
from app.models.formulation import FormulationCandidate, FormulationResponse


def _mock_response():
    return FormulationResponse(
        expr_id="expr_test", candidates=[
            FormulationCandidate(components={"X0": 0.9, "X1": 0.8}, predicted_response=1.7, rank=1),
        ], feature_names=["X0", "X1"], attention_weights={"X0": 0.6, "X1": 0.4},
    )


@pytest.fixture
def mock_service():
    svc = MagicMock()
    svc.formulate.return_value = _mock_response()
    app.dependency_overrides[get_formulation_service] = lambda: svc
    yield svc
    del app.dependency_overrides[get_formulation_service]


@pytest.mark.asyncio
async def test_formulate_success(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/formulation", json={"expr_id": "expr_test", "top_k": 5, "n_samples": 1000})
        assert resp.status_code == 200
        data = resp.json()
        assert data["expr_id"] == "expr_test"
        assert len(data["candidates"]) == 1


@pytest.mark.asyncio
async def test_formulate_defaults(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/formulation", json={"expr_id": "expr_test"})
        assert resp.status_code == 200
        mock_service.formulate.assert_called_once_with("expr_test", top_k=5, n_samples=1000)


@pytest.mark.asyncio
async def test_formulate_invalid_request():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/formulation", json={"expr_id": "expr_test", "top_k": 0})
        assert resp.status_code == 422
