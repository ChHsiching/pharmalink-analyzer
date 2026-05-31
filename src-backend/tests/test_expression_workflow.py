"""Integration test for the full expression workflow: generate -> simplify -> optimize -> undo -> history."""

import time

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock

import sympy

from app.dependencies import get_expression_service
from app.main import app
from app.services.expression import ExpressionService
from app.services.expression_pipeline import PipelineResult


def _fake_pipeline_result() -> PipelineResult:
    x0, x1, x2 = sympy.symbols("X0 X1 X2")
    expr = x0 + x1 * x2
    return PipelineResult(
        sympy_expr=expr,
        latex="X_{0} + X_{1} X_{2}",
        complexity=2,
        loss=0.01,
        r2_score=0.95,
        pareto_equations=[
            {"sympy_expr": expr, "latex": "X_{0} + X_{1} X_{2}", "complexity": 2, "r2_score": 0.95},
            {"sympy_expr": x0, "latex": "X_{0}", "complexity": 0, "r2_score": 0.80},
        ],
        variable_impact={"X0": 1.0, "X1": 0.8, "X2": 0.6},
        indicators={
            "train_r2": 0.96, "test_r2": 0.95,
            "train_mae": 0.04, "test_mae": 0.05,
            "train_mse": 0.005, "test_mse": 0.01,
            "train_nmse": 0.04, "test_nmse": 0.06,
            "train_rmse": 0.07, "test_rmse": 0.1,
            "depth": 2.0, "length": 12.0,
        },
        target_name="yield",
    )


@pytest.fixture
def mock_service():
    resolver = MagicMock()
    pipeline = MagicMock()
    pipeline.run.return_value = _fake_pipeline_result()
    service = ExpressionService(resolver=resolver, pipeline=pipeline)
    app.dependency_overrides[get_expression_service] = lambda: service
    yield service
    del app.dependency_overrides[get_expression_service]


async def _generate_expression(client, model_id="model_1"):
    resp = await client.post(f"/api/v1/expressions/generate/{model_id}?top_k=10&preset=quick")
    assert resp.status_code == 200
    task = resp.json()
    task_id = task["task_id"]
    for _ in range(30):
        status = await client.get(f"/api/v1/expressions/result/{task_id}")
        data = status.json()
        if data["status"] == "completed":
            return data["result"]["expr_id"]
        if data["status"] == "failed":
            pytest.fail(f"Generation failed: {data.get('error')}")
        time.sleep(0.1)
    pytest.fail("Generation never completed")


@pytest.mark.asyncio
async def test_simplify_then_history(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        expr_id = await _generate_expression(client)

        resp = await client.post(f"/api/v1/expressions/simplify/{expr_id}")
        assert resp.status_code == 200
        assert resp.json()["expr_id"] == expr_id

        resp = await client.get(f"/api/v1/expressions/history/{expr_id}")
        assert resp.status_code == 200
        ops = [h["operation"] for h in resp.json()["history"]]
        assert "generate" in ops
        assert "simplify" in ops


@pytest.mark.asyncio
async def test_optimize_navigates_pareto(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        expr_id = await _generate_expression(client)

        resp = await client.post(f"/api/v1/expressions/optimize/{expr_id}")
        assert resp.status_code == 200
        assert resp.json()["complexity"] <= 2


@pytest.mark.asyncio
async def test_undo_restores_previous(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        expr_id = await _generate_expression(client)

        resp = await client.get(f"/api/v1/expressions/history/{expr_id}")
        original = resp.json()["history"][0]["latex"]

        await client.post(f"/api/v1/expressions/simplify/{expr_id}")
        resp = await client.post(f"/api/v1/expressions/undo/{expr_id}?steps=1")
        assert resp.status_code == 200
        assert resp.json()["latex"] == original


@pytest.mark.asyncio
async def test_simplify_invalid_expr_returns_404():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/expressions/simplify/nonexistent_expr")
        assert resp.status_code == 404
        assert "detail" in resp.json()
