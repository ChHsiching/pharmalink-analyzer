import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.dependencies import get_expression_service
from app.models.expression import (
    ExpressionHistoryEntry,
    ExpressionHistoryResponse,
    ExpressionNode,
    ExpressionResponse,
)


class MockExpressionService:
    def generate(self, model_id, top_k=10):
        return ExpressionResponse(
            expr_id="expr_test1",
            model_id=model_id,
            latex="x_{0} + x_{1}",
            complexity=3,
            r2_score=0.92,
            tree=ExpressionNode(type="variable", value="x0", children=[]),
        )

    def simplify(self, expr_id):
        if expr_id == "not_found":
            raise _not_found("not_found")
        return ExpressionResponse(
            expr_id=expr_id,
            model_id="model1",
            latex="x_{0}",
            complexity=1,
            r2_score=0.90,
            tree=ExpressionNode(type="variable", value="x0", children=[]),
        )

    def optimize(self, expr_id):
        if expr_id == "not_found":
            raise _not_found("not_found")
        return ExpressionResponse(
            expr_id=expr_id,
            model_id="model1",
            latex="x_{0}",
            complexity=1,
            r2_score=0.88,
            tree=ExpressionNode(type="variable", value="x0", children=[]),
        )

    def get_tree(self, expr_id):
        if expr_id == "not_found":
            raise _not_found("not_found")
        return ExpressionResponse(
            expr_id=expr_id,
            model_id="model1",
            latex="x_{0} + x_{1}",
            complexity=3,
            r2_score=0.92,
            tree=ExpressionNode(
                type="operator",
                value="+",
                children=[
                    ExpressionNode(type="variable", value="x0", children=[]),
                    ExpressionNode(type="variable", value="x1", children=[]),
                ],
            ),
        )

    def get_history(self, expr_id):
        if expr_id == "not_found":
            raise _not_found("not_found")
        return ExpressionHistoryResponse(
            expr_id=expr_id,
            current_index=1,
            history=[
                ExpressionHistoryEntry(
                    operation="generate", latex="x_{0}+x_{1}", complexity=3, r2_score=0.92
                ),
                ExpressionHistoryEntry(
                    operation="simplify", latex="x_{0}", complexity=1, r2_score=0.90
                ),
            ],
        )

    def undo(self, expr_id, steps=1):
        if expr_id == "not_found":
            raise _not_found("not_found")
        return ExpressionResponse(
            expr_id=expr_id,
            model_id="model1",
            latex="x_{0} + x_{1}",
            complexity=3,
            r2_score=0.92,
            tree=ExpressionNode(type="variable", value="x0", children=[]),
        )


def _not_found(expr_id: str):
    from app.exceptions import ExpressionNotFoundError

    raise ExpressionNotFoundError(expr_id)


@pytest.fixture
def mock_service():
    app.dependency_overrides[get_expression_service] = lambda: MockExpressionService()
    yield
    del app.dependency_overrides[get_expression_service]


@pytest.mark.asyncio
async def test_generate_expression(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/expressions/generate/model1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["expr_id"] == "expr_test1"
    assert "latex" in data
    assert "complexity" in data


@pytest.mark.asyncio
async def test_simplify_expression(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/expressions/simplify/expr_test1")
    assert resp.status_code == 200
    assert resp.json()["complexity"] == 1


@pytest.mark.asyncio
async def test_simplify_not_found(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/expressions/simplify/not_found")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_optimize_expression(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/expressions/optimize/expr_test1")
    assert resp.status_code == 200
    assert resp.json()["complexity"] == 1


@pytest.mark.asyncio
async def test_get_tree(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions/tree/expr_test1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tree"]["type"] == "operator"
    assert len(data["tree"]["children"]) == 2


@pytest.mark.asyncio
async def test_get_history(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions/history/expr_test1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["history"]) == 2
    assert data["current_index"] == 1


@pytest.mark.asyncio
async def test_undo_expression(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/expressions/undo/expr_test1")
    assert resp.status_code == 200
    assert resp.json()["complexity"] == 3


@pytest.mark.asyncio
async def test_tree_not_found(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions/tree/not_found")
    assert resp.status_code == 404
