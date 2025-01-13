import pytest
from httpx import ASGITransport, AsyncClient

from app.exceptions import SymbolicRegressionError
from app.main import app
from app.dependencies import get_expression_service
from app.models.expression import (
    ExpressionHistoryEntry,
    ExpressionHistoryResponse,
    ExpressionNode,
    ExpressionResponse,
)


class MockExpressionService:
    def start_generate(self, model_id, top_k=10):
        from app.models.expression import TaskStatusResponse
        return TaskStatusResponse(task_id="task_test1", status="pending")

    def get_task_result(self, task_id):
        from app.models.expression import TaskStatusResponse
        if task_id == "not_found":
            raise _task_not_found("not_found")
        if task_id == "task_completed":
            return TaskStatusResponse(
                task_id=task_id,
                status="completed",
                result={
                    "expr_id": "expr_test1",
                    "model_id": "model1",
                    "latex": "x_{0} + x_{1}",
                    "complexity": 3,
                    "r2_score": 0.92,
                    "tree": {"type": "variable", "value": "x0", "children": []},
                },
            )
        if task_id == "task_failed":
            return TaskStatusResponse(
                task_id=task_id,
                status="failed",
                error="Julia backend not installed",
            )
        return TaskStatusResponse(task_id=task_id, status="running")

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


def _task_not_found(task_id: str):
    from app.exceptions import ExpressionTaskNotFoundError

    raise ExpressionTaskNotFoundError(task_id)


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
    assert data["task_id"] == "task_test1"
    assert data["status"] == "pending"


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


@pytest.mark.asyncio
async def test_generate_pysr_error_returns_503():
    """SymbolicRegressionError should return 503, not unhandled 500."""
    class MockFailingService:
        def start_generate(self, model_id, top_k=10):
            raise SymbolicRegressionError("Julia backend not installed")

    app.dependency_overrides[get_expression_service] = lambda: MockFailingService()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/expressions/generate/model1")
        assert resp.status_code == 503
        assert "Julia" in resp.json()["detail"]
    finally:
        del app.dependency_overrides[get_expression_service]


@pytest.mark.asyncio
async def test_get_task_result_running(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions/result/task_running")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


@pytest.mark.asyncio
async def test_get_task_result_completed(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions/result/task_completed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["result"]["latex"] == "x_{0} + x_{1}"


@pytest.mark.asyncio
async def test_get_task_result_failed(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions/result/task_failed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert "Julia" in data["error"]


@pytest.mark.asyncio
async def test_get_task_result_not_found(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions/result/not_found")
    assert resp.status_code == 404
