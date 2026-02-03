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
    def list_by_model(self, model_id: str):
        if model_id == "model1":
            return [{"expr_id": "expr_test1", "latex": "x_{0} + x_{1}"}]
        return []

    def start_generate(self, model_id, top_k=10, preset="standard"):
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
                    "variable_impact": {"x0": 1.0, "x1": 0.5},
                    "indicators": {
                        "train_r2": 0.95, "test_r2": 0.92,
                        "train_mae": 0.05, "test_mae": 0.08,
                        "train_mse": 0.01, "test_mse": 0.02,
                        "train_nmse": 0.05, "test_nmse": 0.08,
                        "train_rmse": 0.1, "test_rmse": 0.14,
                        "depth": 3.0, "length": 15.0,
                    },
                    "target_name": "yield",
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
            variable_impact={"x0": 1.0, "x1": 0.5},
            indicators={
                "train_r2": 0.95, "test_r2": 0.92,
                "train_mae": 0.05, "test_mae": 0.08,
                "train_mse": 0.01, "test_mse": 0.02,
                "train_nmse": 0.05, "test_nmse": 0.08,
                "train_rmse": 0.1, "test_rmse": 0.14,
                "depth": 3.0, "length": 15.0,
            },
            target_name="yield",
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
            variable_impact={"x0": 1.0, "x1": 0.5},
            tree=ExpressionNode(type="variable", value="x0", children=[]),
            indicators={
                "train_r2": 0.95, "test_r2": 0.92,
                "train_mae": 0.05, "test_mae": 0.08,
                "train_mse": 0.01, "test_mse": 0.02,
                "train_nmse": 0.05, "test_nmse": 0.08,
                "train_rmse": 0.1, "test_rmse": 0.14,
                "depth": 3.0, "length": 15.0,
            },
            target_name="yield",
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
            variable_impact={"x0": 1.0, "x1": 0.5},
            indicators={
                "train_r2": 0.95, "test_r2": 0.92,
                "train_mae": 0.05, "test_mae": 0.08,
                "train_mse": 0.01, "test_mse": 0.02,
                "train_nmse": 0.05, "test_nmse": 0.08,
                "train_rmse": 0.1, "test_rmse": 0.14,
                "depth": 3.0, "length": 15.0,
            },
            target_name="yield",
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
            variable_impact={"x0": 1.0, "x1": 0.5},
            indicators={
                "train_r2": 0.95, "test_r2": 0.92,
                "train_mae": 0.05, "test_mae": 0.08,
                "train_mse": 0.01, "test_mse": 0.02,
                "train_nmse": 0.05, "test_nmse": 0.08,
                "train_rmse": 0.1, "test_rmse": 0.14,
                "depth": 3.0, "length": 15.0,
            },
            target_name="yield",
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
async def test_generate_with_valid_preset(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/expressions/generate/model1?preset=quick",
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_generate_with_invalid_preset_returns_422(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/expressions/generate/model1?preset=invalid",
        )
    assert resp.status_code == 422


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
async def test_generate_gplearn_error_returns_503():
    """SymbolicRegressionError should return 503, not unhandled 500."""
    class MockFailingService:
        def start_generate(self, model_id, top_k=10, preset="standard"):
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


@pytest.mark.asyncio
async def test_generate_response_includes_variable_impact(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions/result/task_completed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    impact = data["result"]["variable_impact"]
    assert isinstance(impact, dict)
    assert "x0" in impact


@pytest.mark.asyncio
async def test_generate_response_includes_indicators(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions/result/task_completed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert "indicators" in data["result"]
    indicators = data["result"]["indicators"]
    assert isinstance(indicators, dict)
    expected_keys = {
        "train_r2", "test_r2", "train_mae", "test_mae",
        "train_mse", "test_mse", "train_nmse", "test_nmse",
        "train_rmse", "test_rmse", "depth", "length",
    }
    assert set(indicators.keys()) == expected_keys


@pytest.mark.asyncio
async def test_generate_response_includes_target_name(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions/result/task_completed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert "target_name" in data["result"]
    assert isinstance(data["result"]["target_name"], str)


@pytest.mark.asyncio
async def test_list_expressions_by_checkpoint(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions", params={"checkpoint_id": "model1"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["expr_id"] == "expr_test1"
    assert data[0]["latex"] == "x_{0} + x_{1}"


@pytest.mark.asyncio
async def test_list_expressions_empty_for_unknown_checkpoint(mock_service):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/expressions", params={"checkpoint_id": "unknown"})
    assert resp.status_code == 200
    assert resp.json() == []
