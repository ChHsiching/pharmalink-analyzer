"""Tests for ExpressionService — validates facade orchestration."""

import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sympy

from app.exceptions import (
    CheckpointNotFoundError,
    ExpressionNotFoundError,
    SymbolicRegressionError,
    UndoLimitError,
)
from app.models.expression import ExpressionHistoryResponse, ExpressionResponse
from app.services.expression import ExpressionService
from app.services.expression_pipeline import PipelineResult
from app.ml.expression_tree import expr_to_latex, get_complexity


def _fake_pareto():
    a = sympy.Symbol("A")
    b = sympy.Symbol("B")
    return [
        {"index": 0, "latex": "A + B", "complexity": 3, "loss": 0.05, "r2_score": 0.80, "sympy_expr": a + b},
        {"index": 1, "latex": "A^{2} + 1", "complexity": 5, "loss": 0.01, "r2_score": 0.90, "sympy_expr": a ** 2 + 1},
        {"index": 2, "latex": "A^{2} + B^{2} + 1", "complexity": 7, "loss": 0.005, "r2_score": 0.95, "sympy_expr": a ** 2 + b ** 2 + 1},
    ]


def _fake_pipeline_result(**overrides):
    pareto = _fake_pareto()
    best = pareto[-1]  # most complex
    defaults = {
        "sympy_expr": best["sympy_expr"],
        "latex": best["latex"],
        "complexity": best["complexity"],
        "loss": best["loss"],
        "r2_score": best["r2_score"],
        "pareto_equations": pareto,
    }
    defaults.update(overrides)
    return PipelineResult(**defaults)


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.run.return_value = _fake_pipeline_result()
    return pipeline


@pytest.fixture
def service(mock_pipeline):
    resolver = MagicMock()
    return ExpressionService(resolver=resolver, pipeline=mock_pipeline)


class TestExpressionStateDataclass:
    def test_expression_state_is_dataclass(self):
        from dataclasses import is_dataclass
        from app.services.expression_state import ExpressionState
        assert is_dataclass(ExpressionState)

    def test_expression_state_has_required_fields(self):
        from app.services.expression_state import ExpressionState
        state = ExpressionState(
            expr_id="expr_test",
            model_id="model_test",
            current_sympy=sympy.Symbol("A"),
            current_latex="A",
            current_complexity=1,
            current_r2=0.95,
        )
        assert state.expr_id == "expr_test"


class TestGenerate:
    def test_generate_returns_expression_response(self, service, mock_pipeline):
        result = service.generate("model-abc")
        assert isinstance(result, ExpressionResponse)
        assert result.model_id == "model-abc"
        assert result.expr_id.startswith("expr_")
        assert result.latex == "A^{2} + B^{2} + 1"
        assert result.complexity == 7
        assert result.r2_score == 0.95
        assert result.pareto_count == 3
        assert result.pareto_index == 2
        mock_pipeline.run.assert_called_once_with("model-abc", 10, preset="standard")

    def test_generate_expr_id_is_timestamp_format(self, service):
        result = service.generate("model-abc")
        expr_id = result.expr_id
        assert expr_id.startswith("expr_")
        suffix = expr_id[len("expr_"):]
        assert len(suffix) == 14
        assert suffix.isdigit()
        datetime.strptime(suffix, "%Y%m%d%H%M%S")

    def test_generate_propagates_pipeline_error(self, service, mock_pipeline):
        mock_pipeline.run.side_effect = SymbolicRegressionError("Julia backend not installed")
        with pytest.raises(SymbolicRegressionError):
            service.generate("model-abc")

    def test_generate_propagates_checkpoint_not_found(self, service, mock_pipeline):
        mock_pipeline.run.side_effect = CheckpointNotFoundError("missing")
        with pytest.raises(CheckpointNotFoundError):
            service.generate("missing")


class TestSimplify:
    def test_simplify_unknown_expr_raises_404(self, service):
        with pytest.raises(ExpressionNotFoundError):
            service.simplify("expr_nonexistent")

    def test_simplify_updates_expression(self, service):
        result = service.generate("model-abc")
        expr_id = result.expr_id
        simplified = service.simplify(expr_id)
        assert isinstance(simplified, ExpressionResponse)
        assert simplified.expr_id == expr_id


class TestOptimize:
    def test_optimize_unknown_expr_raises_404(self, service):
        with pytest.raises(ExpressionNotFoundError):
            service.optimize("expr_nonexistent")

    def test_optimize_no_training_data_returns_unchanged(self, mock_pipeline):
        """Without training data, optimize returns the expression unchanged."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        mock_pipeline.run.return_value = _fake_pipeline_result(
            sympy_expr=A + B,
            latex="A + B",
            complexity=3,
        )
        service = ExpressionService(resolver=MagicMock(), pipeline=mock_pipeline)
        result = service.generate("model-abc")
        optimized = service.optimize(result.expr_id)
        assert optimized.latex == result.latex
        assert optimized.complexity == result.complexity

    def test_optimize_refits_coefficients(self, mock_pipeline):
        """Optimize refits coefficients and preserves expression structure."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        expr = sympy.Float(3.0) * A + sympy.Float(2.0) * B + sympy.Float(1.0)
        X_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        X_test = [[2.0, 3.0], [6.0, 7.0]]
        y_train = [3 * a + 2 * b + 1 for a, b in X_train]
        y_test = [3 * a + 2 * b + 1 for a, b in X_test]

        mock_pipeline.run.return_value = _fake_pipeline_result(
            sympy_expr=expr,
            latex=expr_to_latex(expr),
            complexity=get_complexity(expr),
            r2_score=0.80,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            aug_names=["A", "B"],
        )
        service = ExpressionService(resolver=MagicMock(), pipeline=mock_pipeline)
        result = service.generate("model-abc")
        optimized = service.optimize(result.expr_id)
        assert isinstance(optimized, ExpressionResponse)
        assert optimized.expr_id == result.expr_id
        # Variables preserved
        assert "A" in optimized.latex or "B" in optimized.latex

    def test_optimize_rejects_r2_decrease(self, mock_pipeline):
        """If refitting would decrease R2, original expression is returned."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        expr = A + B
        # Quadratic data — linear expression can't fit well
        X_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        X_test = [[2.0, 3.0], [6.0, 7.0]]
        y_train = [a ** 2 + b ** 2 for a, b in X_train]
        y_test = [a ** 2 + b ** 2 for a, b in X_test]

        mock_pipeline.run.return_value = _fake_pipeline_result(
            sympy_expr=expr,
            latex="A + B",
            complexity=3,
            r2_score=0.99,  # Inflated initial R2
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            aug_names=["A", "B"],
        )
        service = ExpressionService(resolver=MagicMock(), pipeline=mock_pipeline)
        result = service.generate("model-abc")
        optimized = service.optimize(result.expr_id)
        # Rejected — returns original unchanged
        assert optimized.latex == result.latex

    def test_optimize_updates_indicators(self, mock_pipeline):
        """Optimize recomputes indicators after refitting."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        expr = sympy.Float(3.0) * A + sympy.Float(2.0) * B + sympy.Float(1.0)
        X_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        X_test = [[2.0, 3.0], [6.0, 7.0]]
        y_train = [3 * a + 2 * b + 1 for a, b in X_train]
        y_test = [3 * a + 2 * b + 1 for a, b in X_test]

        mock_pipeline.run.return_value = _fake_pipeline_result(
            sympy_expr=expr,
            latex=expr_to_latex(expr),
            complexity=get_complexity(expr),
            r2_score=0.80,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            aug_names=["A", "B"],
        )
        service = ExpressionService(resolver=MagicMock(), pipeline=mock_pipeline)
        result = service.generate("model-abc")
        optimized = service.optimize(result.expr_id)
        # R2 should improve (data matches expression perfectly)
        assert optimized.r2_score >= 0.99


class TestGetTree:
    def test_get_tree_unknown_expr_raises_404(self, service):
        with pytest.raises(ExpressionNotFoundError):
            service.get_tree("expr_nonexistent")


class TestHistory:
    def test_get_history_unknown_expr_raises_404(self, service):
        with pytest.raises(ExpressionNotFoundError):
            service.get_history("expr_nonexistent")

    def test_history_records_operations(self, service):
        result = service.generate("model-abc")
        expr_id = result.expr_id
        service.simplify(expr_id)
        history = service.get_history(expr_id)
        assert isinstance(history, ExpressionHistoryResponse)
        assert len(history.history) == 2
        assert history.history[0].operation == "generate"
        assert history.history[1].operation == "simplify"
        assert history.current_index == 1


class TestUndo:
    def test_undo_unknown_expr_raises_404(self, service):
        with pytest.raises(ExpressionNotFoundError):
            service.undo("expr_nonexistent")

    def test_undo_restores_previous_state(self, service):
        result = service.generate("model-abc")
        expr_id = result.expr_id
        original_latex = result.latex
        service.simplify(expr_id)
        undone = service.undo(expr_id)
        assert undone.latex == original_latex

    def test_undo_at_beginning_raises_undo_limit(self, service):
        result = service.generate("model-abc")
        expr_id = result.expr_id
        with pytest.raises(UndoLimitError):
            service.undo(expr_id)


class TestAsyncGenerate:
    def test_start_generate_returns_task_id(self, service):
        from app.models.expression import TaskStatusResponse
        result = service.start_generate("model-abc")
        assert isinstance(result, TaskStatusResponse)
        assert result.task_id.startswith("task_")
        assert result.status in ("pending", "running")

    def test_get_task_result_unknown_raises_404(self, service):
        from app.exceptions import ExpressionTaskNotFoundError
        with pytest.raises(ExpressionTaskNotFoundError):
            service.get_task_result("task_nonexistent")

    def test_get_task_result_completed(self, service):
        from app.models.expression import TaskStatusResponse
        task = service.start_generate("model-abc")
        time.sleep(1)
        result = service.get_task_result(task.task_id)
        assert result.status == "completed"
        assert result.result is not None
        assert result.result["latex"] == "A^{2} + B^{2} + 1"

    def test_get_task_result_failed(self, mock_pipeline):
        mock_pipeline.run.side_effect = RuntimeError("Julia crashed")
        service = ExpressionService(resolver=MagicMock(), pipeline=mock_pipeline)
        task = service.start_generate("model-abc")
        time.sleep(1)
        result = service.get_task_result(task.task_id)
        assert result.status == "failed"
        assert result.error is not None


class TestTTLTaskCleanup:
    def test_cleanup_removes_old_completed_tasks(self, service):
        stale_task_id = "task_stale"
        service._tasks[stale_task_id] = {
            "status": "completed",
            "result": {"expr_id": "expr_old"},
            "error": None,
            "created_at": time.time() - 1801,
        }
        service._cleanup_stale_tasks(ttl_seconds=1800)
        assert stale_task_id not in service._tasks

    def test_cleanup_keeps_recent_tasks(self, service):
        recent_task_id = "task_recent"
        service._tasks[recent_task_id] = {
            "status": "completed",
            "result": {"expr_id": "expr_new"},
            "error": None,
            "created_at": time.time() - 100,
        }
        service._cleanup_stale_tasks(ttl_seconds=1800)
        assert recent_task_id in service._tasks

    def test_cleanup_keeps_running_tasks(self, service):
        running_task_id = "task_running"
        service._tasks[running_task_id] = {
            "status": "running",
            "result": None,
            "error": None,
            "created_at": time.time() - 9999,
        }
        service._cleanup_stale_tasks(ttl_seconds=1800)
        assert running_task_id in service._tasks


class TestPersistenceIntegration:
    """Verify that ExpressionService with checkpoint_dir persists across restarts."""

    def test_generate_persists_across_service_restart(self, tmp_path, mock_pipeline):
        resolver = MagicMock()
        svc1 = ExpressionService(
            resolver=resolver, pipeline=mock_pipeline, checkpoint_dir=tmp_path,
        )
        result = svc1.generate("model-abc")
        expr_id = result.expr_id

        # Fresh service with same checkpoint_dir should recover state
        svc2 = ExpressionService(
            resolver=resolver, pipeline=mock_pipeline, checkpoint_dir=tmp_path,
        )
        tree = svc2.get_tree(expr_id)
        assert tree.expr_id == expr_id
        assert tree.latex == result.latex
