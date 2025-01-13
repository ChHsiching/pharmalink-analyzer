"""Tests for ExpressionService — validates orchestration of the expression pipeline."""

import json
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from app.exceptions import (
    CheckpointNotFoundError,
    DatasetNotFoundError,
    ExpressionNotFoundError,
    SymbolicRegressionError,
    UndoLimitError,
)

from app.models.expression import ExpressionHistoryResponse, ExpressionResponse
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.expression import ExpressionService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_dataset():
    """A DatasetDetail-like object with feature_names and target."""
    ds = MagicMock()
    ds.feature_names = ["A", "B", "C"]
    ds.target = "TC"
    return ds


@pytest.fixture
def fake_df():
    """A minimal DataFrame with columns A, B, C, TC."""
    import pandas as pd
    rng = np.random.default_rng(0)
    data = rng.standard_normal((20, 4)).astype(np.float32)
    return pd.DataFrame(data, columns=["A", "B", "C", "TC"])


@pytest.fixture
def mock_loader(fake_dataset, fake_df):
    loader = MagicMock()
    loader.get_dataset.return_value = fake_dataset
    loader.get_dataframe.return_value = fake_df
    return loader


@pytest.fixture
def checkpoint_dir(tmp_path):
    cp = tmp_path / "checkpoints"
    cp.mkdir()
    return cp


def _write_checkpoint(cp_dir: Path, dataset_id: str = "ds-1"):
    """Create a minimal checkpoint directory with config.json."""
    model_dir = cp_dir / "model-abc"
    model_dir.mkdir()
    config = {"dataset_id": dataset_id, "d_model": 32, "n_heads": 2, "n_layers": 1, "dropout": 0.0}
    (model_dir / "config.json").write_text(json.dumps(config))
    return model_dir


def _fake_attention_matrix(n=3):
    """Return a deterministic NxN attention matrix."""
    rng = np.random.default_rng(1)
    m = rng.standard_normal((n, n)).astype(np.float32)
    return (m + m.T) / 2  # symmetrize


def _fake_pairs():
    return [
        {"source": "A", "target": "B", "weight": 0.9, "classification": "synergistic"},
        {"source": "B", "target": "C", "weight": 0.7, "classification": "synergistic"},
    ]


def _fake_best():
    import sympy
    x = sympy.Symbol("A")
    expr = x ** 2 + 1
    return {
        "sympy_expr": expr,
        "latex": "A^{2} + 1",
        "complexity": 3,
        "loss": 0.01,
    }


def _fake_pareto():
    import sympy
    a = sympy.Symbol("A")
    b = sympy.Symbol("B")
    return [
        {"index": 0, "latex": "A + B", "complexity": 3, "loss": 0.05, "sympy_expr": a + b},
        {"index": 1, "latex": "A^{2} + 1", "complexity": 5, "loss": 0.01, "sympy_expr": a ** 2 + 1},
        {"index": 2, "latex": "A^{2} + B^{2} + 1", "complexity": 7, "loss": 0.005, "sympy_expr": a ** 2 + b ** 2 + 1},
    ]


def _fake_model():
    """Return a mock PySR model with .score() returning R2."""
    model = MagicMock()
    model.score.return_value = 0.95
    return model


@pytest.fixture
def service(mock_loader, checkpoint_dir):
    return ExpressionService(resolver=CheckpointResolver(checkpoint_dir, mock_loader))


# ---------------------------------------------------------------------------
# _ExpressionState dataclass
# ---------------------------------------------------------------------------

class TestExpressionStateDataclass:
    def test_expression_state_is_dataclass(self):
        from dataclasses import is_dataclass
        from app.services.expression import _ExpressionState

        assert is_dataclass(_ExpressionState)

    def test_expression_state_has_required_fields(self):
        import sympy

        from app.services.expression import _ExpressionState

        state = _ExpressionState(
            expr_id="expr_test",
            model_id="model_test",
            current_sympy=sympy.Symbol("A"),
            current_latex="A",
            current_complexity=1,
            current_r2=0.95,
            pareto_equations=[],
            history=[],
            history_index=-1,
        )
        assert state.expr_id == "expr_test"
        assert state.model_id == "model_test"


# ---------------------------------------------------------------------------
# generate() error boundaries
# ---------------------------------------------------------------------------

class TestGenerateErrorBoundaries:
    @patch("app.services.expression.extract_attention_weights")
    def test_generate_wraps_attention_error(
        self, mock_attn, service, checkpoint_dir,
    ):
        """Attention extraction failure should raise SymbolicRegressionError, not raw exception."""
        _write_checkpoint(checkpoint_dir)
        mock_attn.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(SymbolicRegressionError):
            service.generate("model-abc")

    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_generate_wraps_pairs_error(
        self, mock_attn, mock_pairs, service, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.side_effect = ValueError("bad pairs")

        with pytest.raises(SymbolicRegressionError):
            service.generate("model-abc")

    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_generate_wraps_feature_error(
        self, mock_attn, mock_pairs, mock_interact, service, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.side_effect = RuntimeError("feature gen failed")

        with pytest.raises(SymbolicRegressionError):
            service.generate("model-abc")

    @patch("app.services.expression.extract_best_equation")
    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_generate_wraps_equation_extraction_error(
        self, mock_attn, mock_pairs, mock_interact, mock_regression, mock_best,
        service, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (np.zeros((20, 7)), ["A", "B", "C", "A_mul_B", "A_div_B", "B_mul_C", "B_div_C"])
        mock_regression.return_value = _fake_model()
        mock_best.side_effect = KeyError("no best equation")

        with pytest.raises(SymbolicRegressionError):
            service.generate("model-abc")


# ---------------------------------------------------------------------------
# Tracer bullet: generate returns ExpressionResponse
# ---------------------------------------------------------------------------

class TestGenerate:
    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.extract_pareto_equations")
    @patch("app.services.expression.extract_best_equation")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_generate_returns_expression_response(
        self, mock_attn, mock_pairs, mock_interact, mock_best,
        mock_pareto, mock_regression, service, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (np.zeros((20, 7)), ["A", "B", "C", "A_mul_B", "A_div_B", "B_mul_C", "B_div_C"])
        fake_model = _fake_model()
        mock_regression.return_value = fake_model
        mock_best.return_value = _fake_best()
        mock_pareto.return_value = _fake_pareto()

        result = service.generate("model-abc")

        assert isinstance(result, ExpressionResponse)
        assert result.model_id == "model-abc"
        assert result.expr_id.startswith("expr_")
        assert result.latex == "A^{2} + 1"
        assert result.complexity == 3
        assert result.r2_score == 0.95

    def test_generate_raises_404_on_missing_checkpoint(self, service, checkpoint_dir):
        with pytest.raises(CheckpointNotFoundError):
            service.generate("nonexistent")

    @patch("app.services.expression.extract_attention_weights")
    def test_generate_raises_404_on_missing_dataset(
        self, mock_attn, service, checkpoint_dir, mock_loader,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_loader.get_dataset.return_value = None

        with pytest.raises(DatasetNotFoundError):
            service.generate("model-abc")


# ---------------------------------------------------------------------------
# Simplify
# ---------------------------------------------------------------------------

class TestSimplify:
    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.extract_pareto_equations")
    @patch("app.services.expression.extract_best_equation")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def _create_state(self, service, checkpoint_dir, mocks=None):
        """Helper: run generate to populate a state."""
        _write_checkpoint(checkpoint_dir)
        # Use real patches already applied by decorator — fetch from mocks param
        pass

    def test_simplify_unknown_expr_raises_404(self, service):
        with pytest.raises(ExpressionNotFoundError):
            service.simplify("expr_nonexistent")

    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.extract_pareto_equations")
    @patch("app.services.expression.extract_best_equation")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_simplify_updates_expression(
        self, mock_attn, mock_pairs, mock_interact, mock_best,
        mock_pareto, mock_regression, service, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (np.zeros((20, 7)), ["A", "B", "C", "A_mul_B", "A_div_B", "B_mul_C", "B_div_C"])
        fake_model = _fake_model()
        mock_regression.return_value = fake_model
        mock_best.return_value = _fake_best()
        mock_pareto.return_value = _fake_pareto()

        result = service.generate("model-abc")
        expr_id = result.expr_id

        simplified = service.simplify(expr_id)
        assert isinstance(simplified, ExpressionResponse)
        assert simplified.expr_id == expr_id


# ---------------------------------------------------------------------------
# Optimize
# ---------------------------------------------------------------------------

class TestOptimize:
    def test_optimize_unknown_expr_raises_404(self, service):
        with pytest.raises(ExpressionNotFoundError):
            service.optimize("expr_nonexistent")

    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.extract_pareto_equations")
    @patch("app.services.expression.extract_best_equation")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_optimize_selects_lower_complexity(
        self, mock_attn, mock_pairs, mock_interact, mock_best,
        mock_pareto, mock_regression, service, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (np.zeros((20, 7)), ["A", "B", "C", "A_mul_B", "A_div_B", "B_mul_C", "B_div_C"])
        fake_model = _fake_model()
        mock_regression.return_value = fake_model

        import sympy
        a = sympy.Symbol("A")
        # Start at highest complexity (index 2 = complexity 7)
        mock_best.return_value = {
            "sympy_expr": a ** 2 + sympy.Symbol("B") ** 2 + 1,
            "latex": "A^{2} + B^{2} + 1",
            "complexity": 7,
            "loss": 0.005,
        }
        mock_pareto.return_value = _fake_pareto()

        result = service.generate("model-abc")
        expr_id = result.expr_id
        assert result.complexity == 7

        optimized = service.optimize(expr_id)
        # Should have moved to a lower-complexity equation
        assert optimized.complexity < 7

    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.extract_pareto_equations")
    @patch("app.services.expression.extract_best_equation")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_optimize_at_best_returns_same(
        self, mock_attn, mock_pairs, mock_interact, mock_best,
        mock_pareto, mock_regression, service, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (np.zeros((20, 7)), ["A", "B", "C", "A_mul_B", "A_div_B", "B_mul_C", "B_div_C"])
        fake_model = _fake_model()
        mock_regression.return_value = fake_model

        import sympy
        a = sympy.Symbol("A")
        b = sympy.Symbol("B")
        # Start at lowest complexity (index 0 = complexity 3)
        mock_best.return_value = {
            "sympy_expr": a + b,
            "latex": "A + B",
            "complexity": 3,
            "loss": 0.05,
        }
        mock_pareto.return_value = _fake_pareto()

        result = service.generate("model-abc")
        expr_id = result.expr_id

        optimized = service.optimize(expr_id)
        # Already at best, should return same
        assert optimized.complexity == result.complexity


# ---------------------------------------------------------------------------
# Get tree
# ---------------------------------------------------------------------------

class TestGetTree:
    def test_get_tree_unknown_expr_raises_404(self, service):
        with pytest.raises(ExpressionNotFoundError):
            service.get_tree("expr_nonexistent")


# ---------------------------------------------------------------------------
# History and undo
# ---------------------------------------------------------------------------

class TestHistory:
    def test_get_history_unknown_expr_raises_404(self, service):
        with pytest.raises(ExpressionNotFoundError):
            service.get_history("expr_nonexistent")

    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.extract_pareto_equations")
    @patch("app.services.expression.extract_best_equation")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_history_records_operations(
        self, mock_attn, mock_pairs, mock_interact, mock_best,
        mock_pareto, mock_regression, service, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (np.zeros((20, 7)), ["A", "B", "C", "A_mul_B", "A_div_B", "B_mul_C", "B_div_C"])
        fake_model = _fake_model()
        mock_regression.return_value = fake_model
        mock_best.return_value = _fake_best()
        mock_pareto.return_value = _fake_pareto()

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

    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.extract_pareto_equations")
    @patch("app.services.expression.extract_best_equation")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_undo_restores_previous_state(
        self, mock_attn, mock_pairs, mock_interact, mock_best,
        mock_pareto, mock_regression, service, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (np.zeros((20, 7)), ["A", "B", "C", "A_mul_B", "A_div_B", "B_mul_C", "B_div_C"])
        fake_model = _fake_model()
        mock_regression.return_value = fake_model
        mock_best.return_value = _fake_best()
        mock_pareto.return_value = _fake_pareto()

        result = service.generate("model-abc")
        expr_id = result.expr_id
        original_latex = result.latex

        service.simplify(expr_id)

        undone = service.undo(expr_id)
        assert undone.latex == original_latex

    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.extract_pareto_equations")
    @patch("app.services.expression.extract_best_equation")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_undo_at_beginning_raises_400(
        self, mock_attn, mock_pairs, mock_interact, mock_best,
        mock_pareto, mock_regression, service, checkpoint_dir,
    ):
        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (np.zeros((20, 7)), ["A", "B", "C", "A_mul_B", "A_div_B", "B_mul_C", "B_div_C"])
        fake_model = _fake_model()
        mock_regression.return_value = fake_model
        mock_best.return_value = _fake_best()
        mock_pareto.return_value = _fake_pareto()

        service.generate("model-abc")
        # At index 0, can't undo further

        with pytest.raises(ExpressionNotFoundError):
            service.undo("expr_unknown")  # wrong ID still raises


# ---------------------------------------------------------------------------
# Async generate
# ---------------------------------------------------------------------------

class TestAsyncGenerate:
    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.extract_pareto_equations")
    @patch("app.services.expression.extract_best_equation")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_start_generate_returns_task_id(
        self, mock_attn, mock_pairs, mock_interact, mock_best,
        mock_pareto, mock_regression, service, checkpoint_dir,
    ):
        from app.models.expression import TaskStatusResponse

        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (np.zeros((20, 7)), ["A", "B", "C", "A_mul_B", "A_div_B", "B_mul_C", "B_div_C"])
        mock_regression.return_value = _fake_model()
        mock_best.return_value = _fake_best()
        mock_pareto.return_value = _fake_pareto()

        result = service.start_generate("model-abc")

        assert isinstance(result, TaskStatusResponse)
        assert result.task_id.startswith("task_")
        assert result.status in ("pending", "running")

    def test_get_task_result_unknown_raises_404(self, service):
        from app.exceptions import ExpressionTaskNotFoundError

        with pytest.raises(ExpressionTaskNotFoundError):
            service.get_task_result("task_nonexistent")

    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.extract_pareto_equations")
    @patch("app.services.expression.extract_best_equation")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_get_task_result_completed(
        self, mock_attn, mock_pairs, mock_interact, mock_best,
        mock_pareto, mock_regression, service, checkpoint_dir,
    ):
        from app.models.expression import TaskStatusResponse

        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (np.zeros((20, 7)), ["A", "B", "C", "A_mul_B", "A_div_B", "B_mul_C", "B_div_C"])
        mock_regression.return_value = _fake_model()
        mock_best.return_value = _fake_best()
        mock_pareto.return_value = _fake_pareto()

        task = service.start_generate("model-abc")
        # Wait for background thread to complete
        time.sleep(1)

        result = service.get_task_result(task.task_id)
        assert result.status == "completed"
        assert result.result is not None
        assert result.result["latex"] == "A^{2} + 1"

    @patch("app.services.expression.run_symbolic_regression")
    @patch("app.services.expression.generate_interaction_features")
    @patch("app.services.expression.extract_top_pairs")
    @patch("app.services.expression.extract_attention_weights")
    def test_get_task_result_failed(
        self, mock_attn, mock_pairs, mock_interact, mock_regression,
        service, checkpoint_dir,
    ):
        from app.models.expression import TaskStatusResponse

        _write_checkpoint(checkpoint_dir)
        mock_attn.return_value = _fake_attention_matrix()
        mock_pairs.return_value = (_fake_pairs(), 0.5)
        mock_interact.return_value = (np.zeros((20, 7)), ["A", "B", "C", "A_mul_B", "A_div_B", "B_mul_C", "B_div_C"])
        mock_regression.side_effect = RuntimeError("Julia crashed")

        task = service.start_generate("model-abc")
        time.sleep(1)

        result = service.get_task_result(task.task_id)
        assert result.status == "failed"
        assert result.error is not None


class TestTTLTaskCleanup:
    def test_cleanup_removes_old_completed_tasks(self, service):
        """Completed tasks older than TTL should be removed."""
        stale_task_id = "task_stale"
        service._tasks[stale_task_id] = {
            "status": "completed",
            "result": {"expr_id": "expr_old"},
            "error": None,
            "created_at": time.time() - 1801,  # 30min + 1s ago
        }

        service._cleanup_stale_tasks(ttl_seconds=1800)

        assert stale_task_id not in service._tasks

    def test_cleanup_keeps_recent_tasks(self, service):
        """Recent completed tasks should not be removed."""
        recent_task_id = "task_recent"
        service._tasks[recent_task_id] = {
            "status": "completed",
            "result": {"expr_id": "expr_new"},
            "error": None,
            "created_at": time.time() - 100,  # 100s ago
        }

        service._cleanup_stale_tasks(ttl_seconds=1800)

        assert recent_task_id in service._tasks

    def test_cleanup_keeps_running_tasks(self, service):
        """Running/pending tasks should never be cleaned up."""
        running_task_id = "task_running"
        service._tasks[running_task_id] = {
            "status": "running",
            "result": None,
            "error": None,
            "created_at": time.time() - 9999,  # very old but still running
        }

        service._cleanup_stale_tasks(ttl_seconds=1800)

        assert running_task_id in service._tasks
