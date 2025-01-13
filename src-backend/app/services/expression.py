import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy

from app.exceptions import DomainError, ExpressionNotFoundError, ExpressionTaskNotFoundError, SymbolicRegressionError, UndoLimitError

logger = logging.getLogger(__name__)

from app.config import CHECKPOINT_DIR
from app.ml.attention_extractor import extract_attention_weights, extract_top_pairs
from app.ml.expression_tree import expr_to_latex, get_complexity, simplify_expr, sympy_to_tree
from app.ml.symbolic_regressor import (
    extract_best_equation,
    extract_pareto_equations,
    generate_interaction_features,
    run_symbolic_regression,
)
from app.models.expression import (
    ExpressionHistoryEntry,
    ExpressionHistoryResponse,
    ExpressionResponse,
    TaskStatusResponse,
)
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.data_loader import data_loader as _default_data_loader


@dataclass
class _ExpressionState:
    expr_id: str
    model_id: str
    current_sympy: sympy.Basic
    current_latex: str
    current_complexity: int
    current_r2: float
    pareto_equations: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    history_index: int = -1


class ExpressionService:
    def __init__(self, resolver: CheckpointResolver):
        self._resolver = resolver
        self._states: dict[str, _ExpressionState] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self._tasks_lock = threading.Lock()

    def _to_response(self, state: _ExpressionState) -> ExpressionResponse:
        return ExpressionResponse(
            expr_id=state.expr_id,
            model_id=state.model_id,
            latex=state.current_latex,
            complexity=state.current_complexity,
            r2_score=state.current_r2,
            tree=sympy_to_tree(state.current_sympy),
        )

    def _push_history(self, state: _ExpressionState, operation: str):
        entry = {
            "operation": operation,
            "sympy": state.current_sympy,
            "latex": state.current_latex,
            "complexity": state.current_complexity,
            "r2": state.current_r2,
        }
        state.history = state.history[: state.history_index + 1]
        state.history.append(entry)
        state.history_index = len(state.history) - 1

    def generate(self, model_id: str, top_k: int = 10) -> ExpressionResponse:
        cp_dir, config = self._resolver.resolve(model_id)

        try:
            X, y, feature_names = self._resolver.get_features_with_target(config.dataset_id)
            matrix = extract_attention_weights(cp_dir, X, config)
            pairs_raw, _ = extract_top_pairs(matrix, feature_names, top_k)
            X_aug, aug_names = generate_interaction_features(X, feature_names, pairs_raw)
        except Exception as e:
            if isinstance(e, DomainError):
                raise
            raise SymbolicRegressionError(f"Expression pipeline failed: {e}") from e

        model = run_symbolic_regression(X_aug, y, aug_names)

        try:
            best = extract_best_equation(model)
            pareto = extract_pareto_equations(model)
            r2 = float(model.score(X_aug, y))
        except Exception as e:
            raise SymbolicRegressionError(f"Equation extraction failed: {e}") from e

        expr_id = f"expr_{uuid.uuid4().hex[:8]}"
        state = _ExpressionState(
            expr_id=expr_id,
            model_id=model_id,
            current_sympy=best["sympy_expr"],
            current_latex=best["latex"],
            current_complexity=best["complexity"],
            current_r2=r2,
            pareto_equations=pareto,
        )
        self._push_history(state, "generate")
        self._states[expr_id] = state
        return self._to_response(state)

    def start_generate(self, model_id: str, top_k: int = 10) -> TaskStatusResponse:
        """Start async expression generation, returning a task_id immediately."""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        with self._tasks_lock:
            self._tasks[task_id] = {
                "status": "pending",
                "result": None,
                "error": None,
                "created_at": time.time(),
            }

        thread = threading.Thread(
            target=self._run_generate,
            args=(task_id, model_id, top_k),
            daemon=True,
        )
        thread.start()
        return TaskStatusResponse(task_id=task_id, status="pending")

    def get_task_result(self, task_id: str) -> TaskStatusResponse:
        """Get the current status of an async generation task."""
        with self._tasks_lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise ExpressionTaskNotFoundError(task_id)
        return TaskStatusResponse(
            task_id=task_id,
            status=task["status"],
            result=task["result"],
            error=task["error"],
        )

    def _run_generate(self, task_id: str, model_id: str, top_k: int):
        """Background thread: run the generate pipeline and update task status."""
        with self._tasks_lock:
            self._tasks[task_id]["status"] = "running"
        try:
            response = self.generate(model_id, top_k)
            result_dict = response.model_dump()
            with self._tasks_lock:
                self._tasks[task_id]["status"] = "completed"
                self._tasks[task_id]["result"] = result_dict
        except Exception as e:
            with self._tasks_lock:
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["error"] = str(e)
        finally:
            self._cleanup_stale_tasks()

    def _cleanup_stale_tasks(self, ttl_seconds: int = 1800):
        """Remove completed/failed tasks older than TTL."""
        now = time.time()
        with self._tasks_lock:
            stale = [
                tid for tid, task in self._tasks.items()
                if task["status"] in ("completed", "failed")
                and now - task["created_at"] > ttl_seconds
            ]
            for tid in stale:
                del self._tasks[tid]
                logger.info("Cleaned up stale task %s", tid)

    def simplify(self, expr_id: str) -> ExpressionResponse:
        state = self._states.get(expr_id)
        if state is None:
            raise ExpressionNotFoundError(expr_id)
        simplified = simplify_expr(state.current_sympy)
        state.current_sympy = simplified
        state.current_latex = expr_to_latex(simplified)
        state.current_complexity = get_complexity(simplified)
        self._push_history(state, "simplify")
        return self._to_response(state)

    def optimize(self, expr_id: str) -> ExpressionResponse:
        state = self._states.get(expr_id)
        if state is None:
            raise ExpressionNotFoundError(expr_id)

        current_idx = 0
        for i, eq in enumerate(state.pareto_equations):
            if eq["complexity"] >= state.current_complexity:
                current_idx = i
                break

        next_idx = max(0, current_idx - 1)
        if next_idx == current_idx:
            return self._to_response(state)

        chosen = state.pareto_equations[next_idx]
        state.current_sympy = chosen["sympy_expr"]
        state.current_latex = chosen["latex"]
        state.current_complexity = chosen["complexity"]
        self._push_history(state, "optimize")
        return self._to_response(state)

    def get_tree(self, expr_id: str) -> ExpressionResponse:
        state = self._states.get(expr_id)
        if state is None:
            raise ExpressionNotFoundError(expr_id)
        return self._to_response(state)

    def get_history(self, expr_id: str) -> ExpressionHistoryResponse:
        state = self._states.get(expr_id)
        if state is None:
            raise ExpressionNotFoundError(expr_id)
        entries = [
            ExpressionHistoryEntry(
                operation=h["operation"],
                latex=h["latex"],
                complexity=h["complexity"],
                r2_score=h["r2"],
            )
            for h in state.history
        ]
        return ExpressionHistoryResponse(
            expr_id=state.expr_id,
            history=entries,
            current_index=state.history_index,
        )

    def undo(self, expr_id: str, steps: int = 1) -> ExpressionResponse:
        state = self._states.get(expr_id)
        if state is None:
            raise ExpressionNotFoundError(expr_id)
        target = max(0, min(state.history_index - steps, len(state.history) - 1))
        if target == state.history_index:
            raise UndoLimitError()
        h = state.history[target]
        state.history_index = target
        state.current_sympy = h["sympy"]
        state.current_latex = h["latex"]
        state.current_complexity = h["complexity"]
        state.current_r2 = h["r2"]
        return self._to_response(state)
