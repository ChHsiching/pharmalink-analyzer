"""ExpressionService — facade orchestrating pipeline, state, and async tasks."""

import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.exceptions import ExpressionTaskNotFoundError
from app.ml.expression_tree import compute_indicators_from_expr, expr_to_latex, get_complexity, simplify_expr, sympy_to_tree
from app.models.expression import (
    ExpressionHistoryEntry,
    ExpressionHistoryResponse,
    ExpressionResponse,
    TaskStatusResponse,
)
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.expression_pipeline import ExpressionPipeline
from app.services.expression_state import ExpressionState, ExpressionStateManager

logger = logging.getLogger(__name__)


class ExpressionService:
    def __init__(
        self,
        resolver: CheckpointResolver,
        pipeline: ExpressionPipeline | None = None,
        state_manager: ExpressionStateManager | None = None,
        checkpoint_dir: Path | None = None,
    ):
        self._resolver = resolver
        self._pipeline = pipeline or ExpressionPipeline(resolver)
        self._state = state_manager or ExpressionStateManager(checkpoint_dir=checkpoint_dir)
        self._tasks: dict[str, dict[str, Any]] = {}
        self._tasks_lock = threading.Lock()

    def _to_response(self, state: ExpressionState) -> ExpressionResponse:
        pareto_index = next(
            (i for i, eq in enumerate(state.pareto_equations)
             if eq["complexity"] == state.current_complexity),
            0,
        )
        return ExpressionResponse(
            expr_id=state.expr_id,
            model_id=state.model_id,
            latex=state.current_latex,
            complexity=state.current_complexity,
            r2_score=state.current_r2,
            tree=sympy_to_tree(state.current_sympy),
            pareto_count=len(state.pareto_equations),
            pareto_index=pareto_index,
            variable_impact=state.variable_impact,
            indicators=state.indicators,
            target_name=state.target_name,
        )

    def generate(self, model_id: str, top_k: int = 10, preset: str = "standard") -> ExpressionResponse:
        result = self._pipeline.run(model_id, top_k, preset=preset)

        expr_id = f"expr_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        state = ExpressionState(
            expr_id=expr_id,
            model_id=model_id,
            current_sympy=result.sympy_expr,
            current_latex=result.latex,
            current_complexity=result.complexity,
            current_r2=result.r2_score,
            pareto_equations=result.pareto_equations,
            variable_impact=result.variable_impact,
            indicators=result.indicators,
            target_name=result.target_name,
            X_train=result.X_train,
            X_test=result.X_test,
            y_train=result.y_train,
            y_test=result.y_test,
            aug_names=result.aug_names,
            X_raw=result.X_raw,
            pairs_raw=result.pairs_raw,
            attention_matrix=result.attention_matrix,
            feature_names=result.feature_names,
        )
        self._state.push_history(state, "generate")
        self._state.put(state)
        return self._to_response(state)

    def start_generate(self, model_id: str, top_k: int = 10, preset: str = "standard") -> TaskStatusResponse:
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
            args=(task_id, model_id, top_k, preset),
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

    def _run_generate(self, task_id: str, model_id: str, top_k: int, preset: str):
        """Background thread: run the generate pipeline and update task status."""
        with self._tasks_lock:
            self._tasks[task_id]["status"] = "running"
        try:
            response = self.generate(model_id, top_k, preset=preset)
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
        import numpy as np

        from app.ml.expression_impact import compute_impact

        state = self._state.get(expr_id)
        simplified = simplify_expr(state.current_sympy)
        state.current_sympy = simplified
        state.current_latex = expr_to_latex(simplified)
        state.current_complexity = get_complexity(simplified)

        if state.X_train and state.aug_names:
            state.indicators = compute_indicators_from_expr(
                simplified, state.X_train, state.X_test,
                state.y_train, state.y_test, state.aug_names,
            )
            state.current_r2 = state.indicators.get("test_r2", state.current_r2)

        if state.X_raw and state.feature_names:
            state.variable_impact = compute_impact(
                sympy_expr=simplified,
                feature_names=state.feature_names,
                X=np.array(state.X_raw),
                pairs_raw=state.pairs_raw,
                attention_matrix=np.array(state.attention_matrix),
                aug_names=state.aug_names,
            )

        self._state.push_history(state, "simplify")
        return self._to_response(state)

    def optimize(self, expr_id: str) -> ExpressionResponse:
        state = self._state.get(expr_id)

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
        state.current_r2 = chosen["r2_score"]
        self._state.push_history(state, "optimize")
        return self._to_response(state)

    def get_tree(self, expr_id: str) -> ExpressionResponse:
        state = self._state.get(expr_id)
        return self._to_response(state)

    def get_history(self, expr_id: str) -> ExpressionHistoryResponse:
        state = self._state.get(expr_id)
        return ExpressionHistoryResponse(
            expr_id=state.expr_id,
            history=[
                ExpressionHistoryEntry(
                    operation=h["operation"],
                    latex=h["latex"],
                    complexity=h["complexity"],
                    r2_score=h["r2"],
                )
                for h in state.history
            ],
            current_index=state.history_index,
        )

    def undo(self, expr_id: str, steps: int = 1) -> ExpressionResponse:
        state = self._state.undo(expr_id, steps)
        return self._to_response(state)

    def list_by_model(self, model_id: str) -> list[dict]:
        """Return [{expr_id, latex}] for all expressions matching model_id."""
        states = self._state.list_by_model(model_id)
        return [{"expr_id": s.expr_id, "latex": s.current_latex} for s in states]
