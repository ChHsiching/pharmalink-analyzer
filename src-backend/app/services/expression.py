import uuid
from pathlib import Path

import numpy as np
from fastapi import HTTPException

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
)
from app.services.data_loader import data_loader as _default_data_loader


class _ExpressionState:
    __slots__ = (
        "expr_id",
        "model_id",
        "current_sympy",
        "current_latex",
        "current_complexity",
        "current_r2",
        "pareto_equations",
        "history",
        "history_index",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ExpressionService:
    def __init__(self, data_loader=None, checkpoint_dir=None):
        self._data_loader = data_loader or _default_data_loader
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        self._states: dict[str, _ExpressionState] = {}

    def _resolve_checkpoint(self, model_id: str):
        cp_dir = self._checkpoint_dir / model_id
        if not cp_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {model_id}")
        config_path = cp_dir / "config.json"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config not found: {model_id}")
        from app.models.training import TrainingConfig

        config = TrainingConfig.model_validate_json(config_path.read_text())
        return cp_dir, config

    def _get_features(self, dataset_id: str):
        dataset = self._data_loader.get_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        df = self._data_loader.get_dataframe(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset data not found: {dataset_id}")
        feature_names = dataset.feature_names
        X = df[feature_names].to_numpy().astype(np.float32)
        y = df[dataset.target].to_numpy().astype(np.float32)
        return X, y, feature_names

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
        cp_dir, config = self._resolve_checkpoint(model_id)
        X, y, feature_names = self._get_features(config.dataset_id)

        matrix = extract_attention_weights(cp_dir, X)
        pairs_raw, _ = extract_top_pairs(matrix, feature_names, top_k)
        X_aug, aug_names = generate_interaction_features(X, feature_names, pairs_raw)
        model = run_symbolic_regression(X_aug, y, aug_names)

        best = extract_best_equation(model)
        pareto = extract_pareto_equations(model)
        r2 = float(model.score(X_aug, y))

        expr_id = f"expr_{uuid.uuid4().hex[:8]}"
        state = _ExpressionState(
            expr_id=expr_id,
            model_id=model_id,
            current_sympy=best["sympy_expr"],
            current_latex=best["latex"],
            current_complexity=best["complexity"],
            current_r2=r2,
            pareto_equations=pareto,
            history=[],
            history_index=-1,
        )
        self._push_history(state, "generate")
        self._states[expr_id] = state
        return self._to_response(state)

    def simplify(self, expr_id: str) -> ExpressionResponse:
        state = self._states.get(expr_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Expression not found: {expr_id}")
        simplified = simplify_expr(state.current_sympy)
        state.current_sympy = simplified
        state.current_latex = expr_to_latex(simplified)
        state.current_complexity = get_complexity(simplified)
        self._push_history(state, "simplify")
        return self._to_response(state)

    def optimize(self, expr_id: str) -> ExpressionResponse:
        state = self._states.get(expr_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Expression not found: {expr_id}")

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
            raise HTTPException(status_code=404, detail=f"Expression not found: {expr_id}")
        return self._to_response(state)

    def get_history(self, expr_id: str) -> ExpressionHistoryResponse:
        state = self._states.get(expr_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Expression not found: {expr_id}")
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
            raise HTTPException(status_code=404, detail=f"Expression not found: {expr_id}")
        target = max(0, min(state.history_index - steps, len(state.history) - 1))
        if target == state.history_index:
            raise HTTPException(status_code=400, detail="No more history to undo/redo")
        h = state.history[target]
        state.history_index = target
        state.current_sympy = h["sympy"]
        state.current_latex = h["latex"]
        state.current_complexity = h["complexity"]
        state.current_r2 = h["r2"]
        return self._to_response(state)


expression_service = ExpressionService()
