"""Expression state management: dataclass, in-memory store, undo/redo history."""

from dataclasses import dataclass, field

import sympy

from app.exceptions import ExpressionNotFoundError, UndoLimitError


@dataclass
class ExpressionState:
    """Immutable snapshot of an expression's current values and history."""

    expr_id: str
    model_id: str
    current_sympy: sympy.Basic
    current_latex: str
    current_complexity: int
    current_r2: float
    pareto_equations: list[dict] = field(default_factory=list)
    variable_impact: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    history_index: int = -1


class ExpressionStateManager:
    """In-memory store for expression states with undo/redo history."""

    def __init__(self) -> None:
        self._states: dict[str, ExpressionState] = {}

    def put(self, state: ExpressionState) -> None:
        """Store an expression state."""
        self._states[state.expr_id] = state

    def get(self, expr_id: str) -> ExpressionState:
        """Retrieve state by expr_id. Raises ExpressionNotFoundError if missing."""
        state = self._states.get(expr_id)
        if state is None:
            raise ExpressionNotFoundError(expr_id)
        return state

    def push_history(self, state: ExpressionState, operation: str) -> None:
        """Record current values as a history entry after an operation."""
        entry = {
            "operation": operation,
            "sympy": state.current_sympy,
            "latex": state.current_latex,
            "complexity": state.current_complexity,
            "r2": state.current_r2,
            "variable_impact": state.variable_impact,
        }
        state.history = state.history[: state.history_index + 1]
        state.history.append(entry)
        state.history_index = len(state.history) - 1

    def undo(self, expr_id: str, steps: int = 1) -> ExpressionState:
        """Restore state from history. Raises UndoLimitError or ExpressionNotFoundError."""
        state = self.get(expr_id)
        target = max(0, min(state.history_index - steps, len(state.history) - 1))
        if target == state.history_index:
            raise UndoLimitError()
        h = state.history[target]
        state.history_index = target
        state.current_sympy = h["sympy"]
        state.current_latex = h["latex"]
        state.current_complexity = h["complexity"]
        state.current_r2 = h["r2"]
        state.variable_impact = h.get("variable_impact", {})
        return state

    def get_history_entries(self, expr_id: str) -> tuple[list[dict], int]:
        """Return (history_entries, current_index). Raises ExpressionNotFoundError."""
        state = self.get(expr_id)
        return state.history, state.history_index
