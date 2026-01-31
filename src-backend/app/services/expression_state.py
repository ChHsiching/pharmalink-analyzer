"""Expression state management: dataclass, in-memory store, undo/redo history."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import sympy

from app.exceptions import ExpressionNotFoundError, UndoLimitError

logger = logging.getLogger(__name__)


def _serialize_sympy(expr: sympy.Basic) -> str:
    """Convert a SymPy expression to its string representation."""
    return str(expr)


def _deserialize_sympy(s: str) -> sympy.Basic:
    """Convert a string back to a SymPy expression."""
    return sympy.sympify(s)


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
    indicators: dict[str, float] = field(default_factory=dict)
    target_name: str = ""
    history: list[dict] = field(default_factory=list)
    history_index: int = -1

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict with SymPy expressions as strings."""
        return {
            "expr_id": self.expr_id,
            "model_id": self.model_id,
            "current_sympy": _serialize_sympy(self.current_sympy),
            "current_latex": self.current_latex,
            "current_complexity": self.current_complexity,
            "current_r2": self.current_r2,
            "pareto_equations": [
                {**eq, "sympy_expr": _serialize_sympy(eq["sympy_expr"])}
                for eq in self.pareto_equations
            ],
            "variable_impact": self.variable_impact,
            "indicators": self.indicators,
            "target_name": self.target_name,
            "history": [
                {**h, "sympy": _serialize_sympy(h["sympy"])}
                for h in self.history
            ],
            "history_index": self.history_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExpressionState":
        """Deserialize from a dict, converting strings back to SymPy expressions."""
        return cls(
            expr_id=data["expr_id"],
            model_id=data["model_id"],
            current_sympy=_deserialize_sympy(data["current_sympy"]),
            current_latex=data["current_latex"],
            current_complexity=data["current_complexity"],
            current_r2=data["current_r2"],
            pareto_equations=[
                {**eq, "sympy_expr": _deserialize_sympy(eq["sympy_expr"])}
                for eq in data.get("pareto_equations", [])
            ],
            variable_impact=data.get("variable_impact", {}),
            indicators=data.get("indicators", {}),
            target_name=data.get("target_name", ""),
            history=[
                {**h, "sympy": _deserialize_sympy(h["sympy"])}
                for h in data.get("history", [])
            ],
            history_index=data.get("history_index", -1),
        )


class ExpressionStateManager:
    """In-memory store for expression states with undo/redo history.

    When checkpoint_dir is provided, state is persisted to
    ``checkpoint_dir/{model_id}/expression_state.json`` on every mutation
    and loaded from disk on startup.  Omitting checkpoint_dir gives the
    original pure in-memory behaviour.
    """

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self._states: dict[str, ExpressionState] = {}
        self._checkpoint_dir: Path | None = checkpoint_dir
        if self._checkpoint_dir is not None:
            self._load_from_disk()

    # -- persistence helpers ------------------------------------------------

    def _state_file(self, state: ExpressionState) -> Path:
        return self._checkpoint_dir / state.model_id / "expression_state.json"

    def _save_to_disk(self, state: ExpressionState) -> None:
        """Write a single state to its JSON file.  Errors are logged, not raised."""
        if self._checkpoint_dir is None:
            return
        try:
            path = self._state_file(state)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state.to_dict(), indent=2))
        except Exception:
            logger.exception("Failed to save expression state %s", state.expr_id)

    def _load_from_disk(self) -> None:
        """Scan checkpoint_dir for persisted states and load them."""
        if self._checkpoint_dir is None:
            return
        if not self._checkpoint_dir.is_dir():
            return
        for state_file in self._checkpoint_dir.rglob("expression_state.json"):
            try:
                data = json.loads(state_file.read_text())
                state = ExpressionState.from_dict(data)
                self._states[state.expr_id] = state
            except Exception:
                logger.exception("Failed to load expression state from %s", state_file)

    # -- public API ---------------------------------------------------------

    def put(self, state: ExpressionState) -> None:
        """Store an expression state."""
        self._states[state.expr_id] = state
        self._save_to_disk(state)

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
        self._save_to_disk(state)

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
        self._save_to_disk(state)
        return state

    def get_history_entries(self, expr_id: str) -> tuple[list[dict], int]:
        """Return (history_entries, current_index). Raises ExpressionNotFoundError."""
        state = self.get(expr_id)
        return state.history, state.history_index
