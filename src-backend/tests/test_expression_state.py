"""Tests for ExpressionState dataclass and ExpressionStateManager."""

import dataclasses

import pytest
import sympy

from app.exceptions import ExpressionNotFoundError, UndoLimitError
from app.services.expression_state import ExpressionState, ExpressionStateManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(expr_id: str = "expr_test") -> ExpressionState:
    """Create a minimal ExpressionState."""
    return ExpressionState(
        expr_id=expr_id,
        model_id="model_1",
        current_sympy=sympy.Symbol("x"),
        current_latex="x",
        current_complexity=1,
        current_r2=0.9,
    )


def _make_state_with_history(expr_id: str = "expr_test") -> ExpressionState:
    """Create a state with two history entries ready for undo tests.

    1. push_history("generate")  -> latex="A"
    2. mutate latex to "A^{2}"
    3. push_history("simplify")  -> latex="A^{2}"
    """
    mgr = ExpressionStateManager()
    state = ExpressionState(
        expr_id=expr_id,
        model_id="model_1",
        current_sympy=sympy.Symbol("A"),
        current_latex="A",
        current_complexity=1,
        current_r2=0.9,
    )
    mgr.put(state)
    mgr.push_history(state, "generate")

    state.current_sympy = sympy.Pow(sympy.Symbol("A"), 2)
    state.current_latex = "A^{2}"
    state.current_complexity = 3
    state.current_r2 = 0.85
    mgr.push_history(state, "simplify")

    return state


# ---------------------------------------------------------------------------
# TestExpressionState
# ---------------------------------------------------------------------------


class TestExpressionState:

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(ExpressionState)

    def test_has_required_fields(self):
        state = ExpressionState(
            expr_id="e1",
            model_id="m1",
            current_sympy=sympy.Symbol("x"),
            current_latex="x",
            current_complexity=1,
            current_r2=0.5,
        )
        assert state.pareto_equations == []
        assert state.history == []
        assert state.history_index == -1


# ---------------------------------------------------------------------------
# TestStateManagerPutGet
# ---------------------------------------------------------------------------


class TestStateManagerPutGet:

    def test_put_and_get(self):
        mgr = ExpressionStateManager()
        state = _make_state()
        mgr.put(state)
        assert mgr.get("expr_test") is state

    def test_get_raises_not_found(self):
        mgr = ExpressionStateManager()
        with pytest.raises(ExpressionNotFoundError):
            mgr.get("no_such_id")


# ---------------------------------------------------------------------------
# TestStateManagerPushHistory
# ---------------------------------------------------------------------------


class TestStateManagerPushHistory:

    def test_records_entry(self):
        mgr = ExpressionStateManager()
        state = _make_state()
        mgr.put(state)
        mgr.push_history(state, "generate")

        assert len(state.history) == 1
        entry = state.history[0]
        assert entry["operation"] == "generate"
        assert entry["latex"] == "x"
        assert entry["complexity"] == 1
        assert entry["r2"] == 0.9
        assert state.history_index == 0

    def test_truncates_redo_stack_on_new_push(self):
        mgr = ExpressionStateManager()
        state = _make_state()
        mgr.put(state)

        # Push two entries
        mgr.push_history(state, "generate")
        state.current_latex = "B"
        mgr.push_history(state, "simplify")  # index=1

        # Undo back to index 0
        mgr.undo("expr_test", steps=1)  # now at index 0

        # Push a new entry — should truncate the redo stack
        state.current_latex = "C"
        mgr.push_history(state, "optimize")

        assert len(state.history) == 2
        assert state.history[0]["operation"] == "generate"
        assert state.history[1]["operation"] == "optimize"


# ---------------------------------------------------------------------------
# TestStateManagerUndo
# ---------------------------------------------------------------------------


class TestStateManagerUndo:

    def test_undo_restores_previous(self):
        mgr = ExpressionStateManager()
        state = _make_state_with_history()
        mgr.put(state)

        result = mgr.undo("expr_test", steps=1)

        assert result.current_latex == "A"
        assert result.history_index == 0

    def test_undo_raises_at_beginning(self):
        mgr = ExpressionStateManager()
        state = _make_state_with_history()
        mgr.put(state)

        # Undo to index 0
        mgr.undo("expr_test", steps=1)

        # Already at beginning — should raise
        with pytest.raises(UndoLimitError):
            mgr.undo("expr_test", steps=1)

    def test_undo_raises_not_found(self):
        mgr = ExpressionStateManager()
        with pytest.raises(ExpressionNotFoundError):
            mgr.undo("no_such_id")


# ---------------------------------------------------------------------------
# TestStateManagerGetHistory
# ---------------------------------------------------------------------------


class TestStateManagerGetHistory:

    def test_returns_entries_and_index(self):
        mgr = ExpressionStateManager()
        state = _make_state_with_history()
        mgr.put(state)

        entries, idx = mgr.get_history_entries("expr_test")

        assert len(entries) == 2
        assert idx == 1

    def test_raises_not_found(self):
        mgr = ExpressionStateManager()
        with pytest.raises(ExpressionNotFoundError):
            mgr.get_history_entries("no_such_id")
