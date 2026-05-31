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


# ---------------------------------------------------------------------------
# TestStateManagerPersistence
# ---------------------------------------------------------------------------


class TestStateManagerPersistence:

    def test_put_saves_to_disk(self, tmp_path):
        """put() should write expression_state.json"""
        mgr = ExpressionStateManager(checkpoint_dir=tmp_path)
        state = _make_state()
        mgr.put(state)
        assert (tmp_path / "model_1" / "expression_state.json").exists()

    def test_load_restores_from_disk(self, tmp_path):
        """Fresh manager should load state from disk"""
        mgr1 = ExpressionStateManager(checkpoint_dir=tmp_path)
        state = _make_state()
        mgr1.put(state)

        mgr2 = ExpressionStateManager(checkpoint_dir=tmp_path)
        restored = mgr2.get("expr_test")
        assert restored.expr_id == "expr_test"
        assert restored.current_latex == "x"

    def test_get_raises_not_found_for_missing(self, tmp_path):
        mgr = ExpressionStateManager(checkpoint_dir=tmp_path)
        with pytest.raises(ExpressionNotFoundError):
            mgr.get("nonexistent")

    def test_history_preserved_across_restart(self, tmp_path):
        mgr1 = ExpressionStateManager(checkpoint_dir=tmp_path)
        state = _make_state_with_history()
        mgr1.put(state)

        mgr2 = ExpressionStateManager(checkpoint_dir=tmp_path)
        restored = mgr2.get("expr_test")
        assert len(restored.history) == 2

    def test_sympy_round_trip(self, tmp_path):
        """Complex SymPy expressions should round-trip through str/sympify"""
        mgr1 = ExpressionStateManager(checkpoint_dir=tmp_path)
        a, b = sympy.symbols("a b")
        expr = a**2 + b * 3.5
        state = ExpressionState(
            expr_id="expr_rt",
            model_id="model_1",
            current_sympy=expr,
            current_latex=sympy.latex(expr),
            current_complexity=5,
            current_r2=0.8,
        )
        mgr1.put(state)

        mgr2 = ExpressionStateManager(checkpoint_dir=tmp_path)
        restored = mgr2.get("expr_rt")
        assert sympy.simplify(restored.current_sympy - expr) == 0

    def test_undo_persists_to_disk(self, tmp_path):
        """undo() should persist the restored state"""
        mgr1 = ExpressionStateManager(checkpoint_dir=tmp_path)
        state = _make_state_with_history()
        mgr1.put(state)
        mgr1.undo("expr_test", steps=1)

        mgr2 = ExpressionStateManager(checkpoint_dir=tmp_path)
        restored = mgr2.get("expr_test")
        assert restored.history_index == 0
        assert restored.current_latex == "A"

    def test_backward_compatible_no_checkpoint_dir(self):
        """ExpressionStateManager() with no args must still work"""
        mgr = ExpressionStateManager()
        state = _make_state()
        mgr.put(state)
        assert mgr.get("expr_test") is state


class TestStateManagerListByModel:

    def test_filters_by_model_id(self):
        mgr = ExpressionStateManager()
        s1 = _make_state("expr_1")
        s1.model_id = "model_A"
        s2 = _make_state("expr_2")
        s2.model_id = "model_B"
        s3 = _make_state("expr_3")
        s3.model_id = "model_A"
        mgr.put(s1)
        mgr.put(s2)
        mgr.put(s3)

        result = mgr.list_by_model("model_A")
        assert len(result) == 2
        ids = {r.expr_id for r in result}
        assert ids == {"expr_1", "expr_3"}

    def test_returns_empty_list_for_unknown_model(self):
        mgr = ExpressionStateManager()
        mgr.put(_make_state())
        result = mgr.list_by_model("nonexistent")
        assert result == []

    def test_returns_empty_list_when_no_states(self):
        mgr = ExpressionStateManager()
        result = mgr.list_by_model("any")
        assert result == []


# ---------------------------------------------------------------------------
# Training data storage + serialization
# ---------------------------------------------------------------------------


class TestTrainingDataStorage:
    def test_state_stores_training_data(self):
        state = ExpressionState(
            expr_id="expr_test",
            model_id="model_test",
            current_sympy=sympy.Symbol("A"),
            current_latex="A",
            current_complexity=1,
            current_r2=0.95,
            X_train=[[1.0, 2.0], [3.0, 4.0]],
            X_test=[[5.0, 6.0]],
            y_train=[1.0, 2.0],
            y_test=[3.0],
            aug_names=["A", "B"],
            X_raw=[[1.0, 2.0]],
            pairs_raw=[{"source": "A", "target": "B"}],
            attention_matrix=[[0.5, 0.5], [0.5, 0.5]],
            feature_names=["A", "B"],
        )
        assert state.X_train == [[1.0, 2.0], [3.0, 4.0]]
        assert state.aug_names == ["A", "B"]

    def test_state_from_dict_backward_compatible(self):
        old_data = {
            "expr_id": "expr_old",
            "model_id": "model_old",
            "current_sympy": "A",
            "current_latex": "A",
            "current_complexity": 1,
            "current_r2": 0.9,
        }
        state = ExpressionState.from_dict(old_data)
        assert state.expr_id == "expr_old"
        assert state.X_train == []
        assert state.aug_names == []
        assert state.feature_names == []

    def test_state_round_trip_with_training_data(self):
        original = ExpressionState(
            expr_id="expr_rt",
            model_id="model_rt",
            current_sympy=sympy.Symbol("A"),
            current_latex="A",
            current_complexity=1,
            current_r2=0.95,
            X_train=[[1.0, 2.0]],
            X_test=[[3.0, 4.0]],
            y_train=[1.0],
            y_test=[2.0],
            aug_names=["A", "B"],
            X_raw=[[5.0, 6.0]],
            pairs_raw=[{"source": "A", "target": "B"}],
            attention_matrix=[[0.5, 0.5]],
            feature_names=["A"],
        )
        data = original.to_dict()
        restored = ExpressionState.from_dict(data)
        assert restored.X_train == [[1.0, 2.0]]
        assert restored.aug_names == ["A", "B"]
        assert restored.pairs_raw == [{"source": "A", "target": "B"}]
