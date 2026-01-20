"""Tests for PySRResultAdapter — validates schema isolation."""
import numpy as np
import pandas as pd
import sympy

from app.ml.pysr_adapter import PySRResultAdapter


def _make_mock_model(*, with_pick=False):
    """Create a minimal mock PySR model for adapter tests."""
    x0, x1 = sympy.symbols("x0 x1")
    best_expr = x0 * x1
    best_latex = "x_{0} x_{1}"

    data = {
        "complexity": [1, 3],
        "loss": [5.0, 0.2],
        "equation": ["x0", "(x0 * x1)"],
        "score": [0.0, 2.5],
    }
    if with_pick:
        data["pick"] = [False, True]

    model = type("MockPySR", (), {
        "equations_": pd.DataFrame(data),
    })()

    exprs = [x0, best_expr]
    latexs = ["x_{0}", best_latex]

    model.sympy = lambda index=None, _e=exprs: _e[1] if index is None else _e[index]
    model.latex = lambda index=None, precision=3, _l=latexs: _l[1] if index is None else _l[index]
    model.score = lambda X, y: 0.92
    return model


class TestGetBestEquation:
    def test_without_pick_uses_score_idxmax(self):
        """When no 'pick' column, adapter selects row with highest score."""
        model = _make_mock_model(with_pick=False)
        adapter = PySRResultAdapter(model)
        result = adapter.get_best_equation()

        assert result["complexity"] == 3
        assert result["loss"] == 0.2
        assert result["sympy_expr"] == model.sympy()
        assert isinstance(result["latex"], str)

    def test_with_pick_uses_pick_column(self):
        """When 'pick' column exists, adapter uses it for backward compat."""
        model = _make_mock_model(with_pick=True)
        adapter = PySRResultAdapter(model)
        result = adapter.get_best_equation()

        assert result["complexity"] == 3
        assert result["loss"] == 0.2


class TestGetParetoEquations:
    def test_returns_all_rows(self):
        model = _make_mock_model(with_pick=False)
        adapter = PySRResultAdapter(model)
        results = adapter.get_pareto_equations()

        assert len(results) == 2
        assert results[0]["index"] == 0
        assert results[0]["complexity"] == 1
        assert results[1]["index"] == 1
        assert results[1]["complexity"] == 3
