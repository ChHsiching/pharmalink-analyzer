"""Correctness tests for symbolic regression and expression tree operations."""

import numpy as np
import pytest
import sympy

from app.ml.symbolic_regressor import (
    generate_interaction_features,
    parse_gplearn_program,
    run_symbolic_regression,
    extract_best_equation,
)
from app.ml.expression_tree import get_complexity, simplify_expr, sympy_to_tree, expr_to_latex


class TestExpressionTree:
    def test_variable(self):
        tree = sympy_to_tree(sympy.Symbol("X0"))
        assert tree["type"] == "variable"
        assert tree["value"] == "X0"

    def test_addition(self):
        x0, x1 = sympy.symbols("X0 X1")
        tree = sympy_to_tree(x0 + x1)
        assert tree["type"] == "operator"
        assert tree["value"] == "+"
        assert len(tree["children"]) == 2

    def test_complexity(self):
        x0, x1 = sympy.symbols("X0 X1")
        assert get_complexity(x0 + x1) == 1

    def test_simplify(self):
        x = sympy.Symbol("X0")
        assert simplify_expr(x + 0) == x

    def test_latex(self):
        x0, x1 = sympy.symbols("X0 X1")
        assert "X" in expr_to_latex(x0 + x1)


class TestInteractionFeatures:
    def test_generates_product_features(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_new, new_names = generate_interaction_features(X, ["X0", "X1"], [
            {"source": "X0", "target": "X1", "weight": 0.8, "classification": "synergistic"},
        ])
        assert X_new.shape[0] == 2
        assert "X0_mul_X1" in new_names


class TestParseGplearnProgram:
    def test_addition(self):
        x0, x1 = sympy.symbols("X0 X1")
        result = parse_gplearn_program("add(X0, X1)", ["X0", "X1"])
        assert result.equals(x0 + x1)

    def test_multiplication(self):
        x0, x1 = sympy.symbols("X0 X1")
        result = parse_gplearn_program("mul(X0, X1)", ["X0", "X1"])
        assert result.equals(x0 * x1)


class TestSymbolicRegression:
    def test_recovers_linear(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 2)
        y = X[:, 0] + X[:, 1]
        model = run_symbolic_regression(X, y, ["X0", "X1"], preset="quick")
        best = extract_best_equation(model)
        assert best["sympy_expr"] is not None
        assert best["loss"] < 0.15
