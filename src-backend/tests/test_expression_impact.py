"""Unit tests for compute_impact() variable influence scoring."""

import numpy as np
import sympy

from app.ml.expression_impact import compute_impact


def _make_test_setup():
    """Shared test fixtures: feature_names, X, attention, pairs, aug_names."""
    feature_names = ["A", "B"]
    X = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    # Uniform attention matrix: all off-diagonal = 0.5
    attention_matrix = np.full((2, 2), 0.5, dtype=np.float32)
    np.fill_diagonal(attention_matrix, 1.0)
    pairs_raw: list[dict] = []
    aug_names = ["A", "B"]
    return feature_names, X, attention_matrix, pairs_raw, aug_names


def test_linear_expression_equal_impact():
    """A + B with uniform attention => both impacts roughly equal."""
    feature_names, X, attention_matrix, pairs_raw, aug_names = _make_test_setup()
    expr = sympy.Symbol("A") + sympy.Symbol("B")

    result = compute_impact(expr, feature_names, X, pairs_raw, attention_matrix, aug_names)

    # Both variables should have roughly equal impact
    assert "A" in result
    assert "B" in result
    assert abs(result["A"] - result["B"]) < 0.05


def test_weighted_expression_higher_impact():
    """2*A + B => A has higher impact than B, A normalized to 1.0."""
    feature_names, X, attention_matrix, pairs_raw, aug_names = _make_test_setup()
    expr = 2 * sympy.Symbol("A") + sympy.Symbol("B")

    result = compute_impact(expr, feature_names, X, pairs_raw, attention_matrix, aug_names)

    assert result["A"] > result["B"]
    assert abs(result["A"] - 1.0) < 0.01


def test_missing_variable_impact_zero():
    """Expression only uses A => B should have impact 0.0."""
    feature_names, X, attention_matrix, pairs_raw, aug_names = _make_test_setup()
    expr = sympy.Symbol("A")

    result = compute_impact(expr, feature_names, X, pairs_raw, attention_matrix, aug_names)

    assert result["A"] == 1.0
    assert result["B"] == 0.0


def test_constant_expression_no_division_by_zero():
    """Constant expression (no free symbols) => all impacts 0.0, no division by zero."""
    feature_names, X, attention_matrix, pairs_raw, aug_names = _make_test_setup()
    expr = sympy.Float(5.0)

    result = compute_impact(expr, feature_names, X, pairs_raw, attention_matrix, aug_names)

    assert result["A"] == 0.0
    assert result["B"] == 0.0


def test_normalization_max_is_one():
    """3*A + B => max impact == 1.0 after normalization."""
    feature_names, X, attention_matrix, pairs_raw, aug_names = _make_test_setup()
    expr = 3 * sympy.Symbol("A") + sympy.Symbol("B")

    result = compute_impact(expr, feature_names, X, pairs_raw, attention_matrix, aug_names)

    max_val = max(result.values())
    assert abs(max_val - 1.0) < 1e-6
    assert result["A"] > result["B"]
