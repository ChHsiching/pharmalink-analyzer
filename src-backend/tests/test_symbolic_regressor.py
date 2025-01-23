"""Tests for symbolic regression — gplearn backend."""
import sys

import numpy as np
import pytest
import sympy
from unittest.mock import MagicMock, patch

from app.ml.symbolic_regressor import (
    generate_interaction_features,
    run_symbolic_regression,
    extract_best_equation,
    extract_pareto_equations,
    parse_gplearn_program,
)
from app.exceptions import SymbolicRegressionError


@pytest.fixture
def interaction_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5)).astype(np.float32)
    names = ["f0", "f1", "f2", "f3", "f4"]
    pairs = [
        {"source": "f0", "target": "f1", "weight": 0.8},
        {"source": "f2", "target": "f3", "weight": 0.6},
    ]
    return X, names, pairs


def test_interaction_features_shape(interaction_data):
    X, names, pairs = interaction_data
    X_aug, aug_names = generate_interaction_features(X, names, pairs)
    assert X_aug.shape[0] == 30
    assert X_aug.shape[1] == 9
    assert len(aug_names) == 9


def test_interaction_features_names(interaction_data):
    X, names, pairs = interaction_data
    _, aug_names = generate_interaction_features(X, names, pairs)
    assert "f0_mul_f1" in aug_names
    assert "f0_div_f1" in aug_names


def test_interaction_features_values(interaction_data):
    X, names, pairs = interaction_data
    X_aug, _ = generate_interaction_features(X, names, pairs)
    np.testing.assert_allclose(X_aug[:, 5], X[:, 0] * X[:, 1], rtol=1e-5)


def test_interaction_features_no_pairs():
    X = np.ones((10, 3), dtype=np.float32)
    X_aug, names = generate_interaction_features(X, ["a", "b", "c"], [])
    assert X_aug.shape == (10, 3)
    assert names == ["a", "b", "c"]


class _MockProgram:
    def __init__(self, s: str):
        self._s = s
    def __str__(self):
        return self._s


def _make_mock_model(program_str="add(X0, X1)", feature_names=None, score=0.85):
    model = MagicMock()
    model._program = _MockProgram(program_str)
    model._feature_names = feature_names or ["A", "B"]
    model._training_score = score
    model.score.return_value = score
    return model


def test_run_fits_model():
    X = np.zeros((10, 3), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    mock_cls = MagicMock()
    mock_model = MagicMock()
    mock_model.score.return_value = 0.9
    mock_cls.return_value = mock_model
    with patch("gplearn.genetic.SymbolicRegressor", mock_cls):
        result = run_symbolic_regression(X, y, ["A", "B", "C"])
    mock_model.fit.assert_called_once_with(X, y)
    assert result._feature_names == ["A", "B", "C"]
    assert result._training_score == 0.9


def test_gplearn_missing_raises_error():
    X = np.zeros((10, 3), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    with patch.dict(sys.modules, {"gplearn": None, "gplearn.genetic": None}):
        with pytest.raises(SymbolicRegressionError, match="gplearn"):
            run_symbolic_regression(X, y, ["A", "B", "C"])


def test_fit_failure_raises_error():
    X = np.zeros((10, 3), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    mock_cls = MagicMock()
    mock_cls.return_value.fit.side_effect = ValueError("Bad data shape")
    with patch("gplearn.genetic.SymbolicRegressor", mock_cls):
        with pytest.raises(SymbolicRegressionError, match="failed"):
            run_symbolic_regression(X, y, ["A", "B", "C"])


def test_extract_best_returns_correct_shape():
    model = _make_mock_model("add(X0, X1)", ["A", "B"])
    result = extract_best_equation(model)
    assert isinstance(result["sympy_expr"], sympy.Basic)
    assert isinstance(result["latex"], str)
    assert isinstance(result["complexity"], int)
    assert isinstance(result["loss"], float)


def test_extract_best_sympy_expression():
    model = _make_mock_model("add(X0, X1)", ["A", "B"])
    result = extract_best_equation(model)
    a, b = sympy.symbols("A B")
    assert result["sympy_expr"] == a + b


def test_extract_best_loss_is_one_minus_r2():
    model = _make_mock_model("add(X0, X1)", ["A", "B"], score=0.75)
    result = extract_best_equation(model)
    assert result["loss"] == pytest.approx(0.25)


def test_pareto_returns_single_element():
    model = _make_mock_model("add(X0, X1)", ["A", "B"])
    results = extract_pareto_equations(model)
    assert len(results) == 1


def test_pareto_has_index_zero():
    model = _make_mock_model("add(X0, X1)", ["A", "B"])
    results = extract_pareto_equations(model)
    assert results[0]["index"] == 0


def test_pareto_matches_best():
    model = _make_mock_model("add(X0, X1)", ["A", "B"])
    best = extract_best_equation(model)
    pareto = extract_pareto_equations(model)
    assert pareto[0]["sympy_expr"] == best["sympy_expr"]
    assert pareto[0]["complexity"] == best["complexity"]
