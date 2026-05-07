import numpy as np
import pandas as pd
import pytest
import sympy

from app.ml.symbolic_regressor import (
    generate_interaction_features,
    extract_best_equation,
    extract_pareto_equations,
)


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
    assert X_aug.shape[1] == 9  # 5 original + 2 pairs * 2 features each
    assert len(aug_names) == 9


def test_interaction_features_names(interaction_data):
    X, names, pairs = interaction_data
    _, aug_names = generate_interaction_features(X, names, pairs)
    assert "f0_mul_f1" in aug_names
    assert "f0_div_f1" in aug_names


def test_interaction_features_values(interaction_data):
    X, names, pairs = interaction_data
    X_aug, _ = generate_interaction_features(X, names, pairs)
    # Column 5 (index 5) should be f0 * f1 since we have 5 original cols
    np.testing.assert_allclose(X_aug[:, 5], X[:, 0] * X[:, 1], rtol=1e-5)


def test_interaction_features_no_pairs():
    X = np.ones((10, 3), dtype=np.float32)
    X_aug, names = generate_interaction_features(X, ["a", "b", "c"], [])
    assert X_aug.shape == (10, 3)
    assert names == ["a", "b", "c"]


class MockPySRModel:
    def __init__(self):
        x0, x1, x2 = sympy.symbols("x0 x1 x2")
        self._exprs = [x0, x0 + x1, x0 * x1 + x2]
        self._latexs = ["x_{0}", "x_{0} + x_{1}", "x_{0} x_{1} + x_{2}"]
        self.equations_ = pd.DataFrame({
            "complexity": [1, 3, 7],
            "loss": [10.0, 3.0, 0.5],
            "equation": ["x0", "(x0 + x1)", "((x0 * x1) + x2)"],
            "score": [0.0, 1.2, 2.5],
            "pick": [False, False, True],
        })

    def sympy(self, index=None):
        return self._exprs[2] if index is None else self._exprs[index]

    def latex(self, index=None, precision=3):
        return self._latexs[2] if index is None else self._latexs[index]

    def score(self, X, y):
        return 0.85


def test_extract_best_equation():
    model = MockPySRModel()
    result = extract_best_equation(model)
    assert result["complexity"] == 7
    assert isinstance(result["sympy_expr"], sympy.Basic)


def test_extract_pareto_equations():
    model = MockPySRModel()
    results = extract_pareto_equations(model)
    assert len(results) == 3
    assert results[0]["complexity"] == 1
