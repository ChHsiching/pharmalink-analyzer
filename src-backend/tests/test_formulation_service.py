"""Tests for FormulationService."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import sympy
import torch

from app.exceptions import ExpressionNotFoundError
from app.services.expression_state import ExpressionState
from app.services.formulation import FormulationService


def _fake_state():
    x0, x1 = sympy.symbols("X0 X1")
    return ExpressionState(
        expr_id="expr_test", model_id="model_test",
        current_sympy=x0 + x1, current_latex="X_{0} + X_{1}",
        current_complexity=1, current_r2=0.95, pareto_equations=[],
    )


@pytest.fixture
def mock_deps():
    state_manager = MagicMock()
    state_manager.get.return_value = _fake_state()

    resolver = MagicMock()
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    resolver.get_features.return_value = (features, ["X0", "X1"])
    config_mock = MagicMock()
    config_mock.n_features = 2
    config_mock.dataset_id = "ds_test"
    resolver.resolve.return_value = (MagicMock(), config_mock)

    with patch("app.services.formulation.load_model") as mock_load, \
         patch("app.services.formulation.extract_top_pairs") as mock_pairs:
        mock_model = MagicMock()
        mock_attn = torch.ones(1, 2, 2) * 0.5
        mock_model.return_value = (torch.ones(2), mock_attn)
        mock_load.return_value = mock_model
        mock_pairs.return_value = (
            [{"source": "X0", "target": "X1", "weight": 0.8, "classification": "synergistic"}], 50.0
        )
        service = FormulationService(state_manager=state_manager, resolver=resolver)
        yield service


def test_formulate_returns_candidates(mock_deps):
    result = mock_deps.formulate("expr_test", top_k=3, n_samples=100)
    assert result.expr_id == "expr_test"
    assert len(result.candidates) > 0
    assert len(result.candidates) <= 3
    assert result.feature_names == ["X0", "X1"]
    scores = [c.predicted_response for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_formulate_expression_not_found(mock_deps):
    mock_deps._state_manager.get.side_effect = ExpressionNotFoundError("expr_bad")
    with pytest.raises(ExpressionNotFoundError):
        mock_deps.formulate("expr_bad")


def test_formulate_response_has_attention_weak_field(mock_deps):
    """FormulationResponse must include attention_weak boolean flag."""
    result = mock_deps.formulate("expr_test", top_k=3, n_samples=100)
    assert hasattr(result, "attention_weak")
    assert isinstance(result.attention_weak, bool)
