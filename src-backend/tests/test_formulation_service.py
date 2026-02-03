"""Tests for FormulationService."""

from unittest.mock import MagicMock
import numpy as np
import pytest
import sympy

from app.exceptions import ExpressionNotFoundError
from app.services.expression_state import ExpressionState
from app.services.formulation import FormulationService


def _fake_state(**overrides):
    x0, x1 = sympy.symbols("X0 X1")
    defaults = dict(
        expr_id="expr_test", model_id="model_test",
        current_sympy=x0 + x1, current_latex="X_{0} + X_{1}",
        current_complexity=1, current_r2=0.95, pareto_equations=[],
        variable_impact={"X0": 0.6, "X1": 0.4},
    )
    defaults.update(overrides)
    return ExpressionState(**defaults)


def _make_resolver(feature_names=None):
    feature_names = feature_names or ["X0", "X1"]
    resolver = MagicMock()
    features = np.random.rand(2, len(feature_names))
    resolver.get_features.return_value = (features, feature_names)
    config_mock = MagicMock()
    config_mock.dataset_id = "ds_test"
    resolver.resolve.return_value = (MagicMock(), config_mock)
    return resolver


def _make_service(state=None, resolver=None, **state_overrides):
    state = state or _fake_state(**state_overrides)
    state_manager = MagicMock()
    state_manager.get.return_value = state
    resolver = resolver or _make_resolver()
    return FormulationService(state_manager=state_manager, resolver=resolver)


def test_formulate_returns_candidates():
    service = _make_service()
    result = service.formulate("expr_test", top_k=3, n_samples=100)
    assert result.expr_id == "expr_test"
    assert len(result.candidates) > 0
    assert len(result.candidates) <= 3
    assert result.feature_names == ["X0", "X1"]
    scores = [c.predicted_response for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_formulate_expression_not_found():
    state_manager = MagicMock()
    state_manager.get.side_effect = ExpressionNotFoundError("expr_bad")
    resolver = MagicMock()
    service = FormulationService(state_manager=state_manager, resolver=resolver)
    with pytest.raises(ExpressionNotFoundError):
        service.formulate("expr_bad")


def test_formulate_response_has_attention_weak_field():
    service = _make_service()
    result = service.formulate("expr_test", top_k=3, n_samples=100)
    assert hasattr(result, "attention_weak")
    assert isinstance(result.attention_weak, bool)


def test_uniform_attention_triggers_rescue():
    """When all variable_impact values are identical (uniform), attention_weak=True."""
    service = _make_service(variable_impact={"X0": 0.5, "X1": 0.5})
    result = service.formulate("expr_test", top_k=3, n_samples=100)
    assert result.attention_weak is True


def test_differentiated_attention_no_rescue():
    """When max/min > 2.0, attention_weak=False and weights unchanged."""
    service = _make_service(variable_impact={"X0": 0.9, "X1": 0.1})
    result = service.formulate("expr_test", top_k=3, n_samples=100)
    assert result.attention_weak is False


def test_softmax_rescue_produces_valid_distribution():
    """Softmax output sums to 1.0 and preserves keys."""
    from app.services.formulation import _softmax_rescue

    uniform = {"X0": 0.5, "X1": 0.5, "X2": 0.5}
    result = _softmax_rescue(uniform, temperature=5.0)
    values = list(result.values())
    assert abs(sum(values) - 1.0) < 1e-6
    assert set(result.keys()) == {"X0", "X1", "X2"}


def test_softmax_rescue_amplifies_skewed_input():
    """Even with skewed input, softmax preserves order with high temperature."""
    from app.services.formulation import _softmax_rescue

    skewed = {"X0": 0.7, "X1": 0.2, "X2": 0.1}
    result = _softmax_rescue(skewed, temperature=5.0)
    assert result["X0"] > result["X1"] > result["X2"]


def test_formulate_uses_variable_impact():
    """FormulationService should use state.variable_impact instead of model attention."""
    service = _make_service(variable_impact={"X0": 0.8, "X1": 0.3})
    result = service.formulate("expr_test", top_k=3, n_samples=100)

    assert result.expr_id == "expr_test"
    assert len(result.candidates) > 0
    assert result.attention_weights == {"X0": 0.8, "X1": 0.3}


def test_formulate_variable_impact_empty_raises():
    """formulate should raise ValueError when variable_impact is empty."""
    service = _make_service(variable_impact={})

    with pytest.raises(ValueError, match="variable_impact"):
        service.formulate("expr_test")
