"""Tests for formulation Pydantic models."""

from pydantic import ValidationError
import pytest

from app.models.formulation import FormulationCandidate, FormulationRequest, FormulationResponse


def test_request_defaults():
    req = FormulationRequest(expr_id="expr_abc")
    assert req.top_k == 5
    assert req.n_samples == 1000


def test_request_validation():
    with pytest.raises(ValidationError):
        FormulationRequest(expr_id="expr_abc", top_k=0)
    with pytest.raises(ValidationError):
        FormulationRequest(expr_id="expr_abc", n_samples=0)


def test_candidate():
    c = FormulationCandidate(components={"QA": 3.5}, predicted_response=0.95, rank=1)
    assert c.components["QA"] == 3.5


def test_response():
    r = FormulationResponse(
        expr_id="expr_abc", candidates=[
            FormulationCandidate(components={"QA": 3.5}, predicted_response=0.95, rank=1),
        ], feature_names=["QA"], attention_weights={"QA": 0.8},
    )
    assert len(r.candidates) == 1
