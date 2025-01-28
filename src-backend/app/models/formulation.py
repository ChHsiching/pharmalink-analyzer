"""Pydantic models for the Optimal Formulation (最优配比) feature."""

from pydantic import BaseModel, Field


class FormulationRequest(BaseModel):
    expr_id: str
    top_k: int = Field(default=5, ge=1, le=50)
    n_samples: int = Field(default=1000, ge=100, le=10000)


class FormulationCandidate(BaseModel):
    components: dict[str, float]
    predicted_response: float
    rank: int


class FormulationResponse(BaseModel):
    expr_id: str
    candidates: list[FormulationCandidate]
    feature_names: list[str]
    attention_weights: dict[str, float]
