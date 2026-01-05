# src-backend/app/models/analysis.py
from pydantic import BaseModel


class AttentionPair(BaseModel):
    source: str
    target: str
    weight: float
    classification: str


class AttentionMatrixResponse(BaseModel):
    model_id: str
    feature_names: list[str]
    matrix: list[list[float]]
    pairs: list[AttentionPair]
    threshold: float


class HeatmapDataResponse(BaseModel):
    model_id: str
    feature_names: list[str]
    values: list[list[float]]
    min_value: float
    max_value: float


class NetworkNode(BaseModel):
    id: str
    name: str


class NetworkEdge(BaseModel):
    source: str
    target: str
    weight: float
    classification: str


class NetworkGraphResponse(BaseModel):
    model_id: str
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]
    threshold: float
