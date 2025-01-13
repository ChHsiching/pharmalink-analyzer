from typing import Any

from pydantic import BaseModel


class ExpressionNode(BaseModel):
    type: str
    value: str
    children: list["ExpressionNode"] = []


class ExpressionResponse(BaseModel):
    expr_id: str
    model_id: str
    latex: str
    complexity: int
    r2_score: float
    tree: ExpressionNode


class ExpressionHistoryEntry(BaseModel):
    operation: str
    latex: str
    complexity: int
    r2_score: float


class ExpressionHistoryResponse(BaseModel):
    expr_id: str
    history: list[ExpressionHistoryEntry]
    current_index: int


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
