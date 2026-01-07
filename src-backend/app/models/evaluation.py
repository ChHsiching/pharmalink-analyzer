from pydantic import BaseModel


class FoldMetrics(BaseModel):
    fold: int
    r2: float
    mse: float
    mae: float


class AggregateMetrics(BaseModel):
    r2_mean: float
    r2_std: float
    mse_mean: float
    mse_std: float
    mae_mean: float
    mae_std: float


class EvaluationMetricsResponse(BaseModel):
    model_id: str
    folds: list[FoldMetrics]
    aggregate: AggregateMetrics


class PredictionPoint(BaseModel):
    actual: float
    predicted: float


class PredictionsResponse(BaseModel):
    model_id: str
    predictions: list[PredictionPoint]


class ResidualBin(BaseModel):
    range_start: float
    range_end: float
    count: int


class ResidualsResponse(BaseModel):
    model_id: str
    residuals: list[float]
    bins: list[ResidualBin]
    mean: float
    std: float


class LossCurvePoint(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float


class FoldLossCurve(BaseModel):
    fold: int
    points: list[LossCurvePoint]


class LossCurveResponse(BaseModel):
    model_id: str
    folds: list[FoldLossCurve]
