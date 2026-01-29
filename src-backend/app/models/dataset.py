from pydantic import BaseModel


class ColumnStats(BaseModel):
    column: str
    mean: float
    std: float
    min: float
    max: float


class DatasetMeta(BaseModel):
    id: str
    name: str
    filename: str
    plant_part: str
    target: str
    n_samples: int
    n_features: int
    feature_names: list[str]
    columns: list[str]
    is_preset: bool


class DatasetDetail(DatasetMeta):
    preview: list[dict[str, float]]


class DatasetStatsResponse(BaseModel):
    dataset_id: str
    stats: list[ColumnStats]


class UpdateTargetRequest(BaseModel):
    target_column: str
