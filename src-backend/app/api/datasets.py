from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models.dataset import DatasetMeta, DatasetDetail, DatasetStatsResponse, UpdateTargetRequest
from app.services.data_loader import data_loader

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetMeta])
async def list_datasets():
    return data_loader.list_datasets()


@router.post("", response_model=DatasetMeta, status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return data_loader.add_dataset(
            file.filename, content, name=file.filename.rsplit(".", 1)[0]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")


@router.get("/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: str):
    detail = data_loader.get_dataset(dataset_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return detail


@router.get("/{dataset_id}/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(dataset_id: str):
    stats = data_loader.get_stats(dataset_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return stats


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str):
    result = data_loader.delete_dataset(dataset_id)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="Dataset not found")
    if result == "is_preset":
        raise HTTPException(status_code=403, detail="Cannot delete preset dataset")


@router.patch("/{dataset_id}/target", response_model=DatasetMeta)
async def update_target(dataset_id: str, req: UpdateTargetRequest):
    try:
        return data_loader.update_target(dataset_id, req.target_column)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg and "Column" in msg:
            raise HTTPException(status_code=400, detail=msg)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg)
        if "checkpoint" in msg:
            raise HTTPException(status_code=409, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
