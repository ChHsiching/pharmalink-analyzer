from fastapi import APIRouter

from app.services.data_loader import data_loader

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    datasets = data_loader.list_datasets()
    return {
        "status": "ok",
        "datasets_loaded": len(datasets) > 0,
        "dataset_count": len(datasets),
    }
