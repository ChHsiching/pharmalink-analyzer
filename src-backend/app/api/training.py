from fastapi import APIRouter, HTTPException

from app.config import CHECKPOINT_DIR
from app.models.training import (
    TrainingConfig, TrainingStatusResponse, CheckpointInfo,
)
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.data_loader import data_loader
from app.services.training import training_service

router = APIRouter(prefix="/models", tags=["models"])

_resolver = CheckpointResolver(CHECKPOINT_DIR, data_loader)


@router.post("/train")
async def start_training(config: TrainingConfig):
    try:
        task_id = await training_service.start_training(config)
        return {"task_id": task_id, "status": "started"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/train/status", response_model=TrainingStatusResponse)
async def get_training_status():
    return training_service.get_status()


@router.post("/train/stop")
async def stop_training():
    if training_service.get_status().status != "running":
        raise HTTPException(status_code=400, detail="No training in progress")
    training_service.request_stop()
    return {"status": "stopping"}


@router.get("/checkpoints", response_model=list[CheckpointInfo])
async def list_checkpoints():
    return _resolver.list_checkpoints()


@router.post("/checkpoints/load")
async def load_checkpoint(checkpoint_id: str):
    checkpoints = _resolver.list_checkpoints()
    ckpt = next((c for c in checkpoints if c.id == checkpoint_id), None)
    if ckpt is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return ckpt
