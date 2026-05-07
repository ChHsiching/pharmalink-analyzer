from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_checkpoint_resolver, get_training_service
from app.models.training import (
    TrainingConfig, TrainingStatusResponse, CheckpointInfo,
)
from app.services.checkpoint_resolver import CheckpointResolver
from app.services.training import TrainingService

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/train")
async def start_training(
    config: TrainingConfig,
    training_service: TrainingService = Depends(get_training_service),
):
    try:
        task_id = await training_service.start_training(config)
        return {"task_id": task_id, "status": "started"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/train/status", response_model=TrainingStatusResponse)
async def get_training_status(
    training_service: TrainingService = Depends(get_training_service),
):
    return training_service.get_status()


@router.post("/train/stop")
async def stop_training(
    training_service: TrainingService = Depends(get_training_service),
):
    if training_service.get_status().status != "running":
        raise HTTPException(status_code=400, detail="No training in progress")
    training_service.request_stop()
    return {"status": "stopping"}


@router.get("/checkpoints", response_model=list[CheckpointInfo])
async def list_checkpoints(
    resolver: CheckpointResolver = Depends(get_checkpoint_resolver),
):
    return resolver.list_checkpoints()


@router.post("/checkpoints/load")
async def load_checkpoint(
    checkpoint_id: str,
    resolver: CheckpointResolver = Depends(get_checkpoint_resolver),
):
    checkpoints = resolver.list_checkpoints()
    ckpt = next((c for c in checkpoints if c.id == checkpoint_id), None)
    if ckpt is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return ckpt
