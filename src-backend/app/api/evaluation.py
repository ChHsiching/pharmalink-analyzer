# src-backend/app/api/evaluation.py
from fastapi import APIRouter

from app.services import evaluation as _evaluation_module

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/metrics/{model_id}")
def get_metrics(model_id: str):
    return _evaluation_module.evaluation_service.get_metrics(model_id)


@router.get("/predictions/{model_id}")
def get_predictions(model_id: str):
    return _evaluation_module.evaluation_service.get_predictions(model_id)


@router.get("/residuals/{model_id}")
def get_residuals(model_id: str):
    return _evaluation_module.evaluation_service.get_residuals(model_id)


@router.get("/loss-curve/{model_id}")
def get_loss_curve(model_id: str):
    return _evaluation_module.evaluation_service.get_loss_curve(model_id)
