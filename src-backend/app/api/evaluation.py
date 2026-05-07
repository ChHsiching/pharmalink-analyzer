from fastapi import APIRouter, Depends

from app.dependencies import get_evaluation_service
from app.services.evaluation import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/metrics/{model_id}")
def get_metrics(
    model_id: str,
    service: EvaluationService = Depends(get_evaluation_service),
):
    return service.get_metrics(model_id)


@router.get("/predictions/{model_id}")
def get_predictions(
    model_id: str,
    service: EvaluationService = Depends(get_evaluation_service),
):
    return service.get_predictions(model_id)


@router.get("/residuals/{model_id}")
def get_residuals(
    model_id: str,
    service: EvaluationService = Depends(get_evaluation_service),
):
    return service.get_residuals(model_id)


@router.get("/loss-curve/{model_id}")
def get_loss_curve(
    model_id: str,
    service: EvaluationService = Depends(get_evaluation_service),
):
    return service.get_loss_curve(model_id)
