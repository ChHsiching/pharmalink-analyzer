from fastapi import APIRouter, Depends, Query

from app.dependencies import get_analysis_service
from app.services.analysis import AnalysisService

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/attention/{model_id}")
async def get_attention_matrix(
    model_id: str,
    top_k: int = Query(10, ge=1, le=100),
    threshold: int | None = Query(None, ge=0, le=100),
    service: AnalysisService = Depends(get_analysis_service),
):
    return service.get_attention_matrix(model_id, top_k, threshold)


@router.get("/attention/{model_id}/heatmap")
async def get_heatmap_data(
    model_id: str,
    service: AnalysisService = Depends(get_analysis_service),
):
    return service.get_heatmap_data(model_id)


@router.get("/network/{model_id}")
async def get_network_graph(
    model_id: str,
    threshold: int | None = Query(None, ge=0, le=100),
    service: AnalysisService = Depends(get_analysis_service),
):
    return service.get_network_graph(model_id, threshold)
