# src-backend/app/api/analysis.py
from fastapi import APIRouter, Query

from app.services import analysis as _analysis_module

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/attention/{model_id}")
async def get_attention_matrix(
    model_id: str,
    top_k: int = Query(10, ge=1, le=100),
    threshold: float | None = Query(None),
):
    return _analysis_module.analysis_service.get_attention_matrix(model_id, top_k, threshold)


@router.get("/attention/{model_id}/heatmap")
async def get_heatmap_data(model_id: str):
    return _analysis_module.analysis_service.get_heatmap_data(model_id)


@router.get("/network/{model_id}")
async def get_network_graph(
    model_id: str,
    threshold: float | None = Query(None),
):
    return _analysis_module.analysis_service.get_network_graph(model_id, threshold)
