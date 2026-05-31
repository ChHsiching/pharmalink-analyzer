from fastapi import APIRouter, Depends

from app.dependencies import get_formulation_service
from app.models.formulation import FormulationRequest, FormulationResponse
from app.services.formulation import FormulationService

router = APIRouter(prefix="/formulation", tags=["formulation"])


@router.post("")
async def formulate(
    request: FormulationRequest,
    service: FormulationService = Depends(get_formulation_service),
) -> FormulationResponse:
    return service.formulate(request.expr_id, top_k=request.top_k, n_samples=request.n_samples)
