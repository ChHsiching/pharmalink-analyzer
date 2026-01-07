from fastapi import APIRouter, Query

from app.services import expression as _expression_module

router = APIRouter(prefix="/expressions", tags=["expressions"])


@router.post("/generate/{model_id}")
async def generate_expression(model_id: str, top_k: int = Query(10, ge=1, le=100)):
    return _expression_module.expression_service.generate(model_id, top_k)


@router.post("/simplify/{expr_id}")
async def simplify_expression(expr_id: str):
    return _expression_module.expression_service.simplify(expr_id)


@router.post("/optimize/{expr_id}")
async def optimize_expression(expr_id: str):
    return _expression_module.expression_service.optimize(expr_id)


@router.get("/tree/{expr_id}")
async def get_expression_tree(expr_id: str):
    return _expression_module.expression_service.get_tree(expr_id)


@router.get("/history/{expr_id}")
async def get_expression_history(expr_id: str):
    return _expression_module.expression_service.get_history(expr_id)


@router.post("/undo/{expr_id}")
async def undo_expression(expr_id: str, steps: int = Query(1)):
    return _expression_module.expression_service.undo(expr_id, steps)
