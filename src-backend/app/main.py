import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import CORS_ORIGINS, API_PREFIX
from app.api.analysis import router as analysis_router
from app.api.datasets import router as datasets_router
from app.api.evaluation import router as evaluation_router
from app.api.expression import router as expression_router
from app.api.health import router as health_router
from app.api.training import router as training_router
from app.exceptions import (
    CheckpointNotFoundError,
    DatasetNotFoundError,
    ExpressionNotFoundError,
    UndoLimitError,
)
from app.services.data_loader import data_loader
from app.services.training import training_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_loader.load_preset_datasets()
    yield


app = FastAPI(title="PharmaLink Analyzer", version="0.1.0", lifespan=lifespan)


@app.exception_handler(CheckpointNotFoundError)
async def checkpoint_not_found_handler(request: Request, exc: CheckpointNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(DatasetNotFoundError)
async def dataset_not_found_handler(request: Request, exc: DatasetNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ExpressionNotFoundError)
async def expression_not_found_handler(request: Request, exc: ExpressionNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(UndoLimitError)
async def undo_limit_handler(request: Request, exc: UndoLimitError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=API_PREFIX)
app.include_router(datasets_router, prefix=API_PREFIX)
app.include_router(training_router, prefix=API_PREFIX)
app.include_router(analysis_router, prefix=API_PREFIX)
app.include_router(evaluation_router, prefix=API_PREFIX)
app.include_router(expression_router, prefix=API_PREFIX)


@app.websocket("/ws/train")
async def training_websocket(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue()
    training_service.set_progress_queue(queue)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=60)
                await websocket.send_json(msg)
                if isinstance(msg, dict) and msg.get("status") in (
                    "completed", "stopped", "error"
                ):
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"status": "heartbeat"})
    except WebSocketDisconnect:
        pass
    finally:
        training_service.clear_progress_queue()
