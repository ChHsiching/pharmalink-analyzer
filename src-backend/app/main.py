import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, API_PREFIX
from app.api.analysis import router as analysis_router
from app.api.datasets import router as datasets_router
from app.api.expression import router as expression_router
from app.api.health import router as health_router
from app.api.training import router as training_router
from app.services.data_loader import data_loader
from app.services.training import training_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_loader.load_preset_datasets()
    yield


app = FastAPI(title="PharmaLink Analyzer", version="0.1.0", lifespan=lifespan)

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
