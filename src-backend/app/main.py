from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, API_PREFIX
from app.api.datasets import router as datasets_router
from app.api.health import router as health_router
from app.services.data_loader import data_loader


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
