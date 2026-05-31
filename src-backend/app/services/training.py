"""TrainingService — facade orchestrating pipeline, checkpoint manager, async tasks, and WebSocket."""

import asyncio
from datetime import datetime
from pathlib import Path

import numpy as np

from app.config import CHECKPOINT_DIR
from app.models.training import (
    TrainingConfig, TrainingProgress, TrainingStatusResponse,
)
from app.services.checkpoint_manager import CheckpointManager
from app.services.data_loader import data_loader as default_data_loader
from app.services.data_loader import DataLoader
from app.services.training_pipeline import TrainingPipeline


class TrainingService:
    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        data_loader: DataLoader | None = None,
        pipeline: TrainingPipeline | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ):
        self._checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self._data_loader = data_loader or default_data_loader
        self._pipeline = pipeline or TrainingPipeline()
        self._checkpoint_manager = checkpoint_manager or CheckpointManager(self._checkpoint_dir)
        self._status = "idle"
        self._config: TrainingConfig | None = None
        self._task_id: str | None = None
        self._progress: list[TrainingProgress] = []
        self._stop_requested = False
        self._progress_queues: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def get_status(self) -> TrainingStatusResponse:
        fold, epoch = 0, 0
        if self._progress:
            last = self._progress[-1]
            fold, epoch = last.fold, last.epoch
        return TrainingStatusResponse(
            status=self._status,
            task_id=self._task_id,
            config=self._config,
            current_fold=fold,
            current_epoch=epoch,
            progress=self._progress,
        )

    def add_progress_queue(self, queue: asyncio.Queue):
        self._progress_queues.add(queue)

    def remove_progress_queue(self, queue: asyncio.Queue):
        self._progress_queues.discard(queue)

    async def start_training(self, config: TrainingConfig) -> str:
        if self._status == "running":
            raise ValueError("Training already in progress")

        df = self._data_loader.get_dataframe(config.dataset_id)
        if df is None:
            raise ValueError(f"Dataset '{config.dataset_id}' not found")

        meta = self._data_loader.get_dataset(config.dataset_id)
        if meta is None:
            raise ValueError(f"Dataset '{config.dataset_id}' not found")
        if not meta.target:
            raise ValueError(
                f"Dataset '{config.dataset_id}' has no target variable selected. "
                "Select a target variable before training."
            )
        self._task_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self._config = config
        self._status = "running"
        self._stop_requested = False
        self._progress = []
        self._loop = asyncio.get_event_loop()

        X = df[meta.feature_names].values.astype(np.float32)
        y = df[meta.target].values.astype(np.float32)
        n_features = len(meta.feature_names)

        self._loop.run_in_executor(
            None, self._train_sync, config, X, y, n_features
        )
        return self._task_id

    def request_stop(self):
        self._stop_requested = True

    def _push_progress(self, progress: TrainingProgress):
        self._progress.append(progress)
        for queue in list(self._progress_queues):
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    queue.put(progress.model_dump()),
                    self._loop,
                )

    def _train_sync(
        self,
        config: TrainingConfig,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int,
    ):
        try:
            result = self._pipeline.train(
                X, y, n_features, config,
                on_progress=self._push_progress,
                is_stopped=lambda: self._stop_requested,
            )
            if not result.stopped and result.model_state is not None:
                self._checkpoint_manager.save(
                    self._task_id, config, result.model_state,
                    result.best_loss, result.scaler, self._progress,
                    fold_states=result.fold_states,
                    fold_scalers=result.fold_scalers,
                )
                self._finish("completed")
            else:
                self._finish("stopped")
        except Exception as e:
            self._finish("error", str(e))

    def _finish(self, status: str, error: str | None = None):
        self._status = status
        msg: dict = {"status": status}
        if error:
            msg["error"] = error
        for queue in list(self._progress_queues):
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    queue.put(msg), self._loop,
                )
