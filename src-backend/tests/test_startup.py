import logging
from pathlib import Path

import pytest
from fastapi import FastAPI

from app.main import lifespan


@pytest.mark.asyncio
async def test_lifespan_logs_error_when_data_dir_missing(caplog, monkeypatch):
    """AC1: DATA_DIR not found → ERROR log."""
    monkeypatch.setattr("app.main.DATA_DIR", Path("/nonexistent/pharmalink"))

    test_app = FastAPI(lifespan=lifespan)
    with caplog.at_level(logging.ERROR, logger="app.main"):
        async with lifespan(test_app):
            pass

    assert any(
        "DATA_DIR" in r.message and r.levelno == logging.ERROR
        for r in caplog.records
    ), f"Expected ERROR log about DATA_DIR, got: {[r.message for r in caplog.records]}"


@pytest.mark.asyncio
async def test_lifespan_logs_warning_when_no_datasets(caplog, monkeypatch):
    """AC2: load_preset_datasets returns empty → WARNING log."""
    from app.services.data_loader import DataLoader

    empty_loader = DataLoader(data_dir=Path("/tmp"))
    monkeypatch.setattr("app.main.data_loader", empty_loader)

    test_app = FastAPI(lifespan=lifespan)
    with caplog.at_level(logging.WARNING, logger="app.main"):
        async with lifespan(test_app):
            pass

    assert any(
        "No preset datasets" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    ), f"Expected WARNING about no datasets, got: {[r.message for r in caplog.records]}"
