"""Integration test for the PyInstaller sidecar binary.

Marked as 'slow' — only runs when the binary has been built.
Run: uv run pytest tests/test_sidecar_integration.py -v -m slow
"""
import subprocess
import time

import httpx
import pytest

BINARY_PATH = "dist/pharmalink-backend"

pytestmark = pytest.mark.slow


@pytest.fixture
def backend_binary():
    """Start the PyInstaller binary as a subprocess."""
    proc = subprocess.Popen(
        [BINARY_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    max_wait = 30
    for _ in range(max_wait):
        try:
            r = httpx.get("http://localhost:8765/api/v1/health", timeout=1)
            if r.status_code == 200:
                break
        except httpx.ConnectError:
            pass
        time.sleep(1)
    else:
        proc.kill()
        pytest.fail("Backend binary did not start within 30 seconds")

    yield proc
    proc.kill()
    proc.wait()


def test_binary_health_check(backend_binary):
    """The sidecar binary responds to health check."""
    r = httpx.get("http://localhost:8765/api/v1/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["datasets_loaded"] is True
    assert data["dataset_count"] > 0


def test_binary_datasets_endpoint(backend_binary):
    """The sidecar binary serves dataset metadata."""
    r = httpx.get("http://localhost:8765/api/v1/datasets", timeout=5)
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert len(r.json()) > 0
