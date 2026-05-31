#!/usr/bin/env bash
set -euo pipefail

BACKEND_PORT=8765
MAX_WAIT=30

echo "Starting Python backend..."
(cd src-backend && uv run uvicorn app.main:app --port "$BACKEND_PORT" --reload) &
BACKEND_PID=$!

echo "Waiting for backend (max ${MAX_WAIT}s)..."
elapsed=0
until curl -sf "http://localhost:${BACKEND_PORT}/docs" > /dev/null 2>&1; do
    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Backend not ready within ${MAX_WAIT}s" >&2
        kill "$BACKEND_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done
echo "Backend ready on port ${BACKEND_PORT}"

cleanup() {
    echo "Stopping backend (PID ${BACKEND_PID})..."
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting Tauri desktop..."
npm run tauri:dev
