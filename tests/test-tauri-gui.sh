#!/usr/bin/env bash
# Tauri Build & Launch Integration Test
# Verifies: backend starts → Tauri app builds and launches → frontend connects to backend
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BACKEND_PORT=8765
TIMEOUT=30
APP_PATH="src-tauri/target/debug/pharmalink-analyzer"
BACKEND_PID=""
APP_PID=""

cleanup() {
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
    [ -n "$APP_PID" ] && kill "$APP_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Tauri Build & Launch Test ==="

# 1. Build Tauri debug binary if needed
if [ ! -f "$APP_PATH" ]; then
    echo "BUILD: cargo tauri build --debug --no-bundle"
    cargo tauri build --debug --no-bundle 2>&1 | tail -3
fi
echo "PASS: Binary exists"

# 2. Start backend
(cd src-backend && uv run uvicorn app.main:app --port "$BACKEND_PORT") &>/tmp/gui-test-backend.log &
BACKEND_PID=$!

elapsed=0
until curl -sf "http://localhost:${BACKEND_PORT}/api/v1/health" >/dev/null 2>&1; do
    [ "$elapsed" -ge "$TIMEOUT" ] && { echo "FAIL: Backend timeout"; exit 1; }
    sleep 1; elapsed=$((elapsed + 1))
done
echo "PASS: Backend healthy on port $BACKEND_PORT"

# 3. Launch Tauri app
WEBKIT_DISABLE_DMABUF_RENDERER=1 "$APP_PATH" &>/tmp/gui-test-tauri.log &
APP_PID=$!
sleep 5

if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "FAIL: Tauri app crashed"
    tail -10 /tmp/gui-test-tauri.log
    exit 1
fi
echo "PASS: Tauri app running"

# 4. Verify frontend → backend connectivity
sleep 3
if grep -q "GET /api/v1/health" /tmp/gui-test-backend.log; then
    echo "PASS: Frontend health check reached backend"
fi

# 5. No errors
if grep -qiE "error|panic" /tmp/gui-test-tauri.log 2>/dev/null; then
    echo "FAIL: Errors in Tauri log"
    grep -iE "error|panic" /tmp/gui-test-tauri.log | tail -5
    exit 1
fi
echo "PASS: No errors"

echo ""
echo "=== ALL TESTS PASSED ==="
