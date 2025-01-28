#!/usr/bin/env bash
set -euo pipefail

echo "=== Building PharmaLink Backend with PyInstaller ==="

TARGET_TRIPLE=$(rustc --print host-tuple)
if [ -z "$TARGET_TRIPLE" ]; then
    echo "ERROR: Failed to determine Rust target triple" >&2
    exit 1
fi
echo "Target triple: $TARGET_TRIPLE"

EXT=""
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    EXT=".exe"
fi

cd src-backend
uv run pyinstaller backend.spec \
    --clean \
    --noconfirm \
    --distpath dist \
    --workpath build
cd ..

BINARY_NAME="pharmalink-backend${EXT}"
BINARY_PATH="src-backend/dist/${BINARY_NAME}"
if [ ! -f "$BINARY_PATH" ]; then
    echo "ERROR: Binary not found at $BINARY_PATH" >&2
    exit 1
fi

mkdir -p src-tauri/binaries
SIDECAR_NAME="binaries/pharmalink-backend-${TARGET_TRIPLE}${EXT}"
cp "$BINARY_PATH" "src-tauri/${SIDECAR_NAME}"
chmod +x "src-tauri/${SIDECAR_NAME}"

echo "=== Sidecar placed at src-tauri/${SIDECAR_NAME} ==="
ls -lh "src-tauri/${SIDECAR_NAME}"
