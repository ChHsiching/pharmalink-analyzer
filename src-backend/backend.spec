# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

block_cipher = None

csv_data = str(Path("docs/csv_data").resolve())
if not Path(csv_data).exists():
    print(f"WARNING: csv_data not found at {csv_data}")

a = Analysis(
    ["pyinstaller_entry.py"],
    pathex=[],
    binaries=[],
    datas=[
        (csv_data, "csv_data"),
    ],
    hiddenimports=[
        "app",
        "app.main",
        "app.config",
        "app.api.health",
        "app.api.datasets",
        "app.api.training",
        "app.api.analysis",
        "app.api.evaluation",
        "app.api.expression",
        "app.api.formulation",
        "app.services.data_loader",
        "app.services.training_service",
        "app.services.analysis_service",
        "app.services.evaluation_service",
        "app.services.expression_service",
        "app.services.formulation_service",
        "app.dependencies",
        "app.exceptions",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._partition_nodes",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="pharmalink-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
