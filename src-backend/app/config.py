import sys
from pathlib import Path

PORT = 8765
API_PREFIX = "/api/v1"
CORS_ORIGINS = ["http://localhost:5173", "tauri://localhost"]

if getattr(sys, "frozen", False):
    _BASE = Path(sys._MEIPASS)
    DATA_DIR = _BASE / "csv_data"
    CHECKPOINT_DIR = Path.home() / ".pharmalink" / "checkpoints"
else:
    DATA_DIR = Path(__file__).parent.parent.parent / "docs" / "csv_data"
    CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
