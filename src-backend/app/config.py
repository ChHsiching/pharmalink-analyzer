from pathlib import Path

PORT = 8765
API_PREFIX = "/api/v1"
CORS_ORIGINS = ["http://localhost:5173", "tauri://localhost"]
DATA_DIR = Path(__file__).parent.parent.parent.parent / "docs" / "csv_data"
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
