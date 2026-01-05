from app.config import PORT, API_PREFIX, CORS_ORIGINS, DATA_DIR, CHECKPOINT_DIR


def test_config_has_defaults():
    assert PORT == 8765
    assert API_PREFIX == "/api/v1"
    assert len(CORS_ORIGINS) >= 2


def test_data_dir_is_path():
    from pathlib import Path
    assert isinstance(DATA_DIR, Path)
    assert isinstance(CHECKPOINT_DIR, Path)


def test_cors_contains_dev_origins():
    assert "http://localhost:5173" in CORS_ORIGINS
    assert "tauri://localhost" in CORS_ORIGINS
