import sys
from pathlib import Path
from unittest.mock import patch

import app.config as config


def test_data_dir_normal_mode():
    """In normal Python, DATA_DIR resolves to project-root/docs/csv_data."""
    assert config.DATA_DIR.name == "csv_data"
    assert "docs" in str(config.DATA_DIR)


def test_data_dir_frozen_mode():
    """In PyInstaller frozen mode, DATA_DIR uses _MEIPASS bundle path."""
    import importlib
    original_frozen = getattr(sys, "frozen", None)
    original_meipass = getattr(sys, "_MEIPASS", None)
    try:
        sys.frozen = True
        sys._MEIPASS = "/tmp/_MEIPASS_BUNDLE"
        importlib.reload(config)
        assert config.DATA_DIR == Path("/tmp/_MEIPASS_BUNDLE/csv_data")
        assert config.CHECKPOINT_DIR == Path.home() / ".pharmalink" / "checkpoints"
    finally:
        if original_frozen is None:
            del sys.frozen
        else:
            sys.frozen = original_frozen
        if original_meipass is None:
            del sys._MEIPASS
        else:
            sys._MEIPASS = original_meipass
        importlib.reload(config)
