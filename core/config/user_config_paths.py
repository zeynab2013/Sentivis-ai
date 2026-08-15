"""User configuration directory resolution."""

import os
import sys
from pathlib import Path

from core.utils.paths import project_root


def user_config_dir() -> Path:
    """Return the per-user configuration directory."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "SentivisAI" / "config"
    return project_root() / "config" / "user"


def user_override_path(filename: str) -> Path:
    """Return path to a user override file, creating parent dirs if needed."""
    return user_config_dir() / filename
