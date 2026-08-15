"""Path resolution utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_MARKERS: tuple[str, ...] = (
    "pyproject.toml",
    "config/app.default.toml",
)


def project_root() -> Path:
    """Return the Sentivis AI project root directory."""
    env_root = os.environ.get("SENTIVIS_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()

    start = Path(__file__).resolve()
    for candidate in (start, *start.parents):
        if any((candidate / marker).is_file() for marker in _PROJECT_MARKERS):
            return candidate

    return start.parents[2]


def ensure_project_root_on_path() -> Path:
    """Ensure project root is on ``sys.path`` for portable launches."""
    root = project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def runtime_directories(root: Path | None = None) -> tuple[Path, ...]:
    """Return standard runtime directories relative to project root."""
    base = root or project_root()
    return (
        base / "models",
        base / "cache",
        base / "logs",
        base / "exports",
        base / "tmp",
        base / "assets" / "user",
        base / ".sentivis",
    )


def ensure_runtime_directories(root: Path | None = None) -> tuple[Path, ...]:
    """Create runtime directories if they do not exist."""
    created: list[Path] = []
    for directory in runtime_directories(root):
        directory.mkdir(parents=True, exist_ok=True)
        created.append(directory)
    return tuple(created)


def resolve_user_path(value: str, root: Path) -> Path:
    """Resolve a configured path relative to project root when not absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def normalize_optional_path(value: object | None) -> Path | None:
    """Return None when a configured path is unset, blank, or the current directory."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {".", "./"}:
        return None
    return Path(text)


def resource_path(*parts: str) -> Path:
    """Return path to a bundled resource under assets/."""
    return project_root().joinpath("assets", *parts)


def uploads_dir(root: Path | None = None) -> Path:
    """Directory for uploaded images during a Streamlit session."""
    directory = (root or project_root()) / "tmp" / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
