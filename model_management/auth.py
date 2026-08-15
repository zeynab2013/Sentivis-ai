"""Secure Hugging Face token handling."""

from __future__ import annotations

import os

from core.config.user_config_paths import user_config_dir
from core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_FILE = "hf_token.secret"


def token_from_environment() -> str | None:
    """Read HF token from environment without logging."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and token.strip():
        return token.strip()
    return None


def token_from_secure_store() -> str | None:
    """Read persisted token from user config directory."""
    path = user_config_dir() / _TOKEN_FILE
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def resolve_hf_token() -> str | None:
    """Resolve HF token from environment first, then secure store."""
    return token_from_environment() or token_from_secure_store()


def store_hf_token(token: str) -> None:
    """Persist HF token locally with restricted permissions."""
    directory = user_config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _TOKEN_FILE
    path.write_text(token.strip(), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        logger.debug("Could not set restrictive permissions on HF token file")
    logger.info("Hugging Face token stored securely")


def clear_hf_token() -> None:
    """Remove persisted HF token."""
    path = user_config_dir() / _TOKEN_FILE
    if path.is_file():
        path.unlink(missing_ok=True)
