"""Ollama integration for optional Gemma provisioning."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from core.logging import get_logger
from model_management.catalog import ProductionModelSpec
from model_management.download.progress import DownloadProgress, DownloadState

logger = get_logger(__name__)


@dataclass(frozen=True)
class OllamaStatus:
    """Detected Ollama runtime state."""

    installed: bool
    running: bool
    detail: str


def detect_ollama() -> OllamaStatus:
    """Detect whether Ollama CLI is installed and reachable."""
    if shutil.which("ollama") is None:
        return OllamaStatus(False, False, "Ollama is not installed. Download from https://ollama.com/download")
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return OllamaStatus(True, True, "Ollama is installed and responding.")
        return OllamaStatus(True, False, result.stderr.strip() or "Ollama installed but not responding.")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return OllamaStatus(True, False, f"Ollama check failed: {exc}")


def pull_ollama_model(
    spec: ProductionModelSpec,
    *,
    on_progress: Callable[[DownloadProgress], None] | None = None,
) -> None:
    """Pull configured Ollama model tag."""
    tag = spec.ollama_tag
    if not tag:
        raise ValueError(f"No Ollama tag configured for {spec.display_name}")

    status = detect_ollama()
    if not status.installed:
        raise OSError(status.detail)
    if not status.running:
        raise OSError(status.detail)

    if on_progress:
        on_progress(
            DownloadProgress(
                spec.display_name,
                DownloadState.RUNNING,
                message=f"Pulling Ollama model {tag}…",
            )
        )

    result = subprocess.run(
        ["ollama", "pull", tag],
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or f"ollama pull {tag} failed")

    logger.info("Ollama model %s installed", tag)
    if on_progress:
        on_progress(
            DownloadProgress(
                spec.display_name,
                DownloadState.COMPLETED,
                message=f"Ollama model {tag} ready",
            )
        )
