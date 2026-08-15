"""Hugging Face model downloader."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.logging import get_logger
from model_management.auth import resolve_hf_token
from model_management.catalog import ProductionModelSpec
from model_management.download.progress import DownloadProgress, DownloadState

logger = get_logger(__name__)


def is_hf_model_cached(repo_id: str) -> bool:
    """Check whether a Hugging Face repo is present in the local cache."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return True
    except Exception:
        return False
    return False


def download_hf_model(
    spec: ProductionModelSpec,
    *,
    on_progress: Callable[[DownloadProgress], None] | None = None,
) -> Path:
    """Download a Hugging Face model into the local HF cache."""
    repo_id = spec.hf_repo_id or spec.model_id
    token = resolve_hf_token()

    if on_progress:
        on_progress(
            DownloadProgress(
                spec.display_name,
                DownloadState.RUNNING,
                message=f"Downloading {repo_id} from Hugging Face…",
            )
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise OSError("huggingface_hub is required for model downloads") from exc

    try:
        cache_path = snapshot_download(
            repo_id=repo_id,
            token=token,
            resume_download=True,
        )
    except Exception as exc:
        message = str(exc)
        if "401" in message or "403" in message or "gated" in message.lower():
            raise PermissionError(
                "Hugging Face authentication required. Set HF_TOKEN or provide a token in the download dialog."
            ) from exc
        raise OSError(f"Hugging Face download failed for {repo_id}: {message}") from exc

    logger.info("Hugging Face model cached at %s", cache_path)
    if on_progress:
        on_progress(
            DownloadProgress(
                spec.display_name,
                DownloadState.COMPLETED,
                message="Hugging Face download complete",
            )
        )
    return Path(cache_path)
