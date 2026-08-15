"""Environment validation probes for startup."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _network_available(timeout_seconds: float = 2.0) -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=timeout_seconds):
            return True
    except OSError:
        return False


@dataclass(frozen=True)
class EnvironmentReport:
    """User-friendly environment diagnostics."""

    python_version: str
    operating_system: str
    cuda_available: bool
    gpu_name: str
    execution_device: str
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    config_files_ok: bool
    models_dir: str
    models_dir_writable: bool
    temp_dir_writable: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def probe_environment(
    *,
    project_root: Path,
    models_dir: Path,
    config_paths: tuple[Path, ...],
    min_disk_gb: float = 1.0,
    min_ram_gb: float = 4.0,
) -> EnvironmentReport:
    """Validate runtime environment before loading AI models."""
    warnings: list[str] = []
    errors: list[str] = []

    python_version = sys.version.split()[0]
    if sys.version_info < (3, 10) or sys.version_info >= (3, 11):
        errors.append(f"Python 3.10.x required (tested on 3.10.11); found {python_version}")

    operating_system = f"{platform.system()} {platform.release()} ({platform.machine()})"

    cuda_available = False
    gpu_name = "None"
    execution_device = "cpu"
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            execution_device = f"cuda:{int(torch.cuda.current_device())}"
        elif platform.system() == "Windows":
            warnings.append("CUDA GPU not detected. Analysis will use CPU fallback if enabled.")
    except ImportError:
        warnings.append("PyTorch is not installed yet. Model loading will fail until dependencies are available.")

    ram_total_gb = 0.0
    ram_available_gb = 0.0
    try:
        import psutil

        memory = psutil.virtual_memory()
        ram_total_gb = memory.total / (1024**3)
        ram_available_gb = memory.available / (1024**3)
        if ram_available_gb < min_ram_gb:
            warnings.append(
                f"Available RAM is low ({ram_available_gb:.1f} GB). "
                f"Recommend at least {min_ram_gb:.0f} GB free."
            )
    except ImportError:
        warnings.append("psutil unavailable; memory checks skipped.")

    disk_path = project_root
    disk_free_gb = shutil.disk_usage(disk_path).free / (1024**3)
    if disk_free_gb < min_disk_gb:
        warnings.append(f"Low disk space ({disk_free_gb:.1f} GB free on {disk_path}).")

    if not _network_available():
        warnings.append(
            "Network unavailable. Offline mode enabled — model downloads and remote services are disabled."
        )

    missing_configs = [str(path) for path in config_paths if not path.is_file()]
    config_files_ok = not missing_configs
    if missing_configs:
        errors.append("Missing configuration files: " + ", ".join(missing_configs))

    models_dir.mkdir(parents=True, exist_ok=True)
    models_dir_writable = os.access(models_dir, os.W_OK)
    if not models_dir_writable:
        errors.append(f"Models directory is not writable: {models_dir}")

    temp_dir_writable = True
    try:
        with tempfile.NamedTemporaryFile(dir=tempfile.gettempdir(), delete=True):
            pass
    except OSError:
        temp_dir_writable = False
        errors.append("Temporary directory is not writable.")

    return EnvironmentReport(
        python_version=python_version,
        operating_system=operating_system,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        execution_device=execution_device,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        disk_free_gb=disk_free_gb,
        config_files_ok=config_files_ok,
        models_dir=str(models_dir),
        models_dir_writable=models_dir_writable,
        temp_dir_writable=temp_dir_writable,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
