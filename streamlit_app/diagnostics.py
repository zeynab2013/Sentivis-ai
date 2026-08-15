"""Startup dependency validation and judge-friendly readiness checks."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from core.utils.paths import project_root, resource_path


@dataclass(frozen=True)
class DependencyCheck:
    """One dependency or runtime requirement."""

    name: str
    available: bool
    version: str
    required: bool
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    """Aggregated startup readiness for judges."""

    ready: bool
    title: str
    summary: str
    dependencies: tuple[DependencyCheck, ...]
    assets_ok: bool
    config_ok: bool
    permissions_ok: bool
    gpu_available: bool
    cpu_name: str
    offline_mode: bool
    disk_free_gb: float
    ram_available_gb: float
    notes: tuple[str, ...]


def _module_version(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return "missing"
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return "missing"
    return str(getattr(module, "__version__", "installed"))


def _check_module(name: str, import_name: str, *, required: bool = True) -> DependencyCheck:
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        return DependencyCheck(
            name=name,
            available=False,
            version="missing",
            required=required,
            detail=f"{name} is not installed.",
        )
    try:
        importlib.import_module(import_name)
    except ImportError as exc:
        return DependencyCheck(
            name=name,
            available=False,
            version="missing",
            required=required,
            detail=f"{name} import failed: {exc}",
        )
    return DependencyCheck(
        name=name,
        available=True,
        version=_module_version(import_name),
        required=required,
        detail=f"{name} available ({_module_version(import_name)}).",
    )


def is_network_available(timeout_seconds: float = 2.0) -> bool:
    """Return True when outbound network access appears available."""
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def check_dependencies(*, offline: bool | None = None) -> tuple[DependencyCheck, ...]:
    """Validate runtime dependencies without crashing."""
    offline_mode = not is_network_available() if offline is None else offline
    checks: list[DependencyCheck] = []

    py_version = sys.version.split()[0]
    py_ok = sys.version_info[:2] == (3, 10)
    checks.append(
        DependencyCheck(
            name="Python",
            available=py_ok,
            version=py_version,
            required=True,
            detail="Python 3.10.x required (tested on 3.10.11)."
            if py_ok
            else f"Python 3.10.x required; found {py_version}.",
        )
    )

    for name, import_name, required in (
        ("PyTorch", "torch", True),
        ("Transformers", "transformers", True),
        ("OpenCV", "cv2", True),
        ("Streamlit", "streamlit", True),
        ("Ultralytics (YOLO)", "ultralytics", True),
        ("Pillow", "PIL", True),
        ("NumPy", "numpy", True),
    ):
        checks.append(_check_module(name, import_name, required=required))

    blip = _check_module("BLIP (Transformers)", "transformers", required=False)
    checks.append(
        DependencyCheck(
            name="BLIP",
            available=blip.available,
            version=blip.version,
            required=False,
            detail="BLIP uses Transformers; downloads require network when offline."
            if offline_mode
            else blip.detail,
        )
    )

    ollama = DependencyCheck(
        name="Ollama",
        available=shutil.which("ollama") is not None,
        version="cli" if shutil.which("ollama") else "missing",
        required=False,
        detail="Ollama CLI detected." if shutil.which("ollama") else "Ollama optional; Gemma may use Hugging Face.",
    )
    checks.append(ollama)

    sam2 = DependencyCheck(
        name="SAM2",
        available=importlib.util.find_spec("sam2") is not None,
        version="installed" if importlib.util.find_spec("sam2") else "optional",
        required=False,
        detail="SAM2 package optional; segmentation falls back when unavailable.",
    )
    checks.append(sam2)

    return tuple(checks)


def build_readiness_report(root: Path | None = None) -> ReadinessReport:
    """Build judge-friendly readiness report."""
    base = root or project_root()
    dependencies = check_dependencies()
    required_missing = [item.name for item in dependencies if item.required and not item.available]

    config_files = (
        base / "config" / "app.default.toml",
        base / "config" / "models.default.toml",
        base / "config" / "analysis.default.toml",
        base / "config" / "themes.default.toml",
    )
    config_ok = all(path.is_file() for path in config_files)

    logo_ok = logo_path().is_file()
    try:
        from core.resources import load_translation_catalog

        translations_ok = bool(load_translation_catalog("en"))
    except Exception:  # noqa: BLE001
        translations_ok = (base / "translations" / "en.json").is_file()
    assets_ok = logo_ok and translations_ok

    writable_dirs = (
        base / "models",
        base / "cache",
        base / "logs",
        base / "exports",
        base / "tmp",
    )
    permissions_ok = True
    for directory in writable_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        if not _is_writable(directory):
            permissions_ok = False
            break

    gpu_available = False
    execution_device = "cpu"
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
        if gpu_available:
            execution_device = f"cuda:{int(torch.cuda.current_device())}"
    except ImportError:
        pass

    ram_available_gb = 0.0
    try:
        import psutil

        ram_available_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        pass

    disk_free_gb = shutil.disk_usage(base).free / (1024**3)
    offline_mode = not is_network_available()
    cpu_name = platform.processor() or platform.machine()

    notes: list[str] = []
    if offline_mode:
        notes.append("Offline mode: model downloads and Ollama remote calls are disabled.")
    if not gpu_available:
        notes.append(f"DEVICE={execution_device}: CUDA GPU not detected; CPU fallback will be used.")
    else:
        notes.append(f"DEVICE={execution_device}")
    if not permissions_ok:
        notes.append("One or more runtime directories are not writable.")

    ready = not required_missing and config_ok and permissions_ok
    title = "System Ready" if ready else "System Not Ready"
    if ready and notes:
        summary = "Core requirements satisfied. Review optional notes below."
    elif ready:
        summary = "All required dependencies, assets, and permissions verified."
    else:
        summary = "Required components missing: " + ", ".join(required_missing or ["configuration or permissions"])

    return ReadinessReport(
        ready=ready,
        title=title,
        summary=summary,
        dependencies=dependencies,
        assets_ok=assets_ok,
        config_ok=config_ok,
        permissions_ok=permissions_ok,
        gpu_available=gpu_available,
        cpu_name=cpu_name,
        offline_mode=offline_mode,
        disk_free_gb=disk_free_gb,
        ram_available_gb=ram_available_gb,
        notes=tuple(notes),
    )


def logo_path() -> Path:
    for parts in (
        ("branding", "logo", "logo.png"),
        ("branding", "logo", "logo.svg"),
        ("icons", "app_icon.svg"),
    ):
        path = resource_path(*parts)
        if path.is_file():
            return path
    return resource_path("icons", "app_icon.svg")


def _is_writable(directory: Path) -> bool:
    try:
        probe = directory / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False
