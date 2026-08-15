"""Windows-friendly console bootstrap for CMD launches."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def prepare_windows_console() -> None:
    """Reduce silent buffering / encoding issues on Windows CMD."""
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Streamlit inherits these for child process output.
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def _emit(message: str) -> None:
    print(message, flush=True)


def _device_label() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"CUDA ({name})"
    except Exception:  # noqa: BLE001
        pass
    return "CPU"


def _ollama_has_model(model_id: str) -> bool:
    if shutil.which("ollama") is None:
        return False
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return False
    haystack = (completed.stdout or "").lower()
    return model_id.lower() in haystack


def _easyocr_available() -> bool:
    return importlib.util.find_spec("easyocr") is not None


def print_startup_banner(*, root: Path, app_path: Path, port: int) -> None:
    _emit("==================================================")
    _emit("Sentivis AI starting...")
    _emit("==================================================")
    _emit(f"Python: {sys.version.split()[0]} ({sys.executable})")
    _emit(f"Project: {root}")
    _emit(f"Entry:   {app_path}")
    _emit(f"Device:  {_device_label()}")
    _emit(f"Logs:    {root / 'logs'}")
    _emit(f"Local URL (default): http://localhost:{port}")
    _emit("Console logging enabled for this interactive session.")
    _emit("")


def print_runtime_diagnostics(*, root: Path) -> None:
    """Compact model/runtime summary without loading heavy weights."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    yolo = "not checked"
    florence = "not checked"
    gemma = "not checked"
    sam2 = "not checked"
    try:
        from app.settings_loader import load_application_settings
        from app.startup.model_discovery import discover_models

        settings = load_application_settings()
        report = discover_models(settings.model_config, settings.app_config.paths.models_dir)
        by_kind = {entry.kind: entry for entry in report.entries}
        yolo_entry = by_kind.get("yolo")
        florence_entry = by_kind.get("florence2")
        sam2_entry = by_kind.get("sam2")
        yolo = (
            "installed/loadable" if yolo_entry and yolo_entry.available else "missing / may download"
        )
        florence = (
            "configured"
            if florence_entry and florence_entry.available
            else "not configured"
        )
        if sam2_entry and sam2_entry.available:
            sam2 = "package/weights present"
        else:
            sam2 = "missing weights / disabled"
        gemma_id = settings.model_config.vlm.model_ids.gemma_vision or "gemma3:4b"
        if _ollama_has_model(gemma_id):
            gemma = f"Ollama model available ({gemma_id})"
        elif shutil.which("ollama"):
            gemma = f"Ollama CLI present; model '{gemma_id}' not listed"
        else:
            gemma = "Ollama not installed; Gemma runtime unavailable via Ollama"
    except Exception as exc:  # noqa: BLE001
        _emit(f"WARNING: startup diagnostics limited: {exc}")

    ocr = "EasyOCR available" if _easyocr_available() else "EasyOCR not importable"

    _emit("Runtime diagnostics")
    _emit("-------------------")
    _emit(f"YOLO:       {yolo}")
    _emit(f"Florence:   {florence}")
    _emit(f"Gemma 3 4B: {gemma}")
    _emit(f"OCR:        {ocr}")
    _emit(f"SAM2:       {sam2}")
    _emit(f"Device:     {_device_label()}")
    _emit("")
    _emit("Launching Streamlit (child process inherits this console)...")
    _emit("Watch for Streamlit 'Local URL' below. Application logs also write to logs/application.log")
    _emit("")
