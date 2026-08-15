"""Startup recovery guidance."""

from __future__ import annotations


def recovery_message(issue: str) -> str:
    """Return user-friendly corrective action for a startup issue."""
    lower = issue.lower()
    if "python 3.10" in lower or "python 3.11" in lower:
        return "Install Python 3.10.11 and recreate the virtual environment before launching Sentivis AI."
    if "configuration" in lower and "missing" in lower:
        return "Reinstall the application or restore the config/ directory from the release package."
    if "writable" in lower and "models" in lower:
        return "Choose a writable models folder in settings or run the application with sufficient permissions."
    if "temporary directory" in lower:
        return "Verify that the system TEMP folder exists and is writable for the current user."
    if "pytorch" in lower:
        return "Install project dependencies with: pip install -e ."
    if "cuda" in lower or "gpu" in lower:
        return "Install a CUDA-capable GPU driver or enable CPU fallback in configuration."
    if "weights not found" in lower or "missing" in lower:
        return "Place model weights in the models directory or allow automatic download on first analysis."
    if "disk space" in lower or "ram" in lower:
        return "Free disk space or memory, then restart the application."
    return "Review the diagnostics report in the logs folder and correct the reported issue."
