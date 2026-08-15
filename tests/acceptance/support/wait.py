"""Qt event-loop helpers for acceptance tests."""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtWidgets import QApplication

from ui.models.pipeline_ui_state import PipelineUiState


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_ms: int = 30_000,
    interval_ms: int = 50,
) -> bool:
    """Poll until predicate is true or timeout expires."""
    app = QApplication.instance()
    deadline = time.perf_counter() + (timeout_ms / 1000.0)
    while time.perf_counter() < deadline:
        if predicate():
            return True
        if app is not None:
            app.processEvents()
        time.sleep(interval_ms / 1000.0)
    return predicate()


def wait_for_analysis(pipeline_vm: object, *, timeout_ms: int = 60_000) -> bool:
    """Wait for pipeline analysis to complete on the UI thread."""
    ui_state = getattr(pipeline_vm, "ui_state", None)
    return wait_until(
        lambda: getattr(pipeline_vm, "has_result", False)
        or ui_state in {PipelineUiState.COMPLETED, PipelineUiState.FAILED, PipelineUiState.CANCELLED},
        timeout_ms=timeout_ms,
    ) and getattr(pipeline_vm, "has_result", False)


def wait_for_signal(emitter: object, signal_name: str, *, timeout_ms: int = 30_000) -> bool:
    """Block until a Qt signal fires, processing pending events."""
    app = QApplication.instance()
    received = False

    def on_signal(*_args: object) -> None:
        nonlocal received
        received = True

    signal = getattr(emitter, signal_name)
    signal.connect(on_signal)
    deadline = time.perf_counter() + (timeout_ms / 1000.0)
    while time.perf_counter() < deadline:
        if received:
            signal.disconnect(on_signal)
            return True
        if app is not None:
            app.processEvents()
        time.sleep(0.05)
    signal.disconnect(on_signal)
    return received
