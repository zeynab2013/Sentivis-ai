"""Build a real desktop shell wired with deterministic stub pipeline models."""

from __future__ import annotations

from dataclasses import dataclass

from app.container import ApplicationContext
from app.startup.orchestrator import StartupOrchestrator
from release.hooks import attach_release_hooks
from tests.acceptance.support.stubs import (
    AcceptanceStubDetector,
    AcceptanceStubReasoning,
    AcceptanceStubVisionLanguage,
)
from tests.support.pipeline_harness import build_test_orchestrator
from ui.app_window import AppWindow


@dataclass
class AcceptanceApp:
    """Running desktop shell with startup context."""

    window: AppWindow
    context: ApplicationContext


def build_acceptance_app(*, stub_pipeline: bool = True) -> AcceptanceApp:
    """Start the real application stack and optionally inject stub models."""
    startup = StartupOrchestrator().run()
    if stub_pipeline:
        stub = build_test_orchestrator(
            AcceptanceStubDetector(),
            AcceptanceStubVisionLanguage(),
            AcceptanceStubReasoning(),
        )
        startup.context.main_controller.pipeline._orchestrator = stub  # noqa: SLF001
    window = AppWindow(startup.context.facade)
    from ui.i18n.translator import get_translator

    get_translator().set_language("en")
    attach_release_hooks(window, startup.context.release_info)
    return AcceptanceApp(window=window, context=startup.context)


def shutdown_acceptance_app(app: AcceptanceApp) -> None:
    """Release models and GPU resources."""
    app.context.model_manager.release_all()
    app.context.memory_manager.clear_gpu_cache()
    app.window.close()
