"""Streamlit backend initialization using the production startup orchestrator."""

from __future__ import annotations

from app.startup.orchestrator import StartupOrchestrator, StartupResult
from core.logging import get_logger
from services.export.export_manager import ExportManager
from streamlit_app.backend import StreamlitBackend

logger = get_logger(__name__)


def initialize_backend() -> StreamlitBackend:
    """Run production startup and return backend adapter."""
    result: StartupResult = StartupOrchestrator().run()
    if not result.report.ready:
        for error in result.report.errors:
            logger.warning("Startup issue: %s", error)
    context = result.context
    orchestrator = context.main_controller.pipeline._orchestrator  # noqa: SLF001
    progress = context.main_controller.pipeline._progress  # noqa: SLF001
    return StreamlitBackend(
        context=context,
        orchestrator=orchestrator,
        progress=progress,
        export_manager=ExportManager(),
    )
