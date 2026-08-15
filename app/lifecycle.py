"""Application lifecycle management."""

from PySide6.QtWidgets import QApplication

from app.container import ApplicationContext
from core.logging import get_logger
from ui.app_window import AppWindow

logger = get_logger(__name__)


class ApplicationLifecycle:
    """Manages Qt application startup and shutdown."""

    def __init__(self, context: ApplicationContext, window: AppWindow, qt_app: QApplication) -> None:
        self._context = context
        self._window = window
        self._qt_app = qt_app

    def run(self) -> None:
        """Start the event loop."""
        self._window.show()
        logger.info("Application window displayed")
        exit_code = self._qt_app.exec()
        self.shutdown()
        raise SystemExit(exit_code)

    def shutdown(self) -> None:
        """Release resources on exit."""
        logger.info("Shutting down Sentivis AI")
        self._context.model_manager.release_all()
        self._context.memory_manager.clear_gpu_cache()
        logger.info("Shutdown complete")
