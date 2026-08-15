"""Application bootstrap and startup."""

from PySide6.QtWidgets import QApplication

from app.lifecycle import ApplicationLifecycle
from app.startup.orchestrator import StartupOrchestrator
from app.startup.recovery import recovery_message
from core.logging import get_logger
from model_management.hooks import attach_model_management
from release.hooks import attach_release_hooks
from ui.app_window import AppWindow
from ui.branding.logo_provider import load_window_icon
from ui.widgets.splash_screen import SplashScreenWidget

logger = get_logger(__name__)


def _ensure_qapplication() -> QApplication:
    """Create the Qt application instance before any widgets are constructed."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    assert isinstance(app, QApplication)
    return app


def bootstrap() -> ApplicationLifecycle:
    """Load configuration, validate environment, wire dependencies, and prepare lifecycle."""
    qt_app = _ensure_qapplication()
    icon = load_window_icon()
    if not icon.isNull():
        qt_app.setWindowIcon(icon)
    SplashScreenWidget.show_brief(qt_app)
    result = StartupOrchestrator().run()
    if not result.report.ready:
        for error in result.report.errors:
            logger.error("Startup issue: %s | Recovery: %s", error, recovery_message(error))

    window = AppWindow(result.context.facade)
    attach_release_hooks(window, result.context.release_info)
    attach_model_management(window, result.context)
    logger.info(
        "Starting %s — %s",
        result.settings.app_config.app_name,
        result.context.release_info.full_version_line,
    )
    return ApplicationLifecycle(result.context, window, qt_app)


def main() -> None:
    """Application entry point for console script."""
    bootstrap().run()
