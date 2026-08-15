"""Bootstrap hooks for model management."""

from __future__ import annotations

import os

from PySide6.QtWidgets import QWidget

from app.container import ApplicationContext
from core.logging import get_logger
from model_management.dialogs.first_launch_dialog import FirstLaunchDialog
from model_management.offline import is_online
from model_management.service import ModelManagementService
from ui.app_window import AppWindow

logger = get_logger(__name__)


def attach_model_management(window: AppWindow, context: ApplicationContext) -> ModelManagementService:
    """Wire model download management without modifying frozen UI widgets."""
    service = ModelManagementService.create(context.model_registry, context.main_controller.app_config.paths.models_dir)
    context.model_management = service

    pipeline_vm = window._facade.pipeline_view_model  # noqa: SLF001
    sidebar = window._sidebar  # noqa: SLF001

    def sync_analyze_gate() -> None:
        if os.getenv("SENTIVIS_TEST_MODE") == "1":
            sidebar.analyze_button.setEnabled(pipeline_vm.is_analyze_enabled)
            return
        ready = service.all_mandatory_ready()
        sidebar.analyze_button.setEnabled(pipeline_vm.is_analyze_enabled and ready)

    if hasattr(pipeline_vm, "state_changed"):
        pipeline_vm.state_changed.connect(sync_analyze_gate)
    sync_analyze_gate()

    if os.getenv("SENTIVIS_TEST_MODE") == "1":
        return service

    if service.missing_mandatory() and is_online():
        dialog = FirstLaunchDialog(service, window)
        if dialog.exec() != FirstLaunchDialog.DialogCode.Accepted and not dialog.skipped:
            logger.warning("Model setup cancelled — analysis disabled until models are installed")
        service.refresh()
        sync_analyze_gate()
    elif service.missing_mandatory():
        report = service.offline_status()
        logger.warning(report.message)

    return service


def ensure_models_before_analysis(service: ModelManagementService, parent: QWidget) -> bool:
    """Block analysis when mandatory models are not ready."""
    if service.all_mandatory_ready():
        return True
    if not is_online():
        return False
    dialog = FirstLaunchDialog(service, parent)
    accepted = dialog.exec() == FirstLaunchDialog.DialogCode.Accepted
    service.refresh()
    return accepted and service.all_mandatory_ready()
