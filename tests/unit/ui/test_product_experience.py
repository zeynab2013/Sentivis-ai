"""Unit tests for product experience UI models."""

from ui.models.app_state import ApplicationState, resolve_application_state
from ui.models.operation_status import OperationStatus, operation_status_for_app_state
from ui.models.pipeline_ui_state import PipelineUiState


def test_analyzing_state_maps_to_running_status() -> None:
    status = operation_status_for_app_state(ApplicationState.ANALYZING.value)
    assert status == OperationStatus.RUNNING


def test_image_loaded_state_is_ready() -> None:
    state = resolve_application_state(
        PipelineUiState.IDLE,
        has_image=True,
        exporting=False,
    )
    assert state == ApplicationState.IMAGE_LOADED
