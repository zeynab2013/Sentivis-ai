"""Unit tests for the design system and UI state models."""

from dataclasses import replace

from ui.design import DARK_TOKENS, LIGHT_TOKENS
from ui.models.app_state import ApplicationState, application_state_message, resolve_application_state
from ui.models.operation_status import OperationStatus, operation_status_for_app_state
from ui.models.pipeline_ui_state import PipelineUiState
from ui.themes.theme_engine import render_stylesheet


def test_design_tokens_expose_spacing_and_typography() -> None:
    tokens = DARK_TOKENS
    assert tokens.spacing("md") == 14
    assert tokens.radius("lg") == 16
    assert tokens.font_size("xl") == 20
    assert tokens.icon_size("md") == 18


def test_dark_and_light_tokens_are_distinct() -> None:
    assert DARK_TOKENS.background != LIGHT_TOKENS.background
    assert DARK_TOKENS.text_primary != LIGHT_TOKENS.text_primary


def test_render_stylesheet_uses_token_values() -> None:
    sheet = render_stylesheet(DARK_TOKENS)
    assert DARK_TOKENS.background in sheet
    assert DARK_TOKENS.primary in sheet
    assert "SentivisButton" in sheet
    assert "StatusBadge" in sheet


def test_render_stylesheet_respects_animation_duration() -> None:
    custom = replace(DARK_TOKENS, animation_ms=250)
    sheet = render_stylesheet(custom)
    assert "250ms" in sheet


def test_resolve_application_state_prioritizes_exporting() -> None:
    state = resolve_application_state(
        PipelineUiState.COMPLETED,
        has_image=True,
        exporting=True,
    )
    assert state == ApplicationState.EXPORTING


def test_resolve_application_state_image_loaded() -> None:
    state = resolve_application_state(
        PipelineUiState.IDLE,
        has_image=True,
        exporting=False,
    )
    assert state == ApplicationState.IMAGE_LOADED


def test_application_state_messages_are_user_facing() -> None:
    assert "Ready" in application_state_message(ApplicationState.IDLE)
    assert "complete" in application_state_message(ApplicationState.COMPLETED).lower()


def test_operation_status_mapping_for_analyzing() -> None:
    status = operation_status_for_app_state(ApplicationState.ANALYZING.value)
    assert status == OperationStatus.RUNNING
