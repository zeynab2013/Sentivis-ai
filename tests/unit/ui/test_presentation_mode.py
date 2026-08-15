"""Unit tests for presentation mode."""

from ui.models.presentation_mode import PresentationModeController


def test_presentation_mode_toggle() -> None:
    controller = PresentationModeController()
    assert not controller.enabled
    controller.toggle()
    assert controller.enabled
    controller.toggle()
    assert not controller.enabled


def test_presentation_mode_set_enabled_emits_once() -> None:
    controller = PresentationModeController()
    changes: list[bool] = []
    controller.mode_changed.connect(changes.append)
    controller.set_enabled(True)
    controller.set_enabled(True)
    assert changes == [True]
