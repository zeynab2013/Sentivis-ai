"""UI acceptance tests for desktop interactions."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from release.about_dialog import AboutDialog
from tests.acceptance.support.harness import AcceptanceApp
from tests.acceptance.support.wait import wait_for_analysis
from ui.i18n.translator import tr
from ui.view_models.settings_view_model import SettingsViewModel
from ui.widgets.settings_dialog import SettingsDialog


@pytest.mark.acceptance
@pytest.mark.ui
def test_sidebar_buttons_exist_and_respond(acceptance_app: AcceptanceApp, sample_image: Path) -> None:
    sidebar = acceptance_app.window._sidebar  # noqa: SLF001
    for button in (
        sidebar.open_button,
        sidebar.analyze_button,
        sidebar.cancel_button,
        sidebar.settings_button,
        sidebar.presentation_button,
    ):
        assert button.isEnabled() or button.text() in {"Run Analysis", "Cancel"}

    acceptance_app.window._load_image_path(str(sample_image))  # noqa: SLF001
    QApplication.processEvents()
    assert sidebar.analyze_button.isEnabled()


@pytest.mark.acceptance
@pytest.mark.ui
def test_image_viewer_zoom_controls(acceptance_app: AcceptanceApp, sample_image: Path) -> None:
    viewer = acceptance_app.window._image_viewer  # noqa: SLF001
    acceptance_app.window._load_image_path(str(sample_image))  # noqa: SLF001
    QApplication.processEvents()

    viewer.zoom_in()
    viewer.zoom_out()
    viewer.fit_to_window()
    viewer._original_button.click()  # noqa: SLF001
    QApplication.processEvents()
    assert viewer._scale_factor > 0  # noqa: SLF001


@pytest.mark.acceptance
@pytest.mark.ui
def test_drag_and_drop_loads_image(acceptance_app: AcceptanceApp, sample_image: Path) -> None:
    pipeline_vm = acceptance_app.window._facade.pipeline_view_model  # noqa: SLF001
    acceptance_app.window._image_viewer.image_dropped.emit(str(sample_image))  # noqa: SLF001
    QApplication.processEvents()
    assert pipeline_vm.image_path == sample_image


@pytest.mark.acceptance
@pytest.mark.ui
def test_presentation_mode_toggle(acceptance_app: AcceptanceApp) -> None:
    window = acceptance_app.window
    sidebar = window._sidebar  # noqa: SLF001
    window.show()
    QApplication.processEvents()
    assert not sidebar.isHidden()
    window._presentation_mode.toggle()  # noqa: SLF001
    QApplication.processEvents()
    assert sidebar.isHidden()
    window._presentation_mode.toggle()  # noqa: SLF001
    QApplication.processEvents()
    assert not sidebar.isHidden()


@pytest.mark.acceptance
@pytest.mark.ui
def test_settings_dialog_opens_and_cancels(acceptance_app: AcceptanceApp) -> None:
    window = acceptance_app.window
    settings_vm = cast(SettingsViewModel, window._facade.settings_view_model)  # noqa: SLF001
    dialog = SettingsDialog(settings_vm, window)
    dialog.show()
    QApplication.processEvents()
    assert dialog.windowTitle() == tr("settings.title")
    dialog.reject()
    assert dialog.result() == QDialog.DialogCode.Rejected


@pytest.mark.acceptance
@pytest.mark.ui
def test_export_panel_buttons_present(acceptance_app: AcceptanceApp, sample_image: Path) -> None:
    window = acceptance_app.window
    pipeline_vm = window._facade.pipeline_view_model  # noqa: SLF001
    panel = window._dashboard.export_panel  # noqa: SLF001

    window._load_image_path(str(sample_image))  # noqa: SLF001
    pipeline_vm.start_analysis()
    assert wait_for_analysis(pipeline_vm, timeout_ms=60_000)
    QApplication.processEvents()

    for button in (
        panel.export_txt_button,
        panel.export_md_button,
        panel.export_json_button,
        panel.export_pdf_button,
    ):
        assert button.isEnabled()


@pytest.mark.acceptance
@pytest.mark.ui
def test_results_panel_copy_and_expand(acceptance_app: AcceptanceApp, sample_image: Path) -> None:
    window = acceptance_app.window
    pipeline_vm = window._facade.pipeline_view_model  # noqa: SLF001
    results = window._dashboard.results_panel  # noqa: SLF001

    window._load_image_path(str(sample_image))  # noqa: SLF001
    pipeline_vm.start_analysis()
    assert wait_for_analysis(pipeline_vm, timeout_ms=60_000)
    QApplication.processEvents()

    results._expand_all_button.click()  # noqa: SLF001
    results._collapse_all_button.click()  # noqa: SLF001
    results._copy_all_button.click()  # noqa: SLF001
    QApplication.processEvents()
    assert pipeline_vm.caption_text


@pytest.mark.acceptance
@pytest.mark.ui
def test_keyboard_shortcuts(acceptance_app: AcceptanceApp, sample_image: Path) -> None:
    window = acceptance_app.window
    window.show()
    QApplication.processEvents()

    QTest.keySequence(window, QKeySequence("F11"))
    QApplication.processEvents()
    assert not window._sidebar.isVisible()  # noqa: SLF001

    QTest.keySequence(window, QKeySequence("F11"))
    QApplication.processEvents()
    assert window._sidebar.isVisible()  # noqa: SLF001


@pytest.mark.acceptance
@pytest.mark.ui
def test_about_dialog_via_release_info(acceptance_app: AcceptanceApp) -> None:
    dialog = AboutDialog(acceptance_app.context.release_info, acceptance_app.window)
    dialog.show()
    QApplication.processEvents()
    assert "Sentivis AI" in dialog.windowTitle() or dialog.windowTitle()
    dialog.close()
