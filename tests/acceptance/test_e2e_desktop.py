"""Desktop application end-to-end acceptance tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from PySide6.QtWidgets import QApplication

from tests.acceptance.support.harness import AcceptanceApp
from tests.acceptance.support.wait import wait_for_analysis, wait_until
from ui.view_models.export_view_model import ExportViewModel
from ui.view_models.pipeline_view_model import PipelineViewModel


@pytest.mark.acceptance
@pytest.mark.e2e
@pytest.mark.ui
def test_desktop_load_image_and_run_analysis(acceptance_app: AcceptanceApp, sample_image: Path) -> None:
    window = acceptance_app.window
    pipeline_vm = cast(PipelineViewModel, window._facade.pipeline_view_model)  # noqa: SLF001

    window._load_image_path(str(sample_image))  # noqa: SLF001
    QApplication.processEvents()

    assert pipeline_vm.image_path == sample_image
    assert window._sidebar.analyze_button.isEnabled()  # noqa: SLF001

    pipeline_vm.start_analysis()
    completed = wait_for_analysis(pipeline_vm, timeout_ms=60_000)
    assert completed, "Analysis did not complete within timeout"

    result = pipeline_vm.current_result
    assert result is not None
    assert result.caption.text
    assert result.scene_context.object_count >= 2
    assert pipeline_vm.caption_text
    assert pipeline_vm.objects_text
    assert window._dashboard.export_panel.export_json_button.isEnabled()  # noqa: SLF001


@pytest.mark.acceptance
@pytest.mark.e2e
@pytest.mark.ui
def test_desktop_export_all_formats_from_ui(acceptance_app: AcceptanceApp, sample_image: Path, tmp_path: Path) -> None:
    window = acceptance_app.window
    pipeline_vm = cast(PipelineViewModel, window._facade.pipeline_view_model)  # noqa: SLF001
    export_vm = cast(ExportViewModel, window._facade.export_view_model)  # noqa: SLF001
    export_panel = window._dashboard.export_panel  # noqa: SLF001

    exports_dir = tmp_path / "ui_exports"
    exports_dir.mkdir()
    acceptance_app.context.main_controller.export._exports_dir = exports_dir  # noqa: SLF001

    window._load_image_path(str(sample_image))  # noqa: SLF001
    pipeline_vm.start_analysis()
    assert wait_for_analysis(pipeline_vm, timeout_ms=60_000)

    formats = [
        ("json", export_panel.export_json_button),
        ("txt", export_panel.export_txt_button),
        ("md", export_panel.export_md_button),
        ("pdf", export_panel.export_pdf_button),
    ]
    exported: list[Path] = []
    for fmt, button in formats:
        button.click()
        QApplication.processEvents()
        assert wait_until(lambda: not pipeline_vm.exporting, timeout_ms=10_000), f"{fmt} export timed out"
        preview = export_vm.preview_path(fmt)
        assert preview is not None
        assert preview.exists(), f"{fmt} file not written: {preview}"
        exported.append(preview)

    assert len(exported) == 4
    assert all(path.stat().st_size > 0 for path in exported)


@pytest.mark.acceptance
@pytest.mark.e2e
@pytest.mark.ui
def test_desktop_graceful_shutdown(acceptance_app: AcceptanceApp) -> None:
    window = acceptance_app.window
    window.show()
    QApplication.processEvents()
    window.close()
    QApplication.processEvents()
    acceptance_app.context.model_manager.release_all()
    acceptance_app.context.memory_manager.clear_gpu_cache()


@pytest.mark.acceptance
@pytest.mark.e2e
@pytest.mark.ui
def test_desktop_analysis_updates_results_panel(acceptance_app: AcceptanceApp, sample_image: Path) -> None:
    window = acceptance_app.window
    pipeline_vm = cast(PipelineViewModel, window._facade.pipeline_view_model)  # noqa: SLF001
    results = window._dashboard.results_panel  # noqa: SLF001

    window._load_image_path(str(sample_image))  # noqa: SLF001
    pipeline_vm.start_analysis()
    assert wait_for_analysis(pipeline_vm, timeout_ms=60_000)
    QApplication.processEvents()
    assert pipeline_vm.caption_text in results._caption.plain_text  # noqa: SLF001
