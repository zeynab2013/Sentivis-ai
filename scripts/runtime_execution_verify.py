"""Execute real Sentivis AI runtime verification and write RUNTIME_EXECUTION_REPORT.md."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Skip blocking first-launch dialog when models are already installed.
os.environ.setdefault("SENTIVIS_TEST_MODE", "1")

from PIL import Image, ImageDraw
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.bootstrap import bootstrap
from core.utils.paths import project_root
from ui.models.pipeline_ui_state import PipelineUiState
from ui.view_models.export_view_model import ExportViewModel
from ui.view_models.pipeline_view_model import PipelineViewModel
from ui.view_models.settings_view_model import SettingsViewModel
from ui.widgets.settings_dialog import SettingsDialog


def wait_until(predicate, *, timeout_ms: int = 30_000, interval_ms: int = 50) -> bool:
    app = QApplication.instance()
    deadline = time.perf_counter() + (timeout_ms / 1000.0)
    while time.perf_counter() < deadline:
        if predicate():
            return True
        if app is not None:
            app.processEvents()
        time.sleep(interval_ms / 1000.0)
    return predicate()


def wait_for_analysis(pipeline_vm: object, *, timeout_ms: int = 60_000) -> bool:
    ui_state = getattr(pipeline_vm, "ui_state", None)
    return wait_until(
        lambda: getattr(pipeline_vm, "has_result", False)
        or ui_state in {PipelineUiState.COMPLETED, PipelineUiState.FAILED, PipelineUiState.CANCELLED},
        timeout_ms=timeout_ms,
    ) and getattr(pipeline_vm, "has_result", False)


@dataclass
class RuntimeReport:
    entry_point: str = "sentivis-ai (app.main:main → app.bootstrap.bootstrap)"
    startup_ok: bool = False
    window_visible: bool = False
    window_title: str = ""
    startup_errors: list[str] = field(default_factory=list)
    models: list[dict[str, str]] = field(default_factory=list)
    image_path: str = ""
    analysis_started: bool = False
    analysis_completed: bool = False
    analysis_seconds: float = 0.0
    caption: str = ""
    object_count: int = 0
    stages_executed: list[str] = field(default_factory=list)
    exports: list[dict[str, str | int]] = field(default_factory=list)
    ui_checks: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    remaining_issues: list[str] = field(default_factory=list)


def _sample_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 480), color=(40, 40, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 180, 520, 420), fill=(200, 40, 40))
    draw.ellipse((150, 360, 230, 440), fill=(20, 20, 20))
    draw.ellipse((410, 360, 490, 440), fill=(20, 20, 20))
    draw.ellipse((280, 100, 360, 200), fill=(220, 180, 140))
    image.save(path)
    return path


def run_verification() -> RuntimeReport:
    report = RuntimeReport()
    root = project_root()
    exports_dir = root / "runtime_verify_exports"
    exports_dir.mkdir(exist_ok=True)
    sample = _sample_image(root / "runtime_verify_sample.png")

    lifecycle = None
    try:
        lifecycle = bootstrap()
        window = lifecycle._window
        ctx = lifecycle._context
        report.startup_ok = True
        window.show()
        QApplication.processEvents()
        report.window_visible = window.isVisible()
        report.window_title = window.windowTitle()

        for record in ctx.model_registry.records:
            report.models.append(
                {
                    "kind": record.kind.name,
                    "status": str(record.runtime_status),
                    "location": str(record.file_location or record.identifier),
                    "detail": record.validation_detail,
                }
            )

        pipeline_vm = cast(PipelineViewModel, window._facade.pipeline_view_model)
        export_vm = cast(ExportViewModel, window._facade.export_view_model)
        ctx.main_controller.export._exports_dir = exports_dir

        window._load_image_path(str(sample))
        QApplication.processEvents()
        report.image_path = str(sample)
        report.ui_checks["image_viewer_loaded"] = pipeline_vm.image_path == sample

        sidebar = window._sidebar
        report.ui_checks["analyze_button_enabled"] = sidebar.analyze_button.isEnabled()
        if not sidebar.analyze_button.isEnabled():
            report.failures.append("Analyze button disabled — models may not be ready")

        started = time.perf_counter()
        report.analysis_started = True
        pipeline_vm.start_analysis()
        completed = wait_for_analysis(pipeline_vm, timeout_ms=1_800_000)
        report.analysis_seconds = time.perf_counter() - started
        report.analysis_completed = completed

        if not completed:
            report.failures.append(f"Analysis did not complete within {report.analysis_seconds:.0f}s")
        else:
            result = pipeline_vm.current_result
            if result is None:
                report.failures.append("Analysis completed but current_result is None")
            else:
                report.caption = result.caption.text
                report.object_count = result.scene_context.object_count
                report.stages_executed = [metric.stage.name for metric in result.metrics.stage_metrics]
                report.ui_checks["caption_displayed"] = bool(pipeline_vm.caption_text)
                report.ui_checks["objects_displayed"] = bool(pipeline_vm.objects_text)
                results_panel = window._dashboard.results_panel
                report.ui_checks["results_panel_caption"] = pipeline_vm.caption_text in results_panel._caption.plain_text
                if "gemma" not in result.caption.sources:
                    report.remaining_issues.append(
                        "Gemma inference used fallback caption — authenticate with Hugging Face (HF_TOKEN) "
                        "and accept the google/gemma-2-2b-it license"
                    )

        viewer = window._image_viewer
        viewer.zoom_in()
        viewer.zoom_out()
        viewer.fit_to_window()
        viewer._original_button.click()
        QApplication.processEvents()
        report.ui_checks["image_viewer_zoom"] = viewer._scale_factor > 0

        export_panel = window._dashboard.export_panel
        for fmt, button in [
            ("json", export_panel.export_json_button),
            ("txt", export_panel.export_txt_button),
            ("md", export_panel.export_md_button),
            ("pdf", export_panel.export_pdf_button),
        ]:
            if not report.analysis_completed:
                report.exports.append({"format": fmt, "path": "", "bytes": 0, "status": "skipped"})
                continue
            button.click()
            QApplication.processEvents()
            ok = wait_until(lambda: not pipeline_vm.exporting, timeout_ms=30_000)
            preview = export_vm.preview_path(fmt)
            if ok and preview and preview.exists():
                report.exports.append(
                    {"format": fmt, "path": str(preview), "bytes": preview.stat().st_size, "status": "ok"}
                )
            else:
                report.failures.append(f"Export {fmt} failed")
                report.exports.append({"format": fmt, "path": str(preview or ""), "bytes": 0, "status": "failed"})

        report.ui_checks["export_panel_enabled"] = export_panel.export_json_button.isEnabled()

        settings_vm = cast(SettingsViewModel, window._facade.settings_view_model)
        dialog = SettingsDialog(settings_vm, window)
        dialog.show()
        QApplication.processEvents()
        report.ui_checks["settings_dialog"] = dialog.windowTitle() == "Settings"
        dialog.reject()

        window._presentation_mode.toggle()
        QApplication.processEvents()
        report.ui_checks["presentation_mode_hide_sidebar"] = sidebar.isHidden()
        window._presentation_mode.toggle()
        QApplication.processEvents()
        report.ui_checks["presentation_mode_restore_sidebar"] = not sidebar.isHidden()

        QTest.keySequence(window, QKeySequence("Ctrl+F"))
        QApplication.processEvents()
        report.ui_checks["search_shortcut"] = window._dashboard.results_panel.search_field.hasFocus()

        copy_button = window._dashboard.results_panel._copy_all_button
        copy_button.click()
        QApplication.processEvents()
        report.ui_checks["copy_all_button"] = copy_button.isEnabled()

        report.ui_checks["history_widget"] = window._sidebar.history_list is not None
        report.ui_checks["notifications"] = window._notifications is not None
        report.ui_checks["progress_timeline"] = window._dashboard.stage_progress is not None

    except Exception as exc:
        report.failures.append(f"{type(exc).__name__}: {exc}")
        report.failures.append(traceback.format_exc())
    finally:
        if lifecycle is not None:
            lifecycle.shutdown()

    if report.failures:
        report.remaining_issues.extend(item for item in report.failures if item not in report.remaining_issues)
    return report


def write_markdown(report: RuntimeReport, path: Path) -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Sentivis AI — Runtime Execution Report",
        "",
        f"**Generated:** {ts}",
        f"**Entry point:** `{report.entry_point}`",
        "",
        "## Startup",
        "",
        f"- Status: {'SUCCESS' if report.startup_ok and report.window_visible else 'FAILED'}",
        f"- Window title: {report.window_title}",
        f"- Window visible: {report.window_visible}",
        "",
        "## Models",
        "",
        "| Kind | Status | Location | Detail |",
        "|------|--------|----------|--------|",
    ]
    for model in report.models:
        lines.append(
            f"| {model['kind']} | {model['status']} | {model['location']} | {model['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            f"- Image: `{report.image_path}`",
            f"- Started: {report.analysis_started}",
            f"- Completed: {report.analysis_completed}",
            f"- Duration: {report.analysis_seconds:.1f}s",
            f"- Caption: {report.caption[:200] + '...' if len(report.caption) > 200 else report.caption}",
            f"- Objects detected: {report.object_count}",
            f"- Stages: {', '.join(report.stages_executed) if report.stages_executed else 'n/a'}",
            "",
            "## Exports",
            "",
            "| Format | Status | Bytes | Path |",
            "|--------|--------|-------|------|",
        ]
    )
    for item in report.exports:
        lines.append(f"| {item['format']} | {item['status']} | {item['bytes']} | `{item['path']}` |")
    lines.extend(["", "## UI Verification", ""])
    for name, ok in sorted(report.ui_checks.items()):
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
    lines.extend(["", "## Failures", ""])
    if report.failures:
        for failure in report.failures:
            lines.append(f"- {failure}")
    else:
        lines.append("- None")
    lines.extend(["", "## Remaining Runtime Issues", ""])
    if report.remaining_issues:
        for issue in report.remaining_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- None — full analysis completed without exception.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = run_verification()
    out = project_root() / "docs" / "RUNTIME_EXECUTION_REPORT.md"
    write_markdown(report, out)
    (project_root() / "runtime_execution_report.json").write_text(
        json.dumps(asdict(report), indent=2),
        encoding="utf-8",
    )
    print(f"Report written to {out}")
    return 0 if report.startup_ok and report.analysis_completed and not report.failures else 1

if __name__ == "__main__":
    sys.exit(main())
