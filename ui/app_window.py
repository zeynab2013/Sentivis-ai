"""Main application window."""

from datetime import datetime
from pathlib import Path

from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QMainWindow, QSplitter, QWidget

from core.contracts.pipeline import PipelineResult
from ui.components.notification import NotificationCenter
from ui.interfaces.application_facade import IApplicationFacade
from ui.models.presentation_mode import PresentationModeController
from ui.view_models.export_view_model import ExportViewModel
from ui.view_models.history_view_model import HistoryViewModel
from ui.view_models.pipeline_view_model import PipelineViewModel
from ui.view_models.settings_view_model import SettingsViewModel
from ui.widgets.dashboard import AnalysisDashboardWidget
from ui.widgets.error_dialog import ErrorDialog
from ui.widgets.image_viewer import ImageViewerWidget
from ui.widgets.settings_dialog import SettingsDialog
from ui.widgets.sidebar import SidebarWidget


class AppWindow(QMainWindow):
    """Primary Sentivis AI desktop shell."""

    def __init__(self, facade: IApplicationFacade) -> None:
        super().__init__()
        self._facade = facade
        self._pipeline_vm: PipelineViewModel = facade.pipeline_view_model  # type: ignore[assignment]
        self._export_vm: ExportViewModel = facade.export_view_model  # type: ignore[assignment]
        self._history_vm: HistoryViewModel = facade.history_view_model  # type: ignore[assignment]
        self._settings_vm: SettingsViewModel = facade.settings_view_model  # type: ignore[assignment]
        self._presentation_mode = PresentationModeController()
        self._pending_export_format = ""
        self._splitter: QSplitter | None = None

        self.setWindowTitle(facade.window_title)
        self.resize(1440, 900)
        self.setMinimumSize(1100, 720)
        facade.theme_manager.apply(self)

        self._sidebar = SidebarWidget()
        self._image_viewer = ImageViewerWidget()
        self._dashboard = AnalysisDashboardWidget()
        self._notifications = NotificationCenter(self)

        self._build_layout()
        self._wire_shortcuts()
        self._wire_signals()
        from ui.media import is_online

        self._sidebar.random_button.setEnabled(is_online())
        self._sync_state()

    def _build_layout(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._splitter = QSplitter()
        self._splitter.addWidget(self._image_viewer)
        self._splitter.addWidget(self._dashboard)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)

        root.addWidget(self._sidebar)
        root.addWidget(self._splitter, stretch=1)

    def _wire_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+O"), self, self._open_image)
        QShortcut(QKeySequence("Ctrl+R"), self, self._pipeline_vm.start_analysis)
        QShortcut(QKeySequence("Escape"), self, self._pipeline_vm.cancel_analysis)
        QShortcut(QKeySequence("Ctrl+,"), self, self._show_settings)
        QShortcut(QKeySequence("Ctrl+F"), self, self._focus_results_search)
        QShortcut(QKeySequence("Ctrl++"), self, self._image_viewer.zoom_in)
        QShortcut(QKeySequence("Ctrl+="), self, self._image_viewer.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, self._image_viewer.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self._image_viewer.fit_to_window)
        QShortcut(QKeySequence("F11"), self, self._presentation_mode.toggle)

    def _wire_signals(self) -> None:
        self._sidebar.open_button.clicked.connect(self._open_image)
        self._sidebar.random_button.clicked.connect(self._load_random_image)
        self._sidebar.clear_button.clicked.connect(self._clear_session)
        self._sidebar.analyze_button.clicked.connect(self._pipeline_vm.start_analysis)
        self._sidebar.cancel_button.clicked.connect(self._pipeline_vm.cancel_analysis)
        self._sidebar.settings_button.clicked.connect(self._show_settings)
        self._sidebar.presentation_button.clicked.connect(self._presentation_mode.toggle)
        self._presentation_mode.mode_changed.connect(self._apply_presentation_mode)

        export_panel = self._dashboard.export_panel
        export_panel.export_json_button.clicked.connect(lambda: self._begin_export("json"))
        export_panel.export_txt_button.clicked.connect(lambda: self._begin_export("txt"))
        export_panel.export_md_button.clicked.connect(lambda: self._begin_export("md"))
        export_panel.export_pdf_button.clicked.connect(lambda: self._begin_export("pdf"))

        self._image_viewer.image_dropped.connect(self._load_image_path)
        self._pipeline_vm.progress_changed.connect(self._sync_state)
        self._pipeline_vm.state_changed.connect(self._sync_state)
        self._pipeline_vm.analysis_completed.connect(self._on_analysis_completed)
        self._pipeline_vm.analysis_failed.connect(self._on_analysis_failed)
        self._export_vm.export_status_changed.connect(self._on_export_status)
        self._export_vm.export_started.connect(self._on_export_started)
        self._export_vm.export_finished.connect(self._on_export_finished)
        self._history_vm.history_changed.connect(self._sync_history)
        self._settings_vm.settings_changed.connect(self._retranslate_ui)
        from ui.i18n.translator import get_translator

        get_translator().language_changed.connect(lambda _: self._retranslate_ui())

    def _focus_results_search(self) -> None:
        if not self._presentation_mode.enabled:
            self._dashboard.results_panel.search_field.setFocus()

    def _apply_presentation_mode(self, enabled: bool) -> None:
        self._sidebar.setVisible(not enabled)
        self._dashboard.set_presentation_mode(enabled)
        self._image_viewer.set_presentation_mode(enabled)
        if self._splitter is not None:
            self._splitter.setStretchFactor(0, 4 if enabled else 3)
            self._splitter.setStretchFactor(1, 1 if enabled else 2)
        label = "Presentation mode enabled" if enabled else "Presentation mode disabled"
        self._notifications.show_info(label)
        self._sync_state()

    def _begin_export(self, export_format: str) -> None:
        self._pending_export_format = export_format
        preview = self._export_vm.preview_path(export_format)
        self._dashboard.export_panel.set_destination_preview(preview)
        if export_format == "json":
            self._export_vm.export_json()
        elif export_format == "txt":
            self._export_vm.export_txt()
        elif export_format == "md":
            self._export_vm.export_md()
        else:
            self._export_vm.export_pdf()

    def _open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if file_path:
            self._load_image_path(file_path)

    def _load_image_path(self, file_path: str, *, auto_analyze: bool = False) -> None:
        path = Path(file_path)
        self._image_viewer.set_loading(True)
        self._pipeline_vm.load_image(path)
        self._image_viewer.show_image(path)
        self._image_viewer.clear_overlays()
        self._image_viewer.set_loading(False)
        self._notifications.show_info(f"Loaded {path.name}")
        self._sync_state()
        if auto_analyze and self._pipeline_vm.is_analyze_enabled:
            self._pipeline_vm.start_analysis()

    def _load_random_image(self) -> None:
        from ui.media import fetch_random_public_image, is_online

        if not is_online():
            self._sidebar.random_button.setEnabled(False)
            self._notifications.show_warning("Offline — Random Image is unavailable.")
            return
        try:
            self._sidebar.random_button.setEnabled(False)
            path = fetch_random_public_image()
            self._load_image_path(str(path), auto_analyze=True)
        except Exception as exc:
            self._notifications.show_error(f"Random image failed: {exc}")
        finally:
            self._sidebar.random_button.setEnabled(is_online())

    def _clear_session(self) -> None:
        self._pipeline_vm.load_image(None)
        self._image_viewer.clear_image()
        self._notifications.show_info("Session cleared.")
        self._sync_state()

    def _sync_state(self) -> None:
        show_dev_details = not self._presentation_mode.enabled
        self._dashboard.stage_progress.bind(
            self._pipeline_vm.stage_items,
            is_analyzing=self._pipeline_vm.is_analyzing,
            stage_label=self._pipeline_vm.stage_label if show_dev_details else "",
        )
        self._dashboard.results_panel.bind(
            progress_percent=self._pipeline_vm.progress_percent,
            status_message=self._pipeline_vm.status_message,
            stage_label=self._pipeline_vm.stage_label if show_dev_details else "",
            has_result=self._pipeline_vm.has_result,
            is_analyzing=self._pipeline_vm.is_analyzing,
            caption=self._pipeline_vm.caption_text,
            narrative_text=self._pipeline_vm.narrative_text,
            short_caption_text=self._pipeline_vm.short_caption_text,
            executive_text=self._pipeline_vm.executive_text,
            scene_description_text=self._pipeline_vm.scene_description_text,
            attributes_text=self._pipeline_vm.attributes_text,
            reasoning_text=self._pipeline_vm.reasoning_text,
            confidence_text=self._pipeline_vm.confidence_text,
            scene_summary=self._pipeline_vm.scene_summary,
            objects_text=self._pipeline_vm.objects_text,
            relationships_text=self._pipeline_vm.relationships_text,
            activities_text=self._pipeline_vm.activities_text,
            environment_text=self._pipeline_vm.environment_text,
            image_quality_text=self._pipeline_vm.image_quality_text,
            quality_text=self._pipeline_vm.quality_text,
            metrics_text=self._pipeline_vm.metrics_text,
        )
        device = self._pipeline_vm.device_label if show_dev_details else ""
        status = self._pipeline_vm.status_message
        self._sidebar.bind_status(self._pipeline_vm.operation_status, status, device)
        self._sidebar.bind_session(
            image_name=self._pipeline_vm.session_image_name,
            duration_ms=self._pipeline_vm.analysis_duration_ms,
            competition_mode=self._pipeline_vm.competition_mode_active,
            model_hint=self._pipeline_vm.model_configuration_hint if show_dev_details else "",
            presentation_mode=self._presentation_mode.enabled,
        )
        self._sidebar.analyze_button.setEnabled(self._pipeline_vm.is_analyze_enabled)
        self._sidebar.cancel_button.setEnabled(self._pipeline_vm.is_cancel_enabled)
        export_enabled = self._pipeline_vm.is_export_enabled
        self._dashboard.export_panel.set_enabled(export_enabled)
        if export_enabled:
            default_format = "pdf" if self._presentation_mode.enabled else "json"
            self._dashboard.export_panel.set_destination_preview(self._export_vm.preview_path(default_format))

    def _sync_history(self) -> None:
        self._sidebar.bind_history(self._history_vm.entries)

    def _on_analysis_completed(self, result: object) -> None:
        if isinstance(result, PipelineResult):
            preview = (result.caption.narrative_short or result.caption.text)[:80]
            analyzed_at = datetime.now().strftime("%H:%M:%S")
            duration_ms = result.metrics.total_duration_ms
            self._history_vm.add_entry(
                result.request.image_path.name,
                preview,
                analyzed_at=analyzed_at,
                duration_ms=duration_ms,
            )
            self._image_viewer.show_result_overlays(result)
            from ui.preferences.ui_preferences import load_comparison_mode

            self._image_viewer.set_comparison_mode(load_comparison_mode())
            self._notifications.show_success(
                f"Analysis complete in {duration_ms:.0f} ms — {result.request.image_path.name}"
            )
            if result.warnings and not self._presentation_mode.enabled:
                self._notifications.show_warning(
                    "Analysis completed with notices: " + "; ".join(result.warnings[:3])
                )
        self._sync_state()

    def _on_analysis_failed(self, message: str) -> None:
        if "cancel" in message.lower():
            self._notifications.show_warning("Analysis cancelled.")
        else:
            ErrorDialog.show_pipeline_error(self, message)
            self._notifications.show_error("Analysis failed. Review the message and try again.")
        self._sync_state()

    def _on_export_started(self, path: str) -> None:
        self._dashboard.export_panel.set_destination_preview(Path(path))
        self._dashboard.export_panel.set_exporting(True)
        self._notifications.show_info(f"Exporting to {Path(path).name}…")
        self._sync_state()

    def _on_export_finished(self) -> None:
        self._dashboard.export_panel.set_exporting(False)
        self._sync_state()

    def _on_export_status(self, message: str) -> None:
        if message.startswith("confirm_overwrite:"):
            path = message.split(":", maxsplit=1)[1]
            if ErrorDialog.confirm_overwrite(self, path) and self._pending_export_format:
                self._export_vm.confirm_export(self._pending_export_format)
            return
        if message.endswith((".json", ".txt", ".pdf", ".md")):
            self._sidebar.bind_status(
                self._pipeline_vm.operation_status,
                f"Exported to {Path(message).name}",
                self._pipeline_vm.device_label,
            )
            self._notifications.show_success(f"Export saved — {Path(message).name}")
        else:
            ErrorDialog.show_export_error(self, message)
            self._notifications.show_error(message)

    def _show_settings(self) -> None:
        dialog = SettingsDialog(
            self._settings_vm,
            self,
            presentation_mode=self._presentation_mode.enabled,
        )
        if dialog.exec():
            self._settings_vm.set_theme(dialog.selected_theme(), self)
            self._settings_vm.set_language(dialog.selected_language())
            self._settings_vm.set_competition_mode(dialog.selected_competition_mode())
            self._settings_vm.set_enhancement(dialog.selected_enhancement())
            self._settings_vm.set_super_resolution(dialog.selected_super_resolution())
            self._settings_vm.set_sam2(dialog.selected_sam2())
            self._settings_vm.set_comparison_mode(dialog.selected_comparison_mode())
            self._settings_vm.set_high_contrast(dialog.selected_high_contrast(), self)
            self._settings_vm.set_large_font(dialog.selected_large_font(), self)
            self._retranslate_ui()

    def _retranslate_ui(self) -> None:
        self._sidebar.retranslate_ui()
        self._image_viewer.retranslate_ui()
        self._dashboard.retranslate_ui()
        self._sync_state()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        super().resizeEvent(event)
        self._notifications.raise_()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        self._notifications.setParent(None)
        super().closeEvent(event)
