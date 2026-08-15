"""Pipeline presentation state."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.constants.pipeline_stages import PipelineStage
from core.contracts.pipeline import PipelineResult, StageProgress
from ui.controllers.pipeline_controller import PipelineController
from ui.formatters.result_formatters import (
    format_activities,
    format_attributes,
    format_confidence_analysis,
    format_detected_objects,
    format_environment,
    format_environment_brief,
    format_execution_metrics,
    format_image_quality,
    format_quality_report,
    format_reasoning_evidence,
    format_relationships,
    format_scene_description,
    format_scene_summary,
)
from ui.models.app_state import ApplicationState, application_state_message, resolve_application_state
from ui.models.operation_status import OperationStatus, operation_status_for_app_state
from ui.models.pipeline_ui_state import (
    UI_PROGRESS_STAGES,
    PipelineUiState,
    StageDisplayItem,
    StageStatus,
    ui_stage_for_pipeline_stage,
)
from ui.preferences.ui_preferences import (
    load_competition_mode,
    load_enable_enhancement,
    load_enable_sam2,
    load_enable_super_resolution,
)


class PipelineViewModel(QObject):
    """Manages pipeline UI state and delegates commands to controller."""

    progress_changed = Signal()
    state_changed = Signal()

    analysis_completed = Signal(object)
    analysis_failed = Signal(str)

    def __init__(self, controller: PipelineController) -> None:
        super().__init__()
        self._controller = controller
        self._ui_state = PipelineUiState.IDLE
        self._progress_percent = 0.0
        self._stage_label = ""
        self._status_message = "Ready"
        self._device_label = ""
        self._caption_text = ""
        self._narrative_text = ""
        self._short_caption_text = ""
        self._executive_text = ""
        self._scene_description_text = ""
        self._attributes_text = ""
        self._reasoning_text = ""
        self._confidence_text = ""
        self._scene_summary = ""
        self._objects_text = ""
        self._relationships_text = ""
        self._activities_text = ""
        self._environment_text = ""
        self._image_quality_text = ""
        self._quality_text = ""
        self._metrics_text = ""
        self._image_path: Path | None = None
        self._current_result: PipelineResult | None = None
        self._stage_items: tuple[StageDisplayItem, ...] = _initial_stage_items()
        self._stage_started_at: dict[str, float] = {}
        self._stage_durations: dict[str, float] = {}
        self._active_ui_stage: str | None = None
        self._exporting = False

        controller.progress_changed.connect(self._on_progress)
        controller.analysis_completed.connect(self._on_completed)
        controller.analysis_failed.connect(self._on_failed)

    @property
    def ui_state(self) -> PipelineUiState:
        return self._ui_state

    @property
    def progress_percent(self) -> float:
        return self._progress_percent

    @property
    def stage_label(self) -> str:
        return self._stage_label

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def device_label(self) -> str:
        return self._device_label

    @property
    def caption_text(self) -> str:
        return self._caption_text

    @property
    def narrative_text(self) -> str:
        return self._narrative_text

    @property
    def short_caption_text(self) -> str:
        return self._short_caption_text

    @property
    def executive_text(self) -> str:
        return self._executive_text

    @property
    def scene_description_text(self) -> str:
        return self._scene_description_text

    @property
    def attributes_text(self) -> str:
        return self._attributes_text

    @property
    def reasoning_text(self) -> str:
        return self._reasoning_text

    @property
    def confidence_text(self) -> str:
        return self._confidence_text

    @property
    def scene_summary(self) -> str:
        return self._scene_summary

    @property
    def objects_text(self) -> str:
        return self._objects_text

    @property
    def relationships_text(self) -> str:
        return self._relationships_text

    @property
    def activities_text(self) -> str:
        return self._activities_text

    @property
    def environment_text(self) -> str:
        return self._environment_text

    @property
    def image_quality_text(self) -> str:
        return self._image_quality_text

    @property
    def quality_text(self) -> str:
        return self._quality_text

    @property
    def metrics_text(self) -> str:
        return self._metrics_text

    @property
    def stage_items(self) -> tuple[StageDisplayItem, ...]:
        return self._stage_items

    @property
    def exporting(self) -> bool:
        return self._exporting

    @property
    def application_state(self) -> ApplicationState:
        return resolve_application_state(
            self._ui_state,
            has_image=self._image_path is not None,
            exporting=self._exporting,
        )

    @property
    def operation_status(self) -> OperationStatus:
        return operation_status_for_app_state(self.application_state.value)

    @property
    def is_analyzing(self) -> bool:
        return self._ui_state == PipelineUiState.RUNNING

    @property
    def has_result(self) -> bool:
        return self._current_result is not None

    @property
    def session_image_name(self) -> str:
        return self._image_path.name if self._image_path else ""

    @property
    def analysis_duration_ms(self) -> float | None:
        if self._current_result is None:
            return None
        return self._current_result.metrics.total_duration_ms

    @property
    def competition_mode_active(self) -> bool:
        if self._current_result is None:
            return False
        return self._current_result.metrics.competition_mode

    @property
    def model_configuration_hint(self) -> str:
        if self._device_label:
            from language.refinement.caption_refiner import ui_text

            return ui_text("status.active_device", "Active device: {device}", device=self._device_label)
        return "YOLO · BLIP · Gemma"

    @property
    def is_analyze_enabled(self) -> bool:
        return (
            self._image_path is not None
            and self._ui_state != PipelineUiState.RUNNING
            and not self._exporting
        )

    @property
    def is_cancel_enabled(self) -> bool:
        return self._ui_state == PipelineUiState.RUNNING

    @property
    def is_export_enabled(self) -> bool:
        return self._current_result is not None and self._ui_state == PipelineUiState.COMPLETED and not self._exporting

    def set_exporting(self, exporting: bool) -> None:
        self._exporting = exporting
        if exporting:
            self._status_message = application_state_message(ApplicationState.EXPORTING)
        self.state_changed.emit()

    @property
    def image_path(self) -> Path | None:
        return self._image_path

    @property
    def current_result(self) -> PipelineResult | None:
        return self._current_result

    def load_image(self, path: Path | None) -> None:
        resolved = path if path is not None and str(path).strip() and path != Path() else None
        if resolved is not None and not resolved.is_file():
            resolved = None
        self._image_path = resolved
        self._ui_state = PipelineUiState.IDLE
        self._current_result = None
        self._reset_result_text()
        self._status_message = application_state_message(self.application_state)
        self.state_changed.emit()

    def start_analysis(self) -> None:
        if not self._image_path or self._ui_state == PipelineUiState.RUNNING:
            return
        self._ui_state = PipelineUiState.RUNNING
        self._progress_percent = 0.0
        self._status_message = "Analyzing…"
        self._device_label = ""
        self._reset_stage_tracking()
        self._reset_result_text()
        self.state_changed.emit()
        self._controller.analyze_image(
            self._image_path,
            competition_mode=load_competition_mode(),
            enable_enhancement=load_enable_enhancement(),
            enable_super_resolution=load_enable_super_resolution(),
            enable_sam2=load_enable_sam2(),
        )

    def cancel_analysis(self) -> None:
        if self._ui_state != PipelineUiState.RUNNING:
            return
        self._controller.cancel()

    def _on_progress(self, event: object) -> None:
        if not isinstance(event, StageProgress):
            return
        self._progress_percent = event.percent
        self._stage_label = f"{event.stage.display_name}: {event.message}"
        if event.device:
            self._device_label = event.device
            from language.refinement.caption_refiner import ui_text

            self._status_message = ui_text(
                "status.analyzing_device", "Analyzing on {device}…", device=event.device
            )
        self._update_stage_progress(event.stage)
        self.progress_changed.emit()
        self.state_changed.emit()

    def _on_completed(self, result: object) -> None:
        if not isinstance(result, PipelineResult):
            return
        self._current_result = result
        self._ui_state = PipelineUiState.COMPLETED
        self._caption_text = result.caption.text
        # Primary narrative stays English scene prose; structured analysis is also English-only.
        self._narrative_text = result.caption.narrative_full or self._format_narrative(result)
        self._short_caption_text = result.caption.narrative_short or result.caption.text
        self._executive_text = result.caption.executive_summary or ""
        self._scene_description_text = format_scene_description(result)
        self._scene_summary = format_scene_summary(result)
        self._objects_text = format_detected_objects(result)
        self._attributes_text = format_attributes(result)
        self._relationships_text = format_relationships(result)
        self._activities_text = format_activities(result)
        self._environment_text = format_environment_brief(result) or format_environment(result)
        self._reasoning_text = format_reasoning_evidence(result)
        self._confidence_text = format_confidence_analysis(result)
        self._image_quality_text = format_image_quality(result)
        self._quality_text = format_quality_report(result)
        self._metrics_text = format_execution_metrics(result)
        self._progress_percent = 100.0
        from language.refinement.caption_refiner import ui_text

        self._status_message = ui_text("status.complete", "Analysis complete")
        self._complete_active_stage()
        self._mark_export_stage_complete()
        self.state_changed.emit()
        self.analysis_completed.emit(result)

    def _on_failed(self, message: str) -> None:
        self._ui_state = (
            PipelineUiState.CANCELLED if "cancel" in message.lower() else PipelineUiState.FAILED
        )
        from language.refinement.caption_refiner import ui_text

        self._status_message = (
            ui_text("status.cancelled", "Analysis cancelled")
            if self._ui_state == PipelineUiState.CANCELLED
            else ui_text("status.failed", "Analysis failed")
        )
        self._fail_active_stage()
        self.state_changed.emit()
        self.analysis_failed.emit(message)

    @staticmethod
    def _format_narrative(result: PipelineResult) -> str:
        full = result.caption.narrative_full.strip()
        short = result.caption.narrative_short.strip()
        if not full and not short:
            from language.refinement.caption_refiner import ui_text

            return ui_text("msg.narrative_unavailable", "Narrative caption unavailable.")
        from language.refinement.caption_refiner import ui_text

        return full or ui_text("msg.narrative_unavailable", "Narrative caption unavailable.")

    def _reset_result_text(self) -> None:
        self._caption_text = ""
        self._narrative_text = ""
        self._short_caption_text = ""
        self._executive_text = ""
        self._scene_description_text = ""
        self._attributes_text = ""
        self._reasoning_text = ""
        self._confidence_text = ""
        self._scene_summary = ""
        self._objects_text = ""
        self._relationships_text = ""
        self._activities_text = ""
        self._environment_text = ""
        self._image_quality_text = ""
        self._quality_text = ""
        self._metrics_text = ""

    def _reset_stage_tracking(self) -> None:
        self._stage_started_at.clear()
        self._stage_durations.clear()
        self._active_ui_stage = None
        self._stage_items = _initial_stage_items()

    def _update_stage_progress(self, stage: PipelineStage) -> None:
        ui_label = ui_stage_for_pipeline_stage(stage)
        if ui_label is None:
            return
        now = time.perf_counter()
        if self._active_ui_stage and self._active_ui_stage != ui_label:
            started = self._stage_started_at.get(self._active_ui_stage)
            if started is not None:
                self._stage_durations[self._active_ui_stage] = (now - started) * 1000.0
        if ui_label not in self._stage_started_at:
            self._stage_started_at[ui_label] = now
        self._active_ui_stage = ui_label
        self._stage_items = _build_stage_items(self._stage_durations, self._active_ui_stage, failed=False)

    def _complete_active_stage(self) -> None:
        if self._active_ui_stage:
            started = self._stage_started_at.get(self._active_ui_stage)
            if started is not None:
                self._stage_durations[self._active_ui_stage] = (time.perf_counter() - started) * 1000.0
        self._stage_items = _build_stage_items(self._stage_durations, None, failed=False, complete_all=True)

    def _fail_active_stage(self) -> None:
        self._stage_items = _build_stage_items(self._stage_durations, self._active_ui_stage, failed=True)

    def _mark_export_stage_complete(self) -> None:
        self._stage_durations["Export"] = 0.0
        self._stage_items = _build_stage_items(self._stage_durations, None, failed=False, complete_all=True)


def _initial_stage_items() -> tuple[StageDisplayItem, ...]:
    return tuple(
        StageDisplayItem(label=definition.label, status=StageStatus.PENDING)
        for definition in UI_PROGRESS_STAGES
    )


def _build_stage_items(
    durations: dict[str, float],
    active: str | None,
    *,
    failed: bool,
    complete_all: bool = False,
) -> tuple[StageDisplayItem, ...]:
    items: list[StageDisplayItem] = []
    reached_active = active is None
    for definition in UI_PROGRESS_STAGES:
        label = definition.label
        if complete_all:
            status = StageStatus.COMPLETED
        elif label == active:
            status = StageStatus.FAILED if failed else StageStatus.RUNNING
            reached_active = True
        elif not reached_active:
            status = StageStatus.COMPLETED
        else:
            status = StageStatus.PENDING
        duration = durations.get(label)
        items.append(StageDisplayItem(label=label, status=status, duration_ms=duration))
    return tuple(items)
