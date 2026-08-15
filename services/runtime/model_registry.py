"""Centralized model registry for runtime asset management."""

from __future__ import annotations

from pathlib import Path

from core.config.app_config import AppConfig
from core.config.model_config import ModelConfig
from core.constants.model_kinds import ModelKind
from core.constants.pipeline_stages import PipelineStage
from core.logging import get_logger
from model_management.catalog import spec_for_kind
from services.plugins.plugin_registry import PluginRegistry
from services.runtime.model_discovery import discover_model_files, resolve_model_search_paths
from services.runtime.model_record import ModelRecord
from services.runtime.model_status import InstallationStatus, IntegrityStatus, ModelRuntimeStatus
from services.runtime.model_validation import ModelValidationResult, ModelValidationService
from services.runtime.yolo_weights import resolve_yolo_weights_path

logger = get_logger(__name__)

_PLUGIN_BINDINGS = {
    ModelKind.YOLO: "detection_plugin",
    ModelKind.BLIP: "vision_language_plugin",
    ModelKind.GEMMA: "reasoning_plugin",
}


class CentralModelRegistry:
    """Tracks configured models, discovery results, validation, and runtime status."""

    def __init__(
        self,
        app_config: AppConfig,
        model_config: ModelConfig,
        plugin_registry: PluginRegistry,
        validator: ModelValidationService,
        *,
        extra_search_paths: tuple[Path, ...] = (),
    ) -> None:
        self._model_config = model_config
        self._plugin_registry = plugin_registry
        self._validator = validator
        self._search_paths = resolve_model_search_paths(
            app_config.paths.models_dir,
            extra_search_paths,
        )
        self._records: dict[ModelKind, ModelRecord] = {}
        self._discovery = discover_model_files(self._search_paths)
        self.refresh()

    @property
    def search_paths(self) -> tuple[str, ...]:
        return tuple(str(path) for path in self._search_paths)

    @property
    def records(self) -> tuple[ModelRecord, ...]:
        return tuple(self._records[kind] for kind in (ModelKind.YOLO, ModelKind.BLIP, ModelKind.GEMMA))

    def get(self, kind: ModelKind) -> ModelRecord:
        return self._records[kind]

    def refresh(self) -> None:
        """Rebuild registry entries from configuration and discovery."""
        self._discovery = discover_model_files(self._search_paths)
        plugin_ids = {
            ModelKind.YOLO: self._model_config.plugins.detection_plugin,
            ModelKind.BLIP: self._model_config.plugins.vision_language_plugin,
            ModelKind.GEMMA: self._model_config.plugins.reasoning_plugin,
        }
        for kind, plugin_id in plugin_ids.items():
            plugin = self._plugin_registry.get(plugin_id)
            descriptor = plugin.descriptor
            record = self._build_record(kind, descriptor.version, descriptor.supported_tasks)
            result = self._validator.validate(record, plugin_version=descriptor.version)
            self._records[kind] = result.record
            if not result.passed:
                logger.warning("Model validation issues for %s: %s", kind.name, result.failure_summary)

    def validate_before_inference(self, kind: ModelKind) -> ModelValidationResult:
        """Validate one model and update registry state."""
        plugin_id = getattr(self._model_config.plugins, _PLUGIN_BINDINGS[kind])
        plugin_version = self._plugin_registry.get(plugin_id).descriptor.version
        result = self._validator.validate(self._records[kind], plugin_version=plugin_version)
        self._records[kind] = result.record
        return result

    def update_runtime_status(self, kind: ModelKind, status: ModelRuntimeStatus) -> None:
        """Update lifecycle status after load/unload events."""
        current = self._records[kind]
        self._records[kind] = current.with_status(status)

    def _build_record(
        self,
        kind: ModelKind,
        plugin_version: str,
        supported_tasks: tuple[PipelineStage, ...],
    ) -> ModelRecord:
        if kind == ModelKind.YOLO:
            return self._build_yolo_record(plugin_version, supported_tasks)
        if kind == ModelKind.BLIP:
            return self._build_remote_record(
                kind,
                self._model_config.blip.model_id,
                f"BLIP ({self._model_config.blip.model_id.split('/')[-1]})",
                "Salesforce / Hugging Face",
                plugin_version,
                supported_tasks,
                (self._model_config.blip.preferred_device, "cpu"),
            )
        return self._build_remote_record(
            kind,
            self._model_config.gemma.model_id,
            f"Gemma ({self._model_config.gemma.model_id.split('/')[-1]})",
            "Google / Hugging Face",
            plugin_version,
            supported_tasks,
            (self._model_config.gemma.preferred_device, "cpu"),
        )

    def _build_yolo_record(
        self,
        plugin_version: str,
        supported_tasks: tuple[PipelineStage, ...],
    ) -> ModelRecord:
        file_location = resolve_yolo_weights_path(
            variant=self._model_config.yolo.variant,
            configured_path=self._model_config.yolo.weights_path,
            search_paths=self._search_paths,
            discovery=self._discovery,
        )
        installed = file_location is not None or bool(self._model_config.yolo.variant.strip())
        status = ModelRuntimeStatus.INSTALLED if installed else ModelRuntimeStatus.MISSING
        integrity = IntegrityStatus.UNKNOWN
        if file_location is not None and file_location.is_file():
            integrity = IntegrityStatus.VALID
        spec = spec_for_kind(ModelKind.YOLO)
        return ModelRecord(
            kind=ModelKind.YOLO,
            identifier=spec.model_id,
            display_name=spec.display_name,
            version=spec.version,
            provider=spec.provider,
            supported_tasks=tuple(task.name for task in supported_tasks),
            file_location=file_location,
            device_compatibility=(self._model_config.yolo.preferred_device, "cpu"),
            runtime_status=status,
            integrity_status=integrity,
            search_paths=self._search_paths,
            download_source=spec.download_source.value,
            expected_size_bytes=spec.expected_size_bytes,
            license_name=spec.license_name,
            installation_status=InstallationStatus.INSTALLED if file_location else InstallationStatus.NOT_INSTALLED,
            mandatory=spec.mandatory,
        )

    def _build_remote_record(
        self,
        kind: ModelKind,
        model_id: str,
        display_name: str,
        provider: str,
        plugin_version: str,
        supported_tasks: tuple[PipelineStage, ...],
        device_compatibility: tuple[str, ...],
    ) -> ModelRecord:
        configured = bool(model_id.strip())
        status = ModelRuntimeStatus.INSTALLED if configured else ModelRuntimeStatus.MISSING
        spec = spec_for_kind(kind)
        return ModelRecord(
            kind=kind,
            identifier=model_id,
            display_name=spec.display_name,
            version=spec.version,
            provider=spec.provider,
            supported_tasks=tuple(task.name for task in supported_tasks),
            file_location=None,
            device_compatibility=device_compatibility,
            runtime_status=status,
            integrity_status=IntegrityStatus.SKIPPED,
            search_paths=self._search_paths,
            download_source=spec.download_source.value,
            expected_size_bytes=spec.expected_size_bytes,
            license_name=spec.license_name,
            installation_status=InstallationStatus.NOT_INSTALLED,
            quantization=spec.quantization,
            mandatory=spec.mandatory,
        )

