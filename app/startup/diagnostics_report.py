"""Exportable startup diagnostics report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.settings_loader import ApplicationSettings
from app.startup.environment_probe import EnvironmentReport
from app.startup.model_discovery import ModelDiscoveryReport
from app.startup.stages import StartupReport


@dataclass(frozen=True)
class DiagnosticsReport:
    """Complete startup diagnostics suitable for support and judging."""

    generated_at: str
    application_name: str
    application_version: str
    operating_system: str
    python_version: str
    gpu_name: str
    cuda_available: bool
    execution_device: str
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    configuration_summary: str
    plugin_summary: tuple[str, ...]
    model_entries: tuple[dict[str, object], ...]
    vision_model: str
    caption_model: str
    semantic_model: str
    startup_stages: tuple[dict[str, object], ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_text(self) -> str:
        lines = [
            "Sentivis AI — Diagnostics Report",
            "================================",
            f"Generated: {self.generated_at}",
            f"Application: {self.application_name} v{self.application_version}",
            "",
            "System",
            "------",
            f"OS: {self.operating_system}",
            f"Python: {self.python_version}",
            f"GPU name: {self.gpu_name}",
            f"CUDA available: {'Yes' if self.cuda_available else 'No'}",
            f"DEVICE: {self.execution_device}",
            f"RAM total: {self.ram_total_gb:.1f} GB",
            f"RAM available: {self.ram_available_gb:.1f} GB",
            f"Disk free: {self.disk_free_gb:.1f} GB",
            "",
            "Runtime models",
            "--------------",
            f"VISION MODEL: {self.vision_model}",
            f"CAPTION MODEL: {self.caption_model}",
            f"SEMANTIC MODEL: {self.semantic_model}",
            f"DEVICE: {self.execution_device}",
            "",
            "Configuration",
            "-------------",
            self.configuration_summary,
            "",
            "Plugins",
            "-------",
            *(f"- {item}" for item in self.plugin_summary),
            "",
            "Models",
            "------",
        ]
        for entry in self.model_entries:
            lines.append(
                f"- {entry['kind']}: {entry['identifier']} "
                f"({'available' if entry['available'] else 'missing'}) — {entry['detail']}"
            )
        if self.warnings:
            lines.extend(["", "Warnings", "--------", *self.warnings])
        if self.errors:
            lines.extend(["", "Errors", "------", *self.errors])
        return "\n".join(lines)

    def write(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "startup-diagnostics.json"
        text_path = directory / "startup-diagnostics.txt"
        json_path.write_text(self.to_json(), encoding="utf-8")
        text_path.write_text(self.to_text(), encoding="utf-8")
        return json_path, text_path


def build_diagnostics_report(
    settings: ApplicationSettings,
    environment: EnvironmentReport,
    models: ModelDiscoveryReport,
    startup: StartupReport,
    *,
    plugin_summary: tuple[str, ...],
) -> DiagnosticsReport:
    stage_payload: tuple[dict[str, object], ...] = tuple(
        {
            "stage": item.stage.value,
            "message": item.message,
            "duration_ms": round(item.duration_ms, 2),
            "warnings": list(item.warnings),
        }
        for item in startup.stages
    )
    model_payload = tuple(
        {
            "kind": entry.kind,
            "identifier": entry.identifier,
            "available": entry.available,
            "detail": entry.detail,
        }
        for entry in models.entries
    )
    warnings = tuple(
        [*environment.warnings, *models.warnings, *startup.warnings]
    )
    errors = tuple([*environment.errors, *startup.errors])
    by_kind = {str(entry.kind).lower(): entry.identifier for entry in models.entries}
    mc = settings.model_config
    vision_vlm = mc.florence.model_id or by_kind.get("florence", "") or mc.blip.model_id
    caption_model = vision_vlm
    semantic_model = mc.gemma.model_id or by_kind.get("gemma", "") or "unknown"
    yolo_id = mc.yolo.variant or by_kind.get("yolo", "") or "unknown"
    vision_model = f"YOLO={yolo_id}; VLM={vision_vlm}"
    return DiagnosticsReport(
        generated_at=datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        application_name=settings.app_config.app_name,
        application_version=settings.app_config.app_version,
        operating_system=environment.operating_system,
        python_version=environment.python_version,
        gpu_name=environment.gpu_name,
        cuda_available=environment.cuda_available,
        execution_device=getattr(environment, "execution_device", None)
        or ("cuda:0" if environment.cuda_available else "cpu"),
        ram_total_gb=environment.ram_total_gb,
        ram_available_gb=environment.ram_available_gb,
        disk_free_gb=environment.disk_free_gb,
        configuration_summary=settings.sources.summary(),
        plugin_summary=plugin_summary,
        model_entries=model_payload,
        vision_model=vision_model,
        caption_model=caption_model,
        semantic_model=semantic_model,
        startup_stages=stage_payload,
        warnings=warnings,
        errors=errors,
    )
