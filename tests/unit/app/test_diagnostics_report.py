"""Unit tests for diagnostics report export."""

from pathlib import Path

from app.settings_loader import load_application_settings
from app.startup.diagnostics_report import build_diagnostics_report
from app.startup.environment_probe import probe_environment
from app.startup.model_discovery import discover_models
from app.startup.stages import StartupReport, StartupStage
from core.utils.paths import project_root


def test_diagnostics_report_exports_json_and_text(tmp_path: Path) -> None:
    root = project_root()
    settings = load_application_settings()
    environment = probe_environment(
        project_root=root,
        models_dir=root / "models",
        config_paths=(
            root / "config" / "app.default.toml",
            root / "config" / "models.default.toml",
            root / "config" / "analysis.default.toml",
            root / "config" / "themes.default.toml",
        ),
    )
    models = discover_models(settings.model_config, settings.app_config.paths.models_dir)
    startup = StartupReport()
    startup.add_stage(StartupStage.ENVIRONMENT_VALIDATION, "ok", 0.0)
    report = build_diagnostics_report(
        settings,
        environment,
        models,
        startup,
        plugin_summary=("yolo", "blip", "gemma"),
    )
    json_path, text_path = report.write(tmp_path)
    assert json_path.exists()
    assert text_path.exists()
    assert "Sentivis AI" in text_path.read_text(encoding="utf-8")
