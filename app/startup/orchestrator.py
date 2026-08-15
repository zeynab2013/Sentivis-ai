"""Startup orchestrator for production infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.container import ApplicationContext, DependencyContainer
from app.settings_loader import ApplicationSettings, load_application_settings
from app.startup.diagnostics_report import DiagnosticsReport, build_diagnostics_report
from app.startup.environment_probe import EnvironmentReport, probe_environment
from app.startup.model_discovery import ModelDiscoveryReport, discover_models
from app.startup.recovery import recovery_message
from app.startup.stages import StartupReport, StartupStage
from core.exceptions.config import ConfigurationError
from core.logging import configure_logging, get_logger
from core.utils.paths import project_root

logger = get_logger(__name__)


@dataclass(frozen=True)
class StartupResult:
    """Outcome of the startup orchestrator."""

    settings: ApplicationSettings
    context: ApplicationContext
    report: StartupReport
    environment: EnvironmentReport
    models: ModelDiscoveryReport
    diagnostics: DiagnosticsReport
    plugin_summary: tuple[str, ...]


class StartupOrchestrator:
    """Runs structured startup stages with recovery and diagnostics."""

    def run(self) -> StartupResult:
        startup_report = StartupReport()
        root = project_root()
        config_paths = (
            root / "config" / "app.default.toml",
            root / "config" / "models.default.toml",
            root / "config" / "analysis.default.toml",
            root / "config" / "themes.default.toml",
        )

        started = perf_counter()
        environment = probe_environment(
            project_root=root,
            models_dir=root / "models",
            config_paths=config_paths,
        )
        startup_report.add_stage(
            StartupStage.ENVIRONMENT_VALIDATION,
            "Environment validated",
            started,
            warnings=environment.warnings,
        )
        for error in environment.errors:
            startup_report.add_error(error)
            logger.error("%s — %s", error, recovery_message(error))

        started = perf_counter()
        try:
            settings = load_application_settings()
        except ConfigurationError as exc:
            startup_report.add_error(str(exc))
            logger.error("Configuration loading failed: %s", exc)
            raise
        startup_report.add_stage(
            StartupStage.CONFIGURATION_LOADING,
            f"Loaded {len(settings.sources.sources)} configuration sources",
            started,
        )

        started = perf_counter()
        dependency_warnings: list[str] = []
        try:
            import torch  # noqa: F401
        except ImportError:
            dependency_warnings.append("PyTorch not installed")
        try:
            import transformers  # noqa: F401
        except ImportError:
            dependency_warnings.append("Transformers not installed")
        try:
            import cv2  # noqa: F401
        except ImportError:
            dependency_warnings.append("OpenCV (cv2) not installed")
        try:
            import streamlit  # noqa: F401
        except ImportError:
            dependency_warnings.append("Streamlit not installed")
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            dependency_warnings.append("Ultralytics (YOLO) not installed")
        startup_report.add_stage(
            StartupStage.DEPENDENCY_VERIFICATION,
            "Core dependencies verified",
            started,
            warnings=tuple(dependency_warnings),
        )
        startup_report.warnings.extend(dependency_warnings)

        started = perf_counter()
        models = discover_models(settings.model_config, settings.app_config.paths.models_dir)
        startup_report.add_stage(
            StartupStage.MODEL_DISCOVERY,
            f"Discovered {len(models.entries)} configured models",
            started,
            warnings=models.warnings,
        )
        startup_report.warnings.extend(models.warnings)

        started = perf_counter()
        plugin_ids = (
            settings.model_config.plugins.detection_plugin,
            settings.model_config.plugins.vision_language_plugin,
            settings.model_config.plugins.reasoning_plugin,
        )
        startup_report.add_stage(
            StartupStage.PLUGIN_DISCOVERY,
            f"Configured plugin IDs: {', '.join(plugin_ids)}",
            started,
        )

        started = perf_counter()
        for directory in (
            settings.app_config.paths.cache_dir,
            settings.app_config.paths.exports_dir,
            settings.app_config.paths.logs_dir,
            settings.app_config.paths.models_dir,
            settings.app_config.paths.cache_dir.parent / "tmp",
            root / "assets" / "user",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        configure_logging(settings.app_config.logging, settings.app_config.paths.logs_dir)
        logger.info("Starting Sentivis AI")
        logger.info(
            "Device probe: CUDA=%s | models discovered=%d | warnings=%d",
            environment.cuda_available,
            len(models.entries),
            len(models.warnings),
        )
        for entry in models.entries:
            if entry.kind in {"yolo", "florence2", "gemma", "sam2", "ollama"}:
                status = "available" if entry.available else "missing/disabled"
                logger.info("Model check: %s=%s (%s)", entry.kind, status, entry.identifier)
        startup_report.add_stage(
            StartupStage.RESOURCE_INITIALIZATION,
            "Runtime directories and logging initialized",
            started,
        )

        started = perf_counter()
        startup_report.add_stage(
            StartupStage.THEME_INITIALIZATION,
            f"Theme ready: {settings.theme_config.name}",
            started,
        )

        context = DependencyContainer().build(
            settings.app_config,
            settings.model_config,
            settings.theme_config,
            settings.analysis_config,
        )
        plugin_names = tuple(
            descriptor.identifier for descriptor in context.plugin_registry.list_plugins()
        )

        started = perf_counter()
        startup_report.add_stage(
            StartupStage.APPLICATION_READY,
            f"{settings.app_config.app_name} v{settings.app_config.app_version} ready",
            started,
        )
        logger.info("Startup complete with %d warnings", len(startup_report.warnings))

        diagnostics = build_diagnostics_report(
            settings,
            environment,
            models,
            startup_report,
            plugin_summary=plugin_names,
        )
        json_path, text_path = diagnostics.write(settings.app_config.paths.logs_dir)
        logger.info("Diagnostics exported to %s and %s", json_path.name, text_path.name)

        return StartupResult(
            settings=settings,
            context=context,
            report=startup_report,
            environment=environment,
            models=models,
            diagnostics=diagnostics,
            plugin_summary=plugin_names,
        )
