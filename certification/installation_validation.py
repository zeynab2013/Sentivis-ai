"""Clean installation simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from app.container import DependencyContainer
from app.settings_loader import load_application_settings
from app.startup.environment_probe import probe_environment
from app.startup.model_discovery import discover_models
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config
from core.utils.paths import project_root


@dataclass(frozen=True)
class InstallationCheck:
    """One clean-install verification step."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class InstallationValidationReport:
    """Simulated first-launch experience on a clean system."""

    checks: tuple[InstallationCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class InstallationValidator:
    """Simulates clean installation directory layout and first launch."""

    def run(self, install_root: Path) -> InstallationValidationReport:
        checks: list[InstallationCheck] = []
        app_config = load_app_config()
        isolated = replace(
            app_config,
            paths=replace(
                app_config.paths,
                cache_dir=install_root / "cache",
                logs_dir=install_root / "logs",
                exports_dir=install_root / "exports",
                models_dir=install_root / "models",
            ),
        )

        for directory in (
            isolated.paths.cache_dir,
            isolated.paths.logs_dir,
            isolated.paths.exports_dir,
            isolated.paths.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        checks.append(
            InstallationCheck(
                "directory_creation",
                all(path.is_dir() for path in (
                    isolated.paths.cache_dir,
                    isolated.paths.logs_dir,
                    isolated.paths.exports_dir,
                    isolated.paths.models_dir,
                )),
                "Runtime directories created",
            )
        )

        root = project_root()
        environment = probe_environment(
            project_root=root,
            models_dir=isolated.paths.models_dir,
            config_paths=(
                root / "config" / "app.default.toml",
                root / "config" / "models.default.toml",
                root / "config" / "analysis.default.toml",
                root / "config" / "themes.default.toml",
            ),
        )
        checks.append(
            InstallationCheck(
                "environment_validation",
                environment.config_files_ok,
                f"{len(environment.errors)} environment errors",
            )
        )

        settings = load_application_settings()
        checks.append(
            InstallationCheck(
                "configuration_loading",
                settings.app_config.app_name == "Sentivis AI",
                f"{len(settings.sources.sources)} config sources",
            )
        )

        models = discover_models(settings.model_config, isolated.paths.models_dir)
        checks.append(
            InstallationCheck(
                "model_discovery",
                len(models.entries) >= 3,
                f"{len(models.entries)} models discovered",
            )
        )

        context = DependencyContainer().build(
            isolated,
            load_model_config(),
            load_theme_config(),
            load_analysis_config(),
        )
        checks.append(
            InstallationCheck(
                "first_launch_services",
                context.runtime_status.health_score >= 70,
                f"health_score={context.runtime_status.health_score}",
            )
        )

        missing_models_ok = all(
            record.runtime_status.value
            not in {"unavailable", "validation_failed"}
            or (
                record.kind.name == "YOLO"
                and record.runtime_status.value == "validation_failed"
            )
            for record in context.model_registry.records
        )
        checks.append(
            InstallationCheck(
                "graceful_missing_models",
                missing_models_ok,
                "Missing local weights reported without fatal errors",
            )
        )

        return InstallationValidationReport(checks=tuple(checks))
