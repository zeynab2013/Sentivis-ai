"""Pre-build validation for release engineering."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

from core.utils.paths import project_root, resource_path
from release.build_profiles import BuildProfileSpec
from release.metadata import ReleaseInfo


@dataclass
class BuildValidationReport:
    """Outcome of release build validation."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.passed = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


class BuildValidator:
    """Validates mandatory components before packaging."""

    _REQUIRED_CONFIG = (
        "app.default.toml",
        "models.default.toml",
        "analysis.default.toml",
        "themes.default.toml",
    )

    _REQUIRED_DOCS = (
        "INSTALLATION_GUIDE.md",
        "QUICK_START_GUIDE.md",
        "USER_MANUAL.md",
        "TROUBLESHOOTING_GUIDE.md",
        "RELEASE_NOTES.md",
        "SYSTEM_REQUIREMENTS.md",
        "KNOWN_LIMITATIONS.md",
        "DIRECTORY_STRUCTURE.md",
    )

    _REQUIRED_DEPENDENCIES = (
        "PySide6",
        "torch",
        "transformers",
        "PIL",
        "psutil",
    )

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or project_root()

    def validate(self, profile: BuildProfileSpec, release_info: ReleaseInfo) -> BuildValidationReport:
        report = BuildValidationReport(passed=True)
        self._validate_python(report)
        self._validate_configuration(report)
        self._validate_entry_point(report)
        self._validate_dependencies(report)
        self._validate_runtime_assets(report)
        self._validate_models_config(report)
        self._validate_icons(report)
        self._validate_themes(report)
        self._validate_documentation(report, profile)
        self._validate_release_resources(report)
        if profile.require_git_commit and release_info.git_commit == "unknown":
            report.add_error("Release profile requires a git commit but none was detected")
        return report

    def _validate_python(self, report: BuildValidationReport) -> None:
        if sys.version_info < (3, 10) or sys.version_info >= (3, 11):
            report.add_warning(
                f"Python 3.10.x required for builds; found {sys.version.split()[0]}"
            )

    def _validate_configuration(self, report: BuildValidationReport) -> None:
        config_dir = self._root / "config"
        for name in self._REQUIRED_CONFIG:
            path = config_dir / name
            if not path.is_file():
                report.add_error(f"Missing configuration file: config/{name}")

    def _validate_entry_point(self, report: BuildValidationReport) -> None:
        main_path = self._root / "app" / "main.py"
        if not main_path.is_file():
            report.add_error("Application entry point missing: app/main.py")
            return
        spec = importlib.util.spec_from_file_location("app.main", main_path)
        if spec is None or spec.loader is None:
            report.add_error("Application entry point could not be loaded")

    def _validate_dependencies(self, report: BuildValidationReport) -> None:
        for module_name in self._REQUIRED_DEPENDENCIES:
            if importlib.util.find_spec(module_name) is None:
                report.add_warning(f"Optional dependency not installed in build environment: {module_name}")

    def _validate_runtime_assets(self, report: BuildValidationReport) -> None:
        for relative in ("icons", "samples", "export_templates"):
            path = resource_path(relative)
            if not path.exists():
                report.add_warning(f"Runtime asset directory missing and will be created: assets/{relative}")

    def _validate_models_config(self, report: BuildValidationReport) -> None:
        models_toml = self._root / "config" / "models.default.toml"
        if models_toml.is_file() and "model_id" not in models_toml.read_text(encoding="utf-8"):
            report.add_error("Model configuration appears incomplete")

    def _validate_icons(self, report: BuildValidationReport) -> None:
        icons_dir = resource_path("icons")
        if not any(icons_dir.glob("*")):
            report.add_error("Application icon assets missing in assets/icons")

    def _validate_themes(self, report: BuildValidationReport) -> None:
        theme_engine = self._root / "ui" / "themes" / "theme_engine.py"
        theme_tokens = self._root / "ui" / "design" / "tokens.py"
        if not theme_engine.is_file() or not theme_tokens.is_file():
            report.add_error("Theme engine or design tokens missing")

    def _validate_documentation(self, report: BuildValidationReport, profile: BuildProfileSpec) -> None:
        if not profile.include_docs:
            return
        docs_dir = self._root / "docs"
        for name in self._REQUIRED_DOCS:
            if not (docs_dir / name).is_file():
                report.add_error(f"Missing documentation: docs/{name}")

    def _validate_release_resources(self, report: BuildValidationReport) -> None:
        resources = self._root / "release" / "resources"
        required = ("LICENSE", "THIRD_PARTY_NOTICES.md", "installer_manifest.json")
        for name in required:
            if not (resources / name).is_file():
                report.add_error(f"Missing release resource: release/resources/{name}")
