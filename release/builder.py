"""Reproducible build orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from core.utils.paths import project_root
from release.build_profiles import BuildProfile, BuildProfileSpec, get_profile
from release.metadata import ReleaseInfo, load_release_info
from release.validator import BuildValidationReport, BuildValidator


class BuildError(RuntimeError):
    """Raised when build validation or packaging fails."""


class ReleaseBuilder:
    """Creates reproducible build artifacts for a profile."""

    _PACKAGE_DIRS = (
        "app",
        "core",
        "vision",
        "language",
        "analysis",
        "services",
        "ui",
        "streamlit_app",
        "model_management",
        "release",
        "certification",
        "acceptance",
    )

    @staticmethod
    def _ignore_copy(_directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if (
                name == "__pycache__"
                or name.endswith(".pyc")
                or name.endswith(".log")
                or name in {".pytest_cache", ".mypy_cache", ".ruff_cache"}
            ):
                ignored.add(name)
        return ignored

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or project_root()
        self._dist_root = self._root / "dist"

    def build(self, profile_name: str) -> Path:
        profile = get_profile(profile_name)
        release_info = load_release_info(build_profile=profile.name.value)
        report = BuildValidator(self._root).validate(profile, release_info)
        if not report.passed:
            details = "\n".join(f"- {error}" for error in report.errors)
            raise BuildError(f"Build validation failed:\n{details}")
        output_dir = self._create_output_dir(profile, release_info)
        self._copy_sources(output_dir, profile)
        self._copy_config(output_dir)
        self._copy_translations(output_dir)
        self._copy_assets(output_dir, profile)
        self._copy_docs(output_dir, profile)
        self._copy_release_resources(output_dir)
        self._write_manifest(output_dir, profile, release_info, report)
        return output_dir

    def _create_output_dir(self, profile: BuildProfileSpec, release_info: ReleaseInfo) -> Path:
        folder = (
            self._dist_root
            / profile.output_subdir
            / f"sentivis-ai-{release_info.application_version}-build{release_info.build_number}"
        )
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _copy_sources(self, output_dir: Path, profile: BuildProfileSpec) -> None:
        for package in self._PACKAGE_DIRS:
            source = self._root / package
            if source.is_dir():
                shutil.copytree(source, output_dir / package, ignore=self._ignore_copy)
        for file_name in ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in"):
            source = self._root / file_name
            if source.is_file():
                shutil.copy2(source, output_dir / file_name)

    def _copy_config(self, output_dir: Path) -> None:
        shutil.copytree(
            self._root / "config",
            output_dir / "config",
            ignore=self._ignore_copy,
        )

    def _copy_translations(self, output_dir: Path) -> None:
        """Ship root translations/ for release trees (package data also under core/)."""
        source = self._root / "translations"
        if source.is_dir():
            shutil.copytree(source, output_dir / "translations", ignore=self._ignore_copy)

    def _copy_assets(self, output_dir: Path, profile: BuildProfileSpec) -> None:
        assets = self._root / "assets"
        if assets.is_dir():
            shutil.copytree(assets, output_dir / "assets", ignore=self._ignore_copy)
        if profile.include_samples:
            samples = output_dir / "assets" / "samples"
            samples.mkdir(parents=True, exist_ok=True)

    def _copy_docs(self, output_dir: Path, profile: BuildProfileSpec) -> None:
        if not profile.include_docs:
            return
        shutil.copytree(
            self._root / "docs",
            output_dir / "docs",
            ignore=self._ignore_copy,
        )

    def _copy_release_resources(self, output_dir: Path) -> None:
        resources = self._root / "release" / "resources"
        bundle = output_dir / "installer"
        bundle.mkdir(parents=True, exist_ok=True)
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "installer_manifest.json"):
            shutil.copy2(resources / name, bundle / name)
        default_config = resources / "default_config"
        if default_config.is_dir():
            shutil.copytree(default_config, bundle / "config", ignore=self._ignore_copy)

    def _write_manifest(
        self,
        output_dir: Path,
        profile: BuildProfileSpec,
        release_info: ReleaseInfo,
        report: BuildValidationReport,
    ) -> None:
        manifest = {
            "release": release_info.to_dict(),
            "profile": profile.name.value,
            "validation": {
                "passed": report.passed,
                "errors": report.errors,
                "warnings": report.warnings,
            },
        }
        (output_dir / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )


def build_profile(profile: str) -> Path:
    """Build one profile and return the output directory."""
    return ReleaseBuilder().build(profile)


def available_profiles() -> tuple[str, ...]:
    return tuple(item.value for item in BuildProfile)
