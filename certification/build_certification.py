"""Build profile certification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from release.build_profiles import PROFILE_SPECS, BuildProfile
from release.builder import BuildError, ReleaseBuilder
from release.metadata import load_release_info
from release.validator import BuildValidator


@dataclass(frozen=True)
class BuildProfileResult:
    """Certification outcome for one build profile."""

    profile: str
    validation_passed: bool
    build_passed: bool
    output_path: str | None
    detail: str


@dataclass(frozen=True)
class BuildCertificationReport:
    """Results for all build profiles."""

    results: tuple[BuildProfileResult, ...]

    @property
    def passed(self) -> bool:
        return all(item.validation_passed and item.build_passed for item in self.results)


class BuildCertifier:
    """Validates and builds every release profile."""

    _REQUIRED_ARTIFACTS = (
        "pyproject.toml",
        "config/app.default.toml",
        "assets/icons/app_icon.svg",
        "release/resources/LICENSE",
        "release/resources/THIRD_PARTY_NOTICES.md",
        "release/resources/installer_manifest.json",
        "docs/INSTALLATION_GUIDE.md",
        "build_manifest.json",
        "installer/LICENSE",
    )

    def certify(self, *, build: bool = True) -> BuildCertificationReport:
        results: list[BuildProfileResult] = []
        for profile in BuildProfile:
            results.append(self._certify_profile(profile.value, build=build))
        return BuildCertificationReport(results=tuple(results))

    def _certify_profile(self, profile_name: str, *, build: bool) -> BuildProfileResult:
        spec = PROFILE_SPECS[BuildProfile(profile_name)]
        if profile_name == "release":
            import os

            os.environ.setdefault("SENTIVIS_GIT_COMMIT", "certified")
        release_info = load_release_info(build_profile=profile_name)

        validation = BuildValidator().validate(spec, release_info)
        if not validation.passed:
            return BuildProfileResult(
                profile=profile_name,
                validation_passed=False,
                build_passed=False,
                output_path=None,
                detail="; ".join(validation.errors),
            )

        if not build:
            return BuildProfileResult(
                profile=profile_name,
                validation_passed=True,
                build_passed=True,
                output_path=None,
                detail="Validation passed",
            )

        try:
            output = ReleaseBuilder().build(profile_name)
        except BuildError as exc:
            return BuildProfileResult(
                profile=profile_name,
                validation_passed=True,
                build_passed=False,
                output_path=None,
                detail=str(exc),
            )

        missing = self._missing_artifacts(output)
        if missing:
            return BuildProfileResult(
                profile=profile_name,
                validation_passed=True,
                build_passed=False,
                output_path=str(output),
                detail=f"Missing artifacts: {', '.join(missing)}",
            )

        manifest = json.loads((output / "build_manifest.json").read_text(encoding="utf-8"))
        version = manifest.get("release", {}).get("application_version", "unknown")
        return BuildProfileResult(
            profile=profile_name,
            validation_passed=True,
            build_passed=True,
            output_path=str(output),
            detail=f"Built v{version}",
        )

    def _missing_artifacts(self, output_dir: Path) -> tuple[str, ...]:
        missing: list[str] = []
        for relative in self._REQUIRED_ARTIFACTS:
            if not (output_dir / relative).exists():
                missing.append(relative)
        return tuple(missing)
