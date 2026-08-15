"""Release metadata and build provenance."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

from release.version import (
    AI_PIPELINE_VERSION,
    APPLICATION_VERSION,
    ARCHITECTURE_VERSION,
    CONFIGURATION_VERSION,
    MODEL_REGISTRY_VERSION,
    PRODUCT_NAME,
    WEBSITE_PLACEHOLDER,
)


@dataclass(frozen=True)
class ReleaseInfo:
    """Complete version and build metadata for distribution."""

    application_name: str
    application_version: str
    architecture_version: str
    ai_pipeline_version: str
    model_registry_version: str
    configuration_version: str
    build_number: str
    git_commit: str
    build_timestamp: str
    build_profile: str
    website: str = WEBSITE_PLACEHOLDER

    @property
    def display_version(self) -> str:
        return f"{self.application_version} (build {self.build_number})"

    @property
    def full_version_line(self) -> str:
        return (
            f"{self.application_name} {self.display_version} · "
            f"Architecture v{self.architecture_version} · "
            f"Pipeline v{self.ai_pipeline_version}"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "application_name": self.application_name,
            "application_version": self.application_version,
            "architecture_version": self.architecture_version,
            "ai_pipeline_version": self.ai_pipeline_version,
            "model_registry_version": self.model_registry_version,
            "configuration_version": self.configuration_version,
            "build_number": self.build_number,
            "git_commit": self.git_commit,
            "build_timestamp": self.build_timestamp,
            "build_profile": self.build_profile,
            "website": self.website,
        }


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_timestamp() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch and epoch.isdigit():
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()  # noqa: UP017
    return datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017


def load_release_info(*, build_profile: str = "development") -> ReleaseInfo:
    """Load release metadata using environment overrides when present."""
    return ReleaseInfo(
        application_name=PRODUCT_NAME,
        application_version=os.environ.get("SENTIVIS_APP_VERSION", APPLICATION_VERSION),
        architecture_version=os.environ.get("SENTIVIS_ARCH_VERSION", ARCHITECTURE_VERSION),
        ai_pipeline_version=os.environ.get("SENTIVIS_PIPELINE_VERSION", AI_PIPELINE_VERSION),
        model_registry_version=os.environ.get(
            "SENTIVIS_MODEL_REGISTRY_VERSION", MODEL_REGISTRY_VERSION
        ),
        configuration_version=os.environ.get("SENTIVIS_CONFIG_VERSION", CONFIGURATION_VERSION),
        build_number=os.environ.get("SENTIVIS_BUILD_NUMBER", "0"),
        git_commit=os.environ.get("SENTIVIS_GIT_COMMIT", _git_commit()),
        build_timestamp=os.environ.get("SENTIVIS_BUILD_TIMESTAMP", _build_timestamp()),
        build_profile=build_profile,
    )
