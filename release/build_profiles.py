"""Reproducible build profile definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BuildProfile(str, Enum):  # noqa: UP042
    """Supported packaging profiles."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    PORTABLE = "portable"
    RELEASE = "release"


@dataclass(frozen=True)
class BuildProfileSpec:
    """Configuration for one build profile."""

    name: BuildProfile
    include_dev_tools: bool
    include_docs: bool
    include_samples: bool
    optimize_bytecode: bool
    require_git_commit: bool
    output_subdir: str


PROFILE_SPECS: dict[BuildProfile, BuildProfileSpec] = {
    BuildProfile.DEVELOPMENT: BuildProfileSpec(
        name=BuildProfile.DEVELOPMENT,
        include_dev_tools=True,
        include_docs=True,
        include_samples=True,
        optimize_bytecode=False,
        require_git_commit=False,
        output_subdir="development",
    ),
    BuildProfile.PRODUCTION: BuildProfileSpec(
        name=BuildProfile.PRODUCTION,
        include_dev_tools=False,
        include_docs=True,
        include_samples=False,
        optimize_bytecode=True,
        require_git_commit=False,
        output_subdir="production",
    ),
    BuildProfile.PORTABLE: BuildProfileSpec(
        name=BuildProfile.PORTABLE,
        include_dev_tools=False,
        include_docs=True,
        include_samples=True,
        optimize_bytecode=True,
        require_git_commit=False,
        output_subdir="portable",
    ),
    BuildProfile.RELEASE: BuildProfileSpec(
        name=BuildProfile.RELEASE,
        include_dev_tools=False,
        include_docs=True,
        include_samples=True,
        optimize_bytecode=True,
        require_git_commit=True,
        output_subdir="release",
    ),
}


def get_profile(name: str) -> BuildProfileSpec:
    try:
        profile = BuildProfile(name.lower())
    except ValueError as exc:
        supported = ", ".join(item.value for item in BuildProfile)
        raise ValueError(f"Unknown build profile {name!r}. Supported: {supported}") from exc
    return PROFILE_SPECS[profile]
