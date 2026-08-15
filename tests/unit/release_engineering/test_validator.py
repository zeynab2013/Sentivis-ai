"""Unit tests for build validation."""

from release.build_profiles import BuildProfile
from release.metadata import load_release_info
from release.validator import BuildValidator


def test_build_validator_passes_for_development_profile() -> None:
    profile = BuildProfile.DEVELOPMENT
    from release.build_profiles import PROFILE_SPECS

    report = BuildValidator().validate(PROFILE_SPECS[profile], load_release_info())
    assert report.passed, report.errors
