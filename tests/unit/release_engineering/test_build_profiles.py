"""Unit tests for build profiles."""

from release.build_profiles import BuildProfile, get_profile


def test_get_profile_returns_spec() -> None:
    spec = get_profile("production")
    assert spec.name == BuildProfile.PRODUCTION
    assert not spec.include_dev_tools
