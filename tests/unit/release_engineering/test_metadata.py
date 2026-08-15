"""Unit tests for release metadata."""

from release.metadata import load_release_info


def test_load_release_info_contains_all_fields() -> None:
    info = load_release_info(build_profile="development")
    assert info.application_name == "Sentivis AI"
    assert info.application_version
    assert info.architecture_version == "2.3"
    assert info.build_number
    assert info.build_timestamp
    assert "build" in info.display_version
