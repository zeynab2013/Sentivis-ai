"""Tests for installation validation."""

from pathlib import Path

from certification.installation_validation import InstallationValidator


def test_clean_installation_simulation(tmp_path: Path) -> None:
    report = InstallationValidator().run(tmp_path / "install")
    assert report.passed
    assert any(check.name == "directory_creation" and check.passed for check in report.checks)
