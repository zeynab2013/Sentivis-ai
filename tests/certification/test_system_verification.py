"""Tests for system verification."""

from pathlib import Path

from certification.system_verification import SystemVerifier


def test_system_verification_end_to_end(tmp_path: Path) -> None:
    report = SystemVerifier().run(tmp_path)
    assert report.passed
    names = {check.name for check in report.checks}
    assert "application_startup" in names
    assert "export_workflow" in names
