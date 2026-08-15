"""Tests for resource audit."""

from certification.resource_audit import ResourceAuditor


def test_resource_audit_passes() -> None:
    report = ResourceAuditor().audit()
    assert report.passed
    assert any(f.category == "icons" and f.passed for f in report.findings)
