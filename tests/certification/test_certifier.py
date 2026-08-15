"""Tests for production certifier."""

from certification.certifier import ProductionCertifier


def test_production_certifier_skip_build() -> None:
    report = ProductionCertifier().certify(build=False, cleanup=False)
    assert report.system.passed
    assert report.installation.passed
    assert report.resources.passed
    assert report.quality.passed
    assert report.health.overall_score >= 85
