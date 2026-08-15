"""Tests for production quality scanner."""

from certification.production_quality import ProductionQualityScanner


def test_production_quality_has_no_todo_markers() -> None:
    report = ProductionQualityScanner().scan(cleanup=False)
    errors = [f for f in report.findings if f.severity == "error"]
    assert not errors, errors
