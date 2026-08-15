"""Unit tests for runtime self-test."""

from app.container import DependencyContainer
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config


def test_self_test_returns_health_score() -> None:
    context = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    report = context.runtime_status.latest_self_test()
    assert 0 <= report.health_score <= 100
    assert report.checks
    assert any(check.name == "configuration" for check in report.checks)
