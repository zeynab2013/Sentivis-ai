"""Unit tests for startup orchestrator."""

from app.startup.orchestrator import StartupOrchestrator


def test_startup_orchestrator_runs_all_stages() -> None:
    result = StartupOrchestrator().run()
    assert result.context.facade is not None
    assert len(result.report.stages) == 8
    assert result.diagnostics.application_name == "Sentivis AI"
    assert result.plugin_summary
