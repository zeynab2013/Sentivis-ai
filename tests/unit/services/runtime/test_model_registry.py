"""Unit tests for central model registry."""

from app.container import DependencyContainer
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config
from core.constants.model_kinds import ModelKind


def test_central_model_registry_tracks_all_models() -> None:
    context = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    records = context.model_registry.records
    kinds = {record.kind for record in records}
    assert kinds == {ModelKind.YOLO, ModelKind.BLIP, ModelKind.GEMMA}
    assert all(record.display_name for record in records)
    assert all(record.version for record in records)


def test_runtime_status_provider_exposes_model_statuses() -> None:
    context = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    statuses = context.runtime_status.model_statuses()
    assert len(statuses) == 3
    assert context.runtime_status.health_score >= 0
