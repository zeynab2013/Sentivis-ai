"""Unit tests for production model catalog."""

from core.constants.model_kinds import ModelKind
from model_management.catalog import PRODUCTION_MODELS, spec_for_kind


def test_production_models_use_required_ids() -> None:
    yolo = spec_for_kind(ModelKind.YOLO)
    blip = spec_for_kind(ModelKind.BLIP)
    gemma = spec_for_kind(ModelKind.GEMMA)
    assert yolo.model_id == "yolo11x"
    assert yolo.local_filename == "yolo11x.pt"
    assert blip.model_id == "Salesforce/blip-image-captioning-large"
    assert gemma.model_id == "google/gemma-2-2b-it"


def test_three_mandatory_production_models() -> None:
    assert len(PRODUCTION_MODELS) == 3
    assert all(spec.mandatory for spec in PRODUCTION_MODELS)
