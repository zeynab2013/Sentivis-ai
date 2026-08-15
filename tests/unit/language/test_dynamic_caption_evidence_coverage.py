"""Dynamic caption detail: rich scenes expand; sparse scenes stay concise."""

from __future__ import annotations

from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from language.semantic.natural_caption_service import NaturalCaptionService


class _StubVision:
    def narrate(self, image: object, understanding: object) -> object:
        raise AssertionError("VLM must not be called in this unit test")


def _rich_kitchen_understanding() -> SceneUnderstanding:
    facts = (
        EvidenceFact("person #1", "is", "person", 0.92, "yolo"),
        EvidenceFact("person #2", "is", "person", 0.90, "yolo"),
        EvidenceFact("dining table #1", "is", "dining table", 0.93, "yolo"),
        EvidenceFact("chair #1", "is", "chair", 0.90, "yolo"),
        EvidenceFact("chair #2", "is", "chair", 0.88, "yolo"),
        EvidenceFact("refrigerator #1", "is", "refrigerator", 0.91, "yolo"),
        EvidenceFact("oven #1", "is", "oven", 0.86, "yolo"),
        EvidenceFact("sink #1", "is", "sink", 0.84, "yolo"),
        EvidenceFact("cup #1", "is", "cup", 0.88, "yolo"),
        EvidenceFact("vase #1", "is", "vase", 0.87, "yolo"),
        EvidenceFact("bowl #1", "is", "bowl", 0.85, "yolo"),
        EvidenceFact("person #1", "near", "dining table #1", 0.80, "relations"),
        EvidenceFact("cup #1", "on", "dining table #1", 0.82, "relations"),
        EvidenceFact("scene", "setting", "kitchen", 0.90, "environment"),
        EvidenceFact("person #1", "action", "dining", 0.70, "pose_estimator"),
    )
    return SceneUnderstanding(
        facts=facts,
        ranked_subjects=(
            "person #1",
            "person #2",
            "dining table #1",
            "chair #1",
            "chair #2",
            "refrigerator #1",
            "oven #1",
            "sink #1",
            "cup #1",
            "vase #1",
            "bowl #1",
            "scene",
        ),
        environment_keys=("indoor_outdoor=indoor", "setting=kitchen"),
        activity_keys=("dining",),
        ocr_text=(),
        evidence_brief=(
            "two people; dining table; chairs; refrigerator; oven; sink; "
            "cup; vase; bowl; kitchen"
        ),
        overall_confidence=0.88,
        discarded_count=0,
        contradictions_resolved=0,
    )


def _sparse_understanding() -> SceneUnderstanding:
    facts = (
        EvidenceFact("cup #1", "is", "cup", 0.9, "yolo"),
        EvidenceFact("scene", "setting", "indoor area", 0.6, "environment"),
    )
    return SceneUnderstanding(
        facts=facts,
        ranked_subjects=("cup #1", "scene"),
        environment_keys=("indoor_outdoor=indoor",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="cup indoor",
        overall_confidence=0.7,
        discarded_count=0,
        contradictions_resolved=0,
    )


def test_rich_kitchen_thin_spine_expands_with_verified_evidence() -> None:
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    understanding = _rich_kitchen_understanding()
    story = service._story_facts(understanding)
    scene = service._build_semantic_scene(understanding)
    brief = service._build_understanding_brief(story, scene, understanding)
    thin = "Two people are in a kitchen around a dining table."
    assert service._scene_richness(story) == "rich"
    assert service._needs_evidence_enrichment(thin, story, understanding)
    assert service._evidence_coverage_gaps(thin, understanding, story)

    expanded = service._competition_quality_pass(
        thin, understanding, story, scene, brief, "en"
    )
    lower = expanded.lower()
    assert len(expanded.split()) > len(thin.split()) + 15
    assert any(tok in lower for tok in ("person", "people"))
    assert "kitchen" in lower
    assert "table" in lower
    # At least some verified fixtures beyond the thin spine must appear.
    fixtures = ("chair", "refrigerator", "oven", "sink", "cup", "vase", "bowl")
    assert sum(1 for tok in fixtures if tok in lower) >= 3
    assert "observed activity:" not in lower
    assert "object inventory" not in lower
    assert "person 1" not in lower


def test_rich_kitchen_generate_keeps_verified_fixtures() -> None:
    """End-to-end generate must not collapse verified kitchen fixtures."""
    from tests.unit.language.test_broken_vlm_assembly_regression import (
        _BrokenFlorenceVision,
        _image,
        _kitchen_context,
        _kitchen_understanding,
    )

    caption = NaturalCaptionService(_BrokenFlorenceVision()).generate(  # type: ignore[arg-type]
        _image(), _kitchen_understanding(), context=_kitchen_context()
    )
    lower = caption.lower()
    assert len(caption.split()) >= 36
    assert "kitchen" in lower
    assert any(tok in lower for tok in ("person", "people"))
    assert "table" in lower
    assert "chair" in lower
    assert "refrigerator" in lower
    # Surface props must survive late assemble/gate dedupe.
    assert "vase" in lower or "cup" in lower
    assert "share the frame" not in lower
    assert "we have" not in lower
    assert "is also nearby" not in lower
    assert lower.count("dining table") <= 1


def test_sparse_scene_does_not_force_long_caption() -> None:
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    understanding = _sparse_understanding()
    story = service._story_facts(understanding)
    scene = service._build_semantic_scene(understanding)
    brief = service._build_understanding_brief(story, scene, understanding)
    short = "A cup sits on a surface indoors."
    assert service._scene_richness(story) in {"simple", "medium"}
    out = service._competition_quality_pass(
        short, understanding, story, scene, brief, "en"
    )
    # Must not invent people/activities or pad with filler.
    lower = out.lower()
    assert "two people" not in lower
    assert "this image shows" not in lower
    assert "the scene contains" not in lower
    assert len(out.split()) <= max(55, len(short.split()) + 25)


def test_order_puts_action_before_accessories() -> None:
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    messy = (
        "A handbag and a light blue handbag are visible behind them. "
        "A person is riding a bicycle. Two people are visible in the scene."
    )
    ordered = service._order_sentences_by_narrative_priority(messy)
    assert ordered.lower().index("riding") < ordered.lower().index("handbag")


def test_densify_merges_clothing_and_riding() -> None:
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    parts = service._densify_choppy_sentences(
        [
            "A single person wearing a red jersey and brown pants.",
            "A person is riding a motorcycle.",
        ]
    )
    joined = " ".join(parts).lower()
    assert "riding" in joined
    assert "red jersey" in joined
    assert joined.count(".") <= 2
