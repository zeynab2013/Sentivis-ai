"""Final naturalness gate: semantic dedupe, entity/setting repeats, unsupported relations."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

from core.contracts.analysis import (
    ActivityHints,
    AttributeSet,
    EnvironmentInfo,
    Relation,
    SceneContext,
    SceneGraph,
    SceneNode,
)
from core.contracts.image import PreprocessedImage, ValidatedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from language.refinement.caption_refiner import clear_ui_language_cache
from language.semantic.natural_caption_service import NaturalCaptionService
from tests.unit.language.test_caption_quality_overhaul import (
    _context,
    _multi_person_understanding,
)
from tests.unit.language.test_indoor_person_retention import (
    _indoor_context,
    _indoor_understanding,
)


@pytest.fixture(autouse=True)
def _english() -> None:
    os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
    clear_ui_language_cache()
    yield
    clear_ui_language_cache()


class _QuietVision:
    def narrate(self, image: object, understanding: object) -> RawCaption:
        return RawCaption(text="", source="stub", confidence=0.0)


class _StubVision:
    def narrate(self, image: object, understanding: object) -> RawCaption:
        return RawCaption(text="indoor room", source="stub", confidence=0.4)


def _image(name: str = "scene.jpg") -> PreprocessedImage:
    pixels = np.zeros((48, 48, 3), dtype=np.uint8)
    source = ValidatedImage(
        path=Path(name),
        width=48,
        height=48,
        format_name="JPEG",
        size_bytes=100,
        pixels=pixels,
    )
    return PreprocessedImage(
        source=source,
        display_pixels=pixels,
        inference_pixels=pixels,
        inference_width=48,
        inference_height=48,
    )


def test_semantic_repetition_removed_from_multi_person() -> None:
    caption = NaturalCaptionService(_QuietVision()).generate(  # type: ignore[arg-type]
        _image(), _multi_person_understanding(), context=_context()
    )
    lower = caption.lower()
    assert lower.count("farther back") <= 1
    assert "both people share" not in lower
    assert "still clearly visible in the space" not in lower
    # Chair/table named once as a group, not restated in a second spatial sentence.
    assert lower.count("dining table") <= 1
    assert lower.count("a chair") + lower.count("near a chair") <= 2


def test_repeated_objects_and_setting_removed_from_kitchen() -> None:
    caption = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), _indoor_understanding(), context=_indoor_context()
    )
    lower = caption.lower()
    assert lower.count("dining table") <= 1
    assert lower.count("kitchen") <= 2
    assert "remains visible deeper" not in lower
    assert "anchors the central" not in lower
    assert "vase" in lower and "refrigerator" in lower


def test_gate_drops_inventory_style_and_awkward_prose() -> None:
    service = NaturalCaptionService(_QuietVision())  # type: ignore[arg-type]
    understanding = _indoor_understanding()
    story = service._story_facts(understanding)
    bad = (
        "A person is preparing food in a kitchen. "
        "A dining table anchors the central space. "
        "A vase, a refrigerator, a television, a chair are visible nearby. "
        "A dining table remains visible deeper in a kitchen."
    )
    cleaned = service._final_naturalness_gate(bad, understanding, story)
    lower = cleaned.lower()
    assert "anchors the central" not in lower
    assert "remains visible deeper" not in lower
    assert lower.count("dining table") <= 1
    assert cleaned.endswith(".")


def test_unsupported_leading_softened_without_evidence() -> None:
    service = NaturalCaptionService(_QuietVision())  # type: ignore[arg-type]
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
            EvidenceFact("horse #1", "is", "horse", 0.9, "yolo"),
            EvidenceFact("scene", "setting", "field", 0.8, "environment"),
        ),
        ranked_subjects=("person #1", "horse #1"),
        environment_keys=("setting=field",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="person horse field",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    story = service._story_facts(understanding)
    cleaned = service._final_naturalness_gate(
        "A person is leading a horse on a field.",
        understanding,
        story,
    )
    lower = cleaned.lower()
    assert "leading" not in lower
    assert "horse" in lower
    assert "person" in lower


def test_supported_leading_preserved() -> None:
    service = NaturalCaptionService(_QuietVision())  # type: ignore[arg-type]
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
            EvidenceFact("horse #1", "is", "horse", 0.9, "yolo"),
            EvidenceFact("person #1", "leading", "horse", 0.8, "relationships"),
            EvidenceFact("scene", "setting", "field", 0.8, "environment"),
        ),
        ranked_subjects=("person #1", "horse #1"),
        environment_keys=("setting=field",),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="person leading horse",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    story = service._story_facts(understanding)
    cleaned = service._final_naturalness_gate(
        "A person is leading a horse on a field.",
        understanding,
        story,
    )
    assert "leading" in cleaned.lower()


def test_natural_grammar_avoids_robotic_depth_phrases() -> None:
    caption = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), _indoor_understanding(), context=_indoor_context()
    )
    lower = caption.lower()
    assert "we have" not in lower
    assert "is part of a kitchen" not in lower
    assert "appear near the main subject" not in lower
    assert "farther back in the frame" not in lower


def test_rich_evidence_still_preserved_for_kitchen() -> None:
    caption = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), _indoor_understanding(), context=_indoor_context()
    )
    lower = caption.lower()
    for token in ("person", "kitchen", "dining table", "refrigerator", "vase", "tv", "chair"):
        assert token in lower or (token == "tv" and "tvs" in lower)
    assert len(caption.split()) >= 25


def test_simple_landscape_stays_concise() -> None:
    understanding = SceneUnderstanding(
        facts=(
            EvidenceFact("scene", "setting", "mountain", 0.75, "environment"),
            EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.9, "environment"),
        ),
        ranked_subjects=("scene",),
        environment_keys=("setting=mountain", "indoor_outdoor=outdoor"),
        activity_keys=(),
        ocr_text=(),
        evidence_brief="mountain outdoor landscape",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )
    context = SceneContext(
        graph=SceneGraph(nodes=(), relations=()),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.5),
        environment=EnvironmentInfo(
            scene_type="mountain",
            setting="mountain",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="outdoor",
            social_context="unknown",
            crowd_level="sparse",
            scene_complexity="simple",
            evidence=(),
        ),
        object_count=0,
        dominant_objects=(),
        spatial_summary="",
    )
    caption = NaturalCaptionService(_QuietVision()).generate(  # type: ignore[arg-type]
        _image("landscape.jpg"), understanding, context=context
    )
    assert "mountain" in caption.lower()
    assert len(caption.split()) <= 40
    assert caption.lower().count("mountain") <= 2
