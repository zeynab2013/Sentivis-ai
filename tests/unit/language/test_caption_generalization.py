"""General-purpose caption quality across diverse scene types — not kitchen-tuned."""

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


def _understanding(
    *,
    facts: tuple[EvidenceFact, ...],
    subjects: tuple[str, ...],
    env: tuple[str, ...] = (),
    brief: str = "",
) -> SceneUnderstanding:
    return SceneUnderstanding(
        facts=facts,
        ranked_subjects=subjects,
        environment_keys=env,
        activity_keys=tuple(
            f.value for f in facts if f.predicate == "activity" and f.subject == "scene"
        ),
        ocr_text=tuple(f.value for f in facts if f.predicate == "visible_text"),
        evidence_brief=brief or " ".join(subjects),
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )


def _context(
    *,
    nodes: tuple[SceneNode, ...],
    relations: tuple[Relation, ...] = (),
    setting: str = "outdoor",
    indoor_outdoor: str = "outdoor",
) -> SceneContext:
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=relations),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.6),
        environment=EnvironmentInfo(
            scene_type=setting,
            setting=setting,
            time_of_day="day",
            weather="unknown",
            indoor_outdoor=indoor_outdoor,
            social_context="unknown",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=len(nodes),
        dominant_objects=tuple(n.label for n in nodes[:4]),
        spatial_summary="",
    )


SCENES = [
    pytest.param(
        "single_person",
        _understanding(
            facts=(
                EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
                EvidenceFact("person #1", "action", "standing", 0.7, "pose_estimator"),
                EvidenceFact("bench #1", "is", "bench", 0.8, "yolo"),
                EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.8, "environment"),
            ),
            subjects=("person #1", "bench #1"),
            env=("indoor_outdoor=outdoor",),
            brief="person bench outdoor",
        ),
        _context(
            nodes=(
                SceneNode(0, "p1", "person", 0.2, "middle-center"),
                SceneNode(1, "b1", "bench", 0.08, "middle-left"),
            ),
            setting="park",
        ),
        ("person",),
        ("dining table anchors", "kitchen furnishings", "we have", "on table we have"),
        id="single_person",
    ),
    pytest.param(
        "multiple_people",
        _understanding(
            facts=(
                EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
                EvidenceFact("person #2", "is", "person", 0.88, "yolo"),
                EvidenceFact("chair #1", "is", "chair", 0.75, "yolo"),
                EvidenceFact("dining table #1", "is", "dining table", 0.72, "yolo"),
                EvidenceFact("scene", "setting", "room", 0.7, "environment"),
                EvidenceFact("scene", "indoor_outdoor", "indoor", 0.8, "environment"),
            ),
            subjects=("person #1", "person #2", "chair #1", "dining table #1"),
            env=("indoor_outdoor=indoor", "setting=room"),
        ),
        _context(
            nodes=(
                SceneNode(0, "p1", "person", 0.2, "middle-center"),
                SceneNode(1, "p2", "person", 0.15, "back-center"),
                SceneNode(2, "c1", "chair", 0.1, "middle-left"),
                SceneNode(3, "t1", "dining table", 0.12, "middle-right"),
            ),
            setting="room",
            indoor_outdoor="indoor",
        ),
        ("person", "another person"),
        ("talking to a person", "looking at another person", "farther back in the frame"),
        id="multiple_people",
    ),
    pytest.param(
        "animal",
        _understanding(
            facts=(
                EvidenceFact("horse #1", "is", "horse", 0.92, "yolo"),
                EvidenceFact("horse #1", "dominant_color", "brown", 0.82, "attributes"),
                EvidenceFact("person #1", "is", "person", 0.85, "yolo"),
                EvidenceFact("person #1", "leading", "horse", 0.78, "relationships"),
                EvidenceFact("scene", "setting", "field", 0.8, "environment"),
            ),
            subjects=("person #1", "horse #1"),
            env=("setting=field",),
            brief="person leading horse field",
        ),
        _context(
            nodes=(
                SceneNode(0, "p1", "person", 0.15, "middle-center"),
                SceneNode(1, "h1", "horse", 0.25, "middle-center"),
            ),
            relations=(Relation(0, 1, "leading", 0.78),),
            setting="field",
        ),
        ("horse", "person"),
        ("talking to a person",),
        id="animal",
    ),
    pytest.param(
        "vehicle",
        _understanding(
            facts=(
                EvidenceFact("car #1", "is", "car", 0.9, "yolo"),
                EvidenceFact("person #1", "is", "person", 0.82, "yolo"),
                EvidenceFact("scene", "setting", "street", 0.85, "environment"),
                EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.85, "environment"),
            ),
            subjects=("car #1", "person #1"),
            env=("setting=street", "indoor_outdoor=outdoor"),
        ),
        _context(
            nodes=(
                SceneNode(0, "c1", "car", 0.3, "middle-center"),
                SceneNode(1, "p1", "person", 0.05, "middle-left"),
            ),
            setting="street",
        ),
        ("car",),
        ("professional cyclist", "competing in a race"),
        id="vehicle",
    ),
    pytest.param(
        "landscape",
        _understanding(
            facts=(
                EvidenceFact("scene", "setting", "mountain", 0.75, "environment"),
                EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.9, "environment"),
            ),
            subjects=("scene",),
            env=("setting=mountain", "indoor_outdoor=outdoor"),
            brief="mountain outdoor landscape",
        ),
        _context(nodes=(), setting="mountain"),
        ("mountain",),
        ("a person talking", "we have"),
        id="landscape",
    ),
    pytest.param(
        "sports",
        _understanding(
            facts=(
                EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
                EvidenceFact("tennis racket #1", "is", "tennis racket", 0.85, "yolo"),
                EvidenceFact("person #1", "holding", "tennis racket", 0.8, "relationships"),
                EvidenceFact("person #1", "action", "playing tennis", 0.75, "pose_estimator"),
                EvidenceFact("scene", "setting", "tennis court", 0.8, "environment"),
            ),
            subjects=("person #1", "tennis racket #1"),
            env=("setting=tennis court",),
        ),
        _context(
            nodes=(
                SceneNode(0, "p1", "person", 0.2, "middle-center"),
                SceneNode(1, "r1", "tennis racket", 0.04, "middle-right"),
            ),
            relations=(Relation(0, 1, "holding", 0.8),),
            setting="tennis court",
        ),
        ("tennis", "person"),
        ("talking to a person",),
        id="sports",
    ),
    pytest.param(
        "food_object",
        _understanding(
            facts=(
                EvidenceFact("bowl #1", "is", "bowl", 0.88, "yolo"),
                EvidenceFact("apple #1", "is", "apple", 0.85, "yolo"),
                EvidenceFact("banana #1", "is", "banana", 0.84, "yolo"),
                EvidenceFact("scene", "setting", "table", 0.7, "environment"),
            ),
            subjects=("bowl #1", "apple #1", "banana #1"),
            env=("setting=table",),
        ),
        _context(
            nodes=(
                SceneNode(0, "b1", "bowl", 0.08, "middle-center"),
                SceneNode(1, "a1", "apple", 0.03, "middle-center"),
                SceneNode(2, "n1", "banana", 0.03, "middle-left"),
            ),
            setting="table",
            indoor_outdoor="indoor",
        ),
        ("bowl",),
        ("we have", "on table we have"),
        id="food_object",
    ),
    pytest.param(
        "text_ocr",
        _understanding(
            facts=(
                EvidenceFact("stop sign #1", "is", "stop sign", 0.95, "yolo"),
                EvidenceFact("scene", "visible_text", "STOP", 0.9, "ocr"),
                EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.85, "environment"),
            ),
            subjects=("stop sign #1",),
            env=("indoor_outdoor=outdoor",),
        ),
        _context(
            nodes=(SceneNode(0, "s1", "stop sign", 0.06, "middle-right"),),
            setting="street",
        ),
        ("stop",),
        ("we have",),
        id="text_ocr",
    ),
]


@pytest.mark.parametrize(
    "name,understanding,context,must_include,must_exclude",
    SCENES,
)
def test_diverse_scenes_produce_grounded_natural_captions(
    name: str,
    understanding: SceneUnderstanding,
    context: SceneContext,
    must_include: tuple[str, ...],
    must_exclude: tuple[str, ...],
) -> None:
    caption = NaturalCaptionService(_QuietVision()).generate(  # type: ignore[arg-type]
        _image(f"{name}.jpg"), understanding, context=context
    )
    lower = caption.lower()
    for token in must_include:
        assert token in lower, f"{name}: expected {token!r} in {caption!r}"
    for token in must_exclude:
        assert token not in lower, f"{name}: forbidden {token!r} in {caption!r}"
    assert "we have" not in lower
    assert "on table we have" not in lower
    assert lower.count("are visible nearby") <= 1
    assert caption.endswith(".")


def test_kitchen_regression_still_rich_and_grounded() -> None:
    """Kitchen path must keep working — but via general evidence synthesis."""
    caption = NaturalCaptionService(_QuietVision()).generate(  # type: ignore[arg-type]
        _image("kitchen.jpg"),
        _indoor_understanding(),
        context=_indoor_context(),
    )
    lower = caption.lower()
    assert any(tok in lower for tok in ("person", "prepar", "kitchen", "food"))
    assert "dining table" in lower or "table" in lower
    assert "refrigerator" in lower
    assert "talking to a person" not in lower
    assert len(caption.split()) >= 25
    assert "dining table anchors" not in lower
    assert "kitchen furnishings" not in lower
    assert lower.count("dining table") <= 1
    assert lower.count("kitchen") <= 2
    assert "remains visible deeper" not in lower
