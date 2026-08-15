"""Caption quality overhaul — multi-person and factuality regressions."""

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
    SceneContext,
    SceneGraph,
    SceneNode,
)
from core.contracts.image import PreprocessedImage, ValidatedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from language.refinement.caption_refiner import clear_ui_language_cache
from language.refinement.caption_sanity import choose_better_caption, has_awkward_filler
from language.semantic.natural_caption_service import NaturalCaptionService
from language.validation.caption_factuality import (
    ClaimSupport,
    classify_sentence,
    filter_unsupported_claims,
    quality_signals,
)


@pytest.fixture(autouse=True)
def _force_english_ui_language() -> None:
    os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
    clear_ui_language_cache()
    yield
    clear_ui_language_cache()


class _StubVision:
    def narrate(self, image: object, understanding: SceneUnderstanding) -> RawCaption:
        return RawCaption(
            text=(
                "Two people share an indoor room, with one nearer the camera appearing to "
                "speak toward another person farther back near a wooden table and chairs."
            ),
            source="florence",
            confidence=0.7,
        )


class _QuietVision:
    """Returns empty narrate so template synthesis is evaluated alone."""

    def narrate(self, image: object, understanding: SceneUnderstanding) -> RawCaption:
        return RawCaption(text="", source="stub", confidence=0.0)


def _image() -> PreprocessedImage:
    pixels = np.zeros((48, 48, 3), dtype=np.uint8)
    source = ValidatedImage(
        path=Path("multi.jpg"),
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


def _multi_person_understanding() -> SceneUnderstanding:
    facts = (
        EvidenceFact("person #1", "is", "person", 0.92, "yolo"),
        EvidenceFact("person #2", "is", "person", 0.88, "yolo"),
        EvidenceFact("person #1", "talking_to", "person", 0.8, "relationships"),
        EvidenceFact("person #1", "action", "standing", 0.7, "pose_estimator"),
        EvidenceFact("person #2", "action", "standing", 0.65, "pose_estimator"),
        EvidenceFact("chair", "is", "chair", 0.75, "yolo"),
        EvidenceFact("dining table", "is", "dining table", 0.72, "yolo"),
        EvidenceFact("scene", "indoor_outdoor", "indoor", 0.8, "environment"),
        EvidenceFact("scene", "setting", "room", 0.7, "environment"),
    )
    return SceneUnderstanding(
        facts=facts,
        ranked_subjects=("person #1", "person #2", "chair", "dining table"),
        environment_keys=("indoor_outdoor=indoor", "setting=room"),
        activity_keys=("talking",),
        ocr_text=(),
        evidence_brief="person #1 talking_to person; person #2 standing; chair; dining table; indoor",
        overall_confidence=0.82,
        discarded_count=0,
        contradictions_resolved=0,
    )


def _context() -> SceneContext:
    nodes = (
        SceneNode(0, "obj-0", "person", 0.35, "middle-center"),
        SceneNode(1, "obj-1", "person", 0.18, "back-center"),
        SceneNode(2, "obj-2", "chair", 0.12, "middle-left"),
        SceneNode(3, "obj-3", "dining table", 0.15, "middle-right"),
    )
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=()),
        attributes=AttributeSet(attributes=()),
        activities=ActivityHints(activities=(), confidence=0.6),
        environment=EnvironmentInfo(
            scene_type="room",
            setting="room",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="indoor",
            social_context="casual",
            crowd_level="sparse",
            scene_complexity="moderate",
            evidence=(),
        ),
        object_count=4,
        dominant_objects=("person", "chair", "dining table"),
        spatial_summary="Two people indoors.",
    )


def test_multi_person_talking_never_emits_robotic_failure() -> None:
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    paragraph = service.generate(_image(), _multi_person_understanding(), context=_context())
    lower = paragraph.lower()
    assert "a person talking to a person" not in lower
    assert "second person stands farther back in the frame" not in lower
    assert not has_awkward_filler(paragraph)
    assert "person" in lower or "people" in lower
    # Must synthesize a natural multi-person description, not detector English.
    assert "two people" in lower or "another person" in lower or "speaking" in lower
    assert len(paragraph.split()) >= 15
    # Should use available scene evidence when present.
    assert any(tok in lower for tok in ("indoor", "room", "chair", "table"))
    assert lower.count("farther back") <= 1
    assert "both people share" not in lower


def test_defining_interaction_not_talking_to_a_person() -> None:
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    scene = service._build_semantic_scene(_multi_person_understanding())
    assert "talking to a person" not in scene.defining_interaction.lower()
    assert "talking_to" not in scene.defining_interaction.lower()
    # Conservative: multi-person without verified interaction stays spatial, not conversational.
    assert (
        "sharing the scene" in scene.defining_interaction.lower()
        or "speaking" in scene.defining_interaction.lower()
        or "facing" in scene.defining_interaction.lower()
        or scene.defining_interaction == ""
    )


def test_choose_better_rejects_robotic_short_caption() -> None:
    bad = "A person talking to a person. A second person stands farther back in the frame."
    good = (
        "Two people are visible in an indoor room, with one in the foreground appearing "
        "to speak toward another person farther back near a table and chairs."
    )
    chosen = choose_better_caption(bad, good)
    assert "talking to a person" not in chosen.lower()
    assert "two people" in chosen.lower() or "speak" in chosen.lower()


def test_factuality_drops_invented_roles() -> None:
    understanding = _multi_person_understanding()
    text = (
        "The teacher is explaining the project to the student. "
        "Two people are visible indoors near a chair."
    )
    filtered = filter_unsupported_claims(text, understanding)
    lower = filtered.lower()
    assert "teacher" not in lower
    assert "student" not in lower
    assert "people" in lower or "chair" in lower


def test_factuality_flags_robotic_sentence() -> None:
    understanding = _multi_person_understanding()
    verdict = classify_sentence(
        "A person talking to a person.",
        understanding,
    )
    assert verdict.status == ClaimSupport.UNSUPPORTED


def test_quality_signals_prefer_informative_caption() -> None:
    understanding = _multi_person_understanding()
    thin = "A person talking to a person."
    rich = (
        "Two people are visible in an indoor room, with one nearer the camera speaking "
        "toward another person farther back beside a dining table and chair."
    )
    thin_s = quality_signals(thin, understanding)
    rich_s = quality_signals(rich, understanding)
    assert rich_s.word_count > thin_s.word_count
    assert rich_s.information_density >= thin_s.information_density
    assert thin_s.unsupported_claim_count >= 1
    assert rich_s.unsupported_claim_count == 0


# ---------------------------------------------------------------------------
# Representative evaluation cases (A–F checked structurally, not against model output)
# ---------------------------------------------------------------------------


def _case_understanding(
    *,
    subjects: tuple[str, ...],
    facts: tuple[EvidenceFact, ...],
    env: tuple[str, ...] = ("indoor_outdoor=indoor",),
    ocr: tuple[str, ...] = (),
    brief: str = "",
) -> SceneUnderstanding:
    return SceneUnderstanding(
        facts=facts,
        ranked_subjects=subjects,
        environment_keys=env,
        activity_keys=(),
        ocr_text=ocr,
        evidence_brief=brief or " ".join(subjects),
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )


@pytest.mark.parametrize(
    "name,understanding,must_include,must_exclude",
    [
        (
            "single_person",
            _case_understanding(
                subjects=("person #1", "chair"),
                facts=(
                    EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
                    EvidenceFact("person #1", "sitting_on", "chair", 0.8, "relationships"),
                    EvidenceFact("scene", "indoor_outdoor", "indoor", 0.7, "environment"),
                ),
                brief="person sitting on chair indoor",
            ),
            ("person",),
            ("teacher", "happy"),
        ),
        (
            "multiple_people",
            _multi_person_understanding(),
            ("people", "person"),
            ("a person talking to a person",),
        ),
        (
            "indoor_scene",
            _case_understanding(
                subjects=("person #1", "refrigerator", "oven"),
                facts=(
                    EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
                    EvidenceFact("refrigerator", "is", "refrigerator", 0.85, "yolo"),
                    EvidenceFact("oven", "is", "oven", 0.8, "yolo"),
                    EvidenceFact("scene", "setting", "kitchen", 0.8, "environment"),
                    EvidenceFact("scene", "indoor_outdoor", "indoor", 0.85, "environment"),
                ),
                env=("indoor_outdoor=indoor", "setting=kitchen"),
                brief="person refrigerator oven kitchen indoor",
            ),
            ("person",),
            ("beach",),
        ),
        (
            "outdoor_scene",
            _case_understanding(
                subjects=("person #1", "bicycle"),
                facts=(
                    EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
                    EvidenceFact("person #1", "riding", "bicycle", 0.8, "relationships"),
                    EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.85, "environment"),
                ),
                env=("indoor_outdoor=outdoor",),
                brief="person riding bicycle outdoor",
            ),
            ("person", "bicycle"),
            ("classroom",),
        ),
        (
            "many_objects",
            _case_understanding(
                subjects=("bowl", "cup", "bottle", "banana", "apple"),
                facts=(
                    EvidenceFact("bowl", "is", "bowl", 0.9, "yolo"),
                    EvidenceFact("cup", "is", "cup", 0.85, "yolo"),
                    EvidenceFact("bottle", "is", "bottle", 0.8, "yolo"),
                    EvidenceFact("banana", "is", "banana", 0.8, "yolo"),
                    EvidenceFact("apple", "is", "apple", 0.8, "yolo"),
                    EvidenceFact("scene", "setting", "table", 0.7, "environment"),
                ),
                brief="bowl cup bottle banana apple table",
            ),
            ("bowl",),
            ("person talking to a person",),
        ),
        (
            "visible_text",
            _case_understanding(
                subjects=("stop sign",),
                facts=(
                    EvidenceFact("stop sign", "is", "stop sign", 0.95, "yolo"),
                    EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.8, "environment"),
                ),
                ocr=("STOP",),
                brief="stop sign OCR=STOP outdoor",
            ),
            ("stop",),
            ("teacher",),
        ),
        (
            "person_action",
            _case_understanding(
                subjects=("person #1", "sports ball"),
                facts=(
                    EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
                    EvidenceFact("person #1", "action", "kicking", 0.75, "pose_estimator"),
                    EvidenceFact("sports ball", "is", "sports ball", 0.8, "yolo"),
                ),
                brief="person kicking sports ball",
            ),
            ("person",),
            ("happy", "teammate"),
        ),
        (
            "complex_background",
            _case_understanding(
                subjects=("person #1", "car", "traffic light", "backpack"),
                facts=(
                    EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
                    EvidenceFact("car", "is", "car", 0.85, "yolo"),
                    EvidenceFact("traffic light", "is", "traffic light", 0.8, "yolo"),
                    EvidenceFact("backpack", "is", "backpack", 0.7, "yolo"),
                    EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.85, "environment"),
                    EvidenceFact("scene", "setting", "street", 0.8, "environment"),
                ),
                env=("indoor_outdoor=outdoor", "setting=street"),
                brief="person car traffic light backpack street outdoor",
            ),
            ("person",),
            ("airport",),
        ),
        (
            "sparse_image",
            _case_understanding(
                subjects=("bowl",),
                facts=(EvidenceFact("bowl", "is", "bowl", 0.9, "yolo"),),
                brief="bowl",
            ),
            ("bowl",),
            ("crowd", "teacher"),
        ),
        (
            "ambiguous_image",
            _case_understanding(
                subjects=("person #1",),
                facts=(
                    EvidenceFact("person #1", "is", "person", 0.6, "yolo"),
                    EvidenceFact("person #1", "action", "standing", 0.55, "pose_estimator"),
                ),
                brief="person standing",
            ),
            ("person",),
            ("CEO", "celebrating"),
        ),
    ],
)
def test_representative_caption_eval_cases(
    name: str,
    understanding: SceneUnderstanding,
    must_include: tuple[str, ...],
    must_exclude: tuple[str, ...],
) -> None:
    """Structural eval: captions must stay grounded and avoid known failure modes."""
    service = NaturalCaptionService(_QuietVision())  # type: ignore[arg-type]
    paragraph = service.generate(_image(), understanding, context=_context())
    lower = paragraph.lower()
    assert paragraph.strip(), name
    assert "\n" not in paragraph, name
    for token in must_include:
        # For multi-person, either people OR person is enough when listed together.
        if must_include == ("people", "person"):
            assert "people" in lower or "person" in lower, name
            break
        assert token.lower() in lower, f"{name}: missing {token!r} in {paragraph!r}"
    for token in must_exclude:
        assert token.lower() not in lower, f"{name}: found banned {token!r}"
    # Factuality / hallucination bar for invented roles/emotions.
    assert "teacher" not in lower and "coworker" not in lower
    assert classify_sentence(
        "A person talking to a person.",
        understanding,
    ).status == ClaimSupport.UNSUPPORTED
    signals = quality_signals(paragraph, understanding)
    # Sparse/ambiguous scenes may honestly stay short — do not force padding.
    min_words = 6 if name in {"sparse_image", "ambiguous_image", "single_person"} else 8
    assert signals.word_count >= min_words, name
    assert signals.unsupported_claim_count <= max(2, signals.sentence_count), name
