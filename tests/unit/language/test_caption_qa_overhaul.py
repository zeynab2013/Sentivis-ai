"""Caption factuality fixtures + grounded Q&A / suggested-question regressions."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

from core.contracts.analysis import (
    ActivityEvidence,
    ActivityHints,
    Attribute,
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
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever
from language.refinement.caption_refiner import clear_ui_language_cache
from language.semantic.natural_caption_service import NaturalCaptionService
from pathlib import Path


@pytest.fixture(autouse=True)
def _force_english() -> None:
    os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
    clear_ui_language_cache()
    yield
    clear_ui_language_cache()


class _StubVision:
    def narrate(self, image: object, understanding: SceneUnderstanding) -> RawCaption:
        return RawCaption(text="stub", source="stub", confidence=0.4)


def _image() -> PreprocessedImage:
    pixels = np.zeros((48, 48, 3), dtype=np.uint8)
    source = ValidatedImage(
        path=Path("fixture.jpg"),
        width=48,
        height=48,
        format_name="JPEG",
        size_bytes=200,
        pixels=pixels,
    )
    return PreprocessedImage(
        source=source,
        display_pixels=pixels,
        inference_pixels=pixels,
        inference_width=48,
        inference_height=48,
        original_display_pixels=pixels,
    )


def _table_scene_understanding() -> SceneUnderstanding:
    return SceneUnderstanding(
        facts=(
            EvidenceFact("person #1", "is", "person", 0.85, "yolo"),
            EvidenceFact("person #1", "action", "standing", 0.70, "pose_estimator"),
            EvidenceFact("chair #1", "is", "chair", 0.80, "yolo"),
            EvidenceFact("table #1", "is", "dining table", 0.75, "yolo"),
            EvidenceFact("person #1", "near", "dining table", 0.70, "relationships"),
            EvidenceFact("scene", "indoor_outdoor", "indoor", 0.85, "environment"),
            EvidenceFact("scene", "setting", "room", 0.70, "environment"),
        ),
        ranked_subjects=("person #1", "dining table #1", "chair #1"),
        environment_keys=("indoor_outdoor=indoor", "setting=room"),
        activity_keys=("standing",),
        ocr_text=(),
        evidence_brief="person standing near dining table and chair",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )


def _table_context() -> SceneContext:
    nodes = (
        SceneNode(0, "obj-0", "person", 0.08, "middle-center"),
        SceneNode(1, "obj-1", "dining table", 0.10, "middle-center"),
        SceneNode(2, "obj-2", "chair", 0.04, "middle-left"),
    )
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=(Relation(0, 1, "near", 0.7),)),
        attributes=AttributeSet(
            attributes=(
                Attribute(0, "confidence", "0.85"),
                Attribute(1, "confidence", "0.75"),
                Attribute(2, "confidence", "0.80"),
            )
        ),
        activities=ActivityHints(
            activities=(ActivityEvidence("standing", 0.7, (0,), (), "pose"),),
            confidence=0.7,
        ),
        environment=EnvironmentInfo(
            scene_type="room",
            setting="room",
            time_of_day="day",
            weather="unknown",
            indoor_outdoor="indoor",
            social_context="home",
            crowd_level="sparse",
            scene_complexity="simple",
            evidence=("indoor room",),
        ),
        object_count=3,
        dominant_objects=("person", "dining table", "chair"),
        spatial_summary="Person near dining table.",
    )


def test_caption_grounded_no_unsupported_occupation() -> None:
    caption = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), _table_scene_understanding(), context=_table_context()
    )
    lower = caption.lower()
    assert "person" in lower or "standing" in lower
    for banned in ("teacher", "businessman", "25 years", "laptop", "working on"):
        assert banned not in lower


def test_caption_includes_chair_when_evidenced() -> None:
    """Regression: object sentence must not blank out when primary leftover is a chair."""
    caption = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), _table_scene_understanding(), context=_table_context()
    )
    lower = caption.lower()
    # At least one of the secondary fixtures should appear when evidenced.
    assert "chair" in lower or "table" in lower or "dining" in lower


def test_qa_near_objects_direct() -> None:
    packet = build_evidence_packet(
        _table_context(),
        canonical_caption_en="A person is standing in a room.",
        evidence_brief="person near dining table; chair",
    )
    answer = VisualEvidenceRetriever().try_direct_answer(
        packet, "What objects are positioned near the person?"
    )
    assert answer
    assert "caption" not in answer.lower()
    assert any(tok in answer.lower() for tok in ("table", "chair", "near"))


def test_qa_unsupported_name() -> None:
    packet = build_evidence_packet(
        _table_context(),
        canonical_caption_en="A person is standing in a room.",
    )
    session = VisionAssistantSession(image_key="t1", evidence=packet)
    answer = VisionAssistant(client=None).answer(session, "What is the person's name?", language="en")  # type: ignore[arg-type]
    lower = answer.lower()
    assert any(
        tok in lower
        for tok in ("cannot", "can't", "not", "unable", "determine", "evidence", "visible")
    )
    assert "john" not in lower
    assert session.assistant_vlm_calls == 0


def test_qa_followup_which_one_resolves() -> None:
    packet = build_evidence_packet(
        _table_context(),
        canonical_caption_en="A person stands near a dining table and chair.",
    )
    session = VisionAssistantSession(image_key="t2", evidence=packet)
    assistant = VisionAssistant(client=None)  # type: ignore[arg-type]
    # Seed conversation with an inventory answer.
    from language.assistant.vision_assistant import AssistantTurn

    session.turns.append(AssistantTurn(role="user", text="What objects are near the person?"))
    session.turns.append(
        AssistantTurn(role="assistant", text="A dining table and a chair are near the person.")
    )
    resolved = assistant._resolve_followup("Which one is closest to the person?", session.turns)  # noqa: SLF001
    assert "dining table" in resolved.lower() or "chair" in resolved.lower()


def test_suggested_questions_diverse_and_answerable() -> None:
    packet = build_evidence_packet(
        _table_context(),
        canonical_caption_en="A person is standing in a room.",
        evidence_brief="person; dining table; chair; standing",
    )
    questions = generate_suggested_questions(packet, language="en", limit=3)
    assert 1 <= len(questions) <= 3
    joined = " ".join(questions).lower()
    assert "what is in the image" not in joined
    assert "can you describe" not in joined
    # No exact duplicates
    assert len(set(q.lower() for q in questions)) == len(questions)


def test_suggested_questions_skip_when_no_people_for_people_count() -> None:
    nodes = (SceneNode(0, "obj-0", "car", 0.2, "middle-center"),)
    ctx = SceneContext(
        graph=SceneGraph(nodes=nodes, relations=()),
        attributes=AttributeSet(attributes=(Attribute(0, "confidence", "0.90"),)),
        activities=ActivityHints(activities=(), confidence=0.0),
        environment=EnvironmentInfo(
            scene_type="street",
            setting="street",
            time_of_day="day",
            weather="clear",
            indoor_outdoor="outdoor",
            social_context="public",
            crowd_level="sparse",
            scene_complexity="simple",
            evidence=("outdoor street",),
        ),
        object_count=1,
        dominant_objects=("car",),
        spatial_summary="Car on street.",
    )
    packet = build_evidence_packet(ctx, canonical_caption_en="A car is on the street.")
    questions = generate_suggested_questions(packet, language="en", limit=3)
    for q in questions:
        assert "how many people" not in q.lower()
        assert "person doing" not in q.lower()


def test_image_change_resets_session_key() -> None:
    """Different paths/captions produce different image keys (context reset)."""
    from streamlit_app.components import vision_assistant as va
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest, PipelineResult
    from core.contracts.language import CaptionQualityReport, RefinedCaption
    from core.contracts.metrics import PipelineMetrics

    def _result(path: str, caption: str) -> PipelineResult:
        return PipelineResult(
            request=PipelineRequest(image_path=Path(path), options=AnalysisOptions()),
            scene_context=_table_context(),
            caption=RefinedCaption(
                text=caption,
                sources=("test",),
                narrative_full=caption,
                narrative_short=caption,
                executive_summary=caption,
                canonical_caption_en=caption,
            ),
            quality_report=CaptionQualityReport(
                grammar_score=0.8,
                fluency_score=0.8,
                evidence_consistency=0.8,
                object_coverage=0.8,
                relationship_coverage=0.8,
                activity_coverage=0.8,
                context_coverage=0.8,
                hallucination_risk=0.0,
                overall_quality=0.8,
                notes=(),
            ),
            metrics=PipelineMetrics(
                total_duration_ms=1.0,
                peak_ram_mb=0.0,
                peak_vram_mb=0.0,
                stage_metrics=(),
                model_timings=(),
                objects_detected=3,
                relationships_inferred=1,
                activities_inferred=1,
                scene_graph_nodes=3,
                scene_graph_edges=1,
                caption_quality_score=0.8,
                recovery_events=0,
                fallback_events=0,
                competition_mode=False,
                qa_passed=True,
            ),
            qa_passed=True,
            stages_completed=(),
            warnings=(),
        )

    a = va._image_key(_result("a.jpg", "caption one"))
    b = va._image_key(_result("b.jpg", "caption two"))
    assert a != b
