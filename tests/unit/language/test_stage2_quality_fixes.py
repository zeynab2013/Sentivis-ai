"""Grammar / color / relation consistency regressions for stage-2 quality pass."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

from core.contracts.analysis import Relation, SceneGraph, SceneNode
from core.contracts.image import PreprocessedImage, ValidatedImage
from core.contracts.language import RawCaption
from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from language.evaluation.caption_quality_evaluator import CaptionQualityEvaluator
from language.refinement.caption_refiner import clear_ui_language_cache
from language.refinement.caption_sanity import sanitize_caption
from language.semantic.natural_caption_service import NaturalCaptionService
from analysis.relationships.relation_metrics import count_meaningful_relations


@pytest.fixture(autouse=True)
def _force_english() -> None:
    os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
    clear_ui_language_cache()
    yield
    clear_ui_language_cache()


class _QuietVision:
    def narrate(self, image: object, understanding: SceneUnderstanding) -> RawCaption:
        return RawCaption(text="", source="stub", confidence=0.0)


def _image() -> PreprocessedImage:
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    source = ValidatedImage(
        path=Path("farm.jpg"),
        width=32,
        height=32,
        format_name="JPEG",
        size_bytes=10,
        pixels=pixels,
    )
    return PreprocessedImage(
        source=source,
        display_pixels=pixels,
        inference_pixels=pixels,
        inference_width=32,
        inference_height=32,
    )


def _farm_understanding() -> SceneUnderstanding:
    facts = (
        EvidenceFact("person #1", "is", "person", 0.9, "yolo"),
        EvidenceFact("person #2", "is", "person", 0.85, "yolo"),
        EvidenceFact("horse #1", "is", "horse", 0.9, "yolo"),
        EvidenceFact("horse #2", "is", "horse", 0.8, "yolo"),
        EvidenceFact("person #1", "leading", "horse", 0.82, "relationships"),
        EvidenceFact("person #1", "shirt_color", "light blue", 0.8, "attributes"),
        EvidenceFact("person #1", "pants_color", "olive", 0.78, "attributes"),
        EvidenceFact("horse #1", "dominant_color", "tan", 0.8, "attributes"),
        EvidenceFact("scene", "setting", "farm pasture", 0.75, "environment"),
        EvidenceFact("scene", "indoor_outdoor", "outdoor", 0.85, "environment"),
        EvidenceFact("scene", "activity", "leading a horse", 0.7, "activities"),
    )
    return SceneUnderstanding(
        facts=facts,
        ranked_subjects=("person #1", "person #2", "horse #1", "horse #2"),
        environment_keys=("indoor_outdoor=outdoor", "setting=farm pasture"),
        activity_keys=("leading a horse",),
        ocr_text=(),
        evidence_brief="person leading horse; light blue shirt; olive pants; tan horse; farm pasture",
        overall_confidence=0.8,
        discarded_count=0,
        contradictions_resolved=0,
    )


def test_sanitize_fixes_with_one_is() -> None:
    bad = (
        "Two people are visible in a farm pasture, with one is leading a horse "
        "while others remain farther back."
    )
    fixed = sanitize_caption(bad)
    assert "with one is" not in fixed.lower()
    assert "one of them is leading" in fixed.lower()


def test_grammar_score_penalizes_with_one_is() -> None:
    evaluator = CaptionQualityEvaluator()
    bad = "Two people are visible, with one is leading a horse."
    good = "Two people are visible. One of them is leading a horse."
    assert evaluator._grammar_score(bad) < 0.8
    assert evaluator._grammar_score(good) >= 0.85


def test_farm_caption_grammar_and_colors() -> None:
    service = NaturalCaptionService(_QuietVision())  # type: ignore[arg-type]
    paragraph = service.generate(_image(), _farm_understanding())
    lower = paragraph.lower()
    assert "with one is" not in lower
    assert "a person talking to a person" not in lower
    assert "leading" in lower or "horse" in lower
    assert any(
        re.search(rf"\b{re.escape(tok)}\b", lower)
        for tok in ("blue", "olive", "tan", "light blue")
    ), paragraph
    assert len(paragraph.split()) >= 18
    assert "farther back nearby" not in lower


def test_meaningful_relation_count_matches_filter() -> None:
    nodes = (
        SceneNode(0, "o0", "person", 0.2, "middle"),
        SceneNode(1, "o1", "horse", 0.2, "middle"),
        SceneNode(2, "o2", "person", 0.1, "back"),
    )
    relations = (
        Relation(0, 1, "leading", 0.9),
        Relation(0, 1, "near", 0.9),
        Relation(0, 2, "left_of", 0.8),
        Relation(0, 1, "holding", 0.5),  # below confidence gate
    )
    graph = SceneGraph(nodes=nodes, relations=relations)
    # Interactions at metrics floor + QA-safe spatial layout; weak holding excluded.
    assert count_meaningful_relations(graph) == 3
    from analysis.relationships.relation_metrics import meaningful_relations

    kinds = {r.relation_type for r in meaningful_relations(graph)}
    assert "holding" not in kinds
