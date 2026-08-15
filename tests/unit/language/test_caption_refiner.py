"""Unit tests for caption refinement."""

import time

from analysis.context.context_builder import ContextBuilder
from analysis.scene_graph.scene_graph_builder import SceneGraphBuilder
from core.config.loader import load_analysis_config
from core.contracts.analysis import (
    ActivityHints,
    SceneContext,
)
from core.contracts.detection import DetectionResult
from core.contracts.language import RawCaption, RefinedCaption
from language.refinement.caption_refiner import CaptionRefiner


def _empty_context() -> SceneContext:
    now = time.time()
    detections = DetectionResult(detections=(), image_width=100, image_height=100, inference_timestamp=now)
    graph = SceneGraphBuilder(load_analysis_config()).build(detections, ())
    return ContextBuilder(load_analysis_config()).build(
        graph,
        __import__("core.contracts.analysis", fromlist=["AttributeSet"]).AttributeSet(attributes=()),
        ActivityHints(activities=(), confidence=0.4),
    )


def test_refiner_polishes_caption() -> None:
    primary = RawCaption(text="a person walking in a park", source="gemma", confidence=0.8)
    refined = CaptionRefiner().refine(primary, None, _empty_context())
    assert isinstance(refined, RefinedCaption)
    assert refined.text.endswith(".")


def test_refiner_uses_fallback_when_primary_empty() -> None:
    context = _empty_context()
    primary = RawCaption(text="", source="gemma", confidence=0.0)
    fallback = RawCaption(text="A park scene", source="blip", confidence=0.7)
    refined = CaptionRefiner().refine(primary, fallback, context)
    assert "park" in refined.text.lower() or "uncertain" in refined.text.lower()
