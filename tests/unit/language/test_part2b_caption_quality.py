"""PART 2-B — Competition-level caption quality regressions.

Verifies factuality + useful detail without hardcoding full caption strings
(unless testing a specific safety rule).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

from analysis.evidence.verified_evidence_builder import (
    build_verified_scene_evidence,
    language_understanding_from_verified,
)
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
from core.contracts.reasoning import SceneUnderstanding
from language.refinement.caption_arbitration import (
    arbitrate_captions,
    score_caption_candidate,
)
from language.refinement.caption_refiner import clear_ui_language_cache
from language.refinement.caption_sanity import sanitize_caption
from language.semantic.natural_caption_service import NaturalCaptionService
from language.validation.caption_factuality import (
    ClaimSupport,
    classify_sentence_against_verified,
    filter_unsupported_claims_verified,
)


@pytest.fixture(autouse=True)
def _force_english_ui_language() -> None:
    os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
    clear_ui_language_cache()
    yield
    clear_ui_language_cache()


class _StubVision:
    """Weak VLM so NaturalCaptionService exercises the synthesis path."""

    def narrate(self, image: object, understanding: SceneUnderstanding) -> RawCaption:
        return RawCaption(text="a scene.", source="stub", confidence=0.3)

    def describe(self, image: object) -> RawCaption:
        return RawCaption(text="", source="stub", confidence=0.0)


def _env(
    *,
    scene_type: str = "indoor",
    setting: str = "room",
    indoor_outdoor: str = "indoor",
    crowd: str = "few",
    complexity: str = "medium",
) -> EnvironmentInfo:
    return EnvironmentInfo(
        scene_type=scene_type,
        setting=setting,
        time_of_day="day",
        weather="unknown",
        indoor_outdoor=indoor_outdoor,
        social_context="",
        crowd_level=crowd,
        scene_complexity=complexity,
        evidence=(),
    )


def _ctx(
    nodes: tuple[SceneNode, ...],
    relations: tuple[Relation, ...] = (),
    attrs: tuple[Attribute, ...] = (),
    *,
    activities: tuple[ActivityEvidence, ...] = (),
    env: EnvironmentInfo | None = None,
) -> SceneContext:
    return SceneContext(
        graph=SceneGraph(nodes=nodes, relations=relations),
        attributes=AttributeSet(attributes=attrs),
        activities=ActivityHints(activities=activities, confidence=0.7 if activities else 0.0),
        environment=env or _env(),
        object_count=len(nodes),
        dominant_objects=tuple(n.label for n in nodes),
        spatial_summary="",
    )


def _image() -> PreprocessedImage:
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    source = ValidatedImage(
        path=Path("x.jpg"),
        width=32,
        height=32,
        format_name="JPEG",
        size_bytes=100,
        pixels=pixels,
    )
    return PreprocessedImage(
        source=source,
        display_pixels=pixels,
        inference_pixels=pixels,
        inference_width=32,
        inference_height=32,
    )


def _synth_caption(ctx: SceneContext) -> str:
    verified = build_verified_scene_evidence(ctx)
    understanding = language_understanding_from_verified(verified)
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    return service.generate(_image(), understanding, context=ctx).strip()


# --- 1. Single person ---
def test_single_person_no_invented_companion() -> None:
    ctx = _ctx(
        (SceneNode(0, "person_1", "person", 0.35, "middle-center"),),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "shirt_color", "navy"),
            Attribute(0, "clothing_type", "jacket"),
        ),
        env=_env(setting="street", scene_type="outdoor", indoor_outdoor="outdoor"),
    )
    caption = _synth_caption(ctx)
    lower = caption.lower()
    assert "talking to" not in lower
    assert "another person" not in lower
    assert "two people" not in lower
    assert "person" in lower
    assert any(tok in lower for tok in ("navy", "jacket", "street", "outdoor"))
    assert len(caption.split()) >= 8


# --- 2. Multiple people ---
def test_multi_person_keeps_distinct_subjects() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-left"),
            SceneNode(1, "person_2", "person", 0.22, "middle-right"),
            SceneNode(2, "table_1", "table", 0.12, "bottom-center"),
        ),
        relations=(Relation(0, 2, "near", 0.8),),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "shirt_color", "red"),
            Attribute(1, "confidence", "88%"),
            Attribute(1, "shirt_color", "blue"),
            Attribute(2, "confidence", "80%"),
        ),
    )
    caption = _synth_caption(ctx)
    lower = caption.lower()
    assert "talking to a person" not in lower
    assert "person" in lower or "people" in lower
    assert any(
        tok in lower
        for tok in ("two people", "another person", "farther", "second", "while")
    )


# --- 3. Person + object ---
def test_person_plus_object_covers_both() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "laptop_1", "laptop", 0.12, "bottom-center"),
        ),
        relations=(Relation(0, 1, "near", 0.78),),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "shirt_color", "black"),
            Attribute(1, "confidence", "85%"),
        ),
    )
    caption = _synth_caption(ctx)
    lower = caption.lower()
    assert "person" in lower
    assert "laptop" in lower
    assert "holding" not in lower


# --- 4. Person holding object ---
def test_person_holding_object_preserves_interaction() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "cup_1", "cup", 0.05, "middle-center"),
        ),
        relations=(Relation(0, 1, "holding", 0.82),),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(1, "confidence", "80%"),
        ),
    )
    verified = build_verified_scene_evidence(ctx)
    assert any(r.relation_type == "holding" and r.narrative_safe for r in verified.relations)
    caption = _synth_caption(ctx)
    assert "hold" in caption.lower() or "cup" in caption.lower()


# --- 5. Person near object (spatial ≠ interaction) ---
def test_person_near_object_not_holding() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "table_1", "table", 0.15, "bottom-center"),
        ),
        relations=(Relation(0, 1, "near", 0.8),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
    )
    caption = _synth_caption(ctx)
    lower = caption.lower()
    assert "holding" not in lower
    assert "using" not in lower
    assert "talking" not in lower
    assert "table" in lower or "person" in lower


# --- 6. Crowded scene ---
def test_crowded_scene_mentions_multiple_people() -> None:
    nodes = tuple(
        SceneNode(i, f"person_{i}", "person", 0.2 - i * 0.01, "middle-center")
        for i in range(5)
    )
    attrs = tuple(Attribute(i, "confidence", "85%") for i in range(5))
    ctx = _ctx(nodes, attrs=attrs, env=_env(crowd="crowded", complexity="complex"))
    caption = _synth_caption(ctx)
    lower = caption.lower()
    assert any(tok in lower for tok in ("people", "several", "crowd", "other"))
    assert len(caption.split()) >= 10


# --- 7. Color-rich scene ---
def test_color_rich_scene_integrates_colors_naturally() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "chair_1", "chair", 0.1, "bottom-right"),
        ),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "shirt_color", "red"),
            Attribute(0, "pants_color", "blue"),
            Attribute(0, "clothing_type", "shirt"),
            Attribute(1, "confidence", "80%"),
            Attribute(1, "dominant_color", "brown"),
        ),
    )
    caption = _synth_caption(ctx)
    lower = caption.lower()
    assert "red" in lower or "blue" in lower
    assert not lower.startswith("person, red")
    assert ", red shirt, blue" not in lower


# --- 8. Indoor scene ---
def test_indoor_scene_uses_setting() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.28, "middle-center"),
            SceneNode(1, "chair_1", "chair", 0.1, "bottom-center"),
        ),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "80%")),
        env=_env(setting="office", scene_type="indoor", indoor_outdoor="indoor"),
    )
    caption = _synth_caption(ctx)
    lower = caption.lower()
    assert any(tok in lower for tok in ("indoor", "office", "room", "chair", "person"))


# --- 9. Outdoor scene ---
def test_outdoor_scene_uses_setting() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.28, "middle-center"),
            SceneNode(1, "tree_1", "tree", 0.12, "top-right"),
        ),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "75%")),
        env=_env(setting="park", scene_type="outdoor", indoor_outdoor="outdoor"),
    )
    caption = _synth_caption(ctx)
    lower = caption.lower()
    assert any(tok in lower for tok in ("outdoor", "park", "tree", "field", "street"))


# --- 10. Minimal scene ---
def test_minimal_scene_stays_concise_and_natural() -> None:
    ctx = _ctx(
        (SceneNode(0, "chair_1", "chair", 0.2, "middle-center"),),
        attrs=(Attribute(0, "confidence", "88%"),),
        env=_env(complexity="simple", crowd="empty"),
    )
    caption = _synth_caption(ctx)
    assert caption
    assert "chair" in caption.lower()
    assert len(caption.split()) < 80
    assert not caption.lower().startswith("the image shows")


# --- 11. Multiple objects ---
def test_multiple_objects_coverage() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.25, "middle-center"),
            SceneNode(1, "table_1", "table", 0.12, "bottom-center"),
            SceneNode(2, "bottle_1", "bottle", 0.04, "bottom-center"),
            SceneNode(3, "chair_1", "chair", 0.08, "bottom-right"),
        ),
        attrs=tuple(Attribute(i, "confidence", "85%") for i in range(4)),
    )
    caption = _synth_caption(ctx)
    lower = caption.lower()
    hits = sum(1 for tok in ("table", "bottle", "chair", "person") if tok in lower)
    assert hits >= 2


# --- 12. Multiple activities ---
def test_multiple_activities_grounded() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-left"),
            SceneNode(1, "person_2", "person", 0.22, "middle-right"),
            SceneNode(2, "bicycle_1", "bicycle", 0.1, "bottom-right"),
        ),
        relations=(Relation(1, 2, "riding", 0.8),),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "pose", "standing"),
            Attribute(1, "confidence", "88%"),
            Attribute(2, "confidence", "85%"),
        ),
        activities=(
            ActivityEvidence(
                activity="standing",
                confidence=0.7,
                supporting_node_indices=(0,),
                supporting_relation_types=(),
                rationale="pose",
            ),
            ActivityEvidence(
                activity="riding",
                confidence=0.8,
                supporting_node_indices=(1, 2),
                supporting_relation_types=("riding",),
                rationale="relation",
            ),
        ),
    )
    caption = _synth_caption(ctx)
    lower = caption.lower()
    assert "preparing food" not in lower
    assert "cooking" not in lower
    assert "person" in lower or "people" in lower


# --- 13. Unsupported interaction ---
def test_unsupported_interaction_filtered() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "laptop_1", "laptop", 0.1, "bottom-center"),
        ),
        relations=(Relation(0, 1, "near", 0.75),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
    )
    verified = build_verified_scene_evidence(ctx)
    dirty = (
        "A person is holding a laptop and talking to another person while preparing food "
        "in the kitchen."
    )
    score = score_caption_candidate(dirty, verified)
    factual = "A person stands beside a laptop in an indoor room."
    winner = arbitrate_captions([dirty, factual], verified)
    assert score.unsupported >= 1 or score.rejected or "interaction" in score.reason
    assert "talking" not in winner.lower()
    assert "preparing food" not in winner.lower()
    assert "holding" not in winner.lower()


# --- 14. Hallucinated candidate loses ---
def test_hallucinated_candidate_loses_arbitration() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "horse_1", "horse", 0.2, "middle-right"),
        ),
        relations=(Relation(0, 1, "leading", 0.8),),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "shirt_color", "black"),
            Attribute(1, "confidence", "88%"),
        ),
        env=_env(setting="field", scene_type="outdoor", indoor_outdoor="outdoor"),
    )
    verified = build_verified_scene_evidence(ctx)
    factual = (
        "A person wearing a black shirt leads a horse across a grassy field, "
        "with daylight across the outdoor scene."
    )
    hallucinated = (
        "A joyful teacher named Sarah is talking to her student while holding a smartphone "
        "and discussing their weekend plans near a red sports car."
    )
    winner = arbitrate_captions([hallucinated, factual], verified)
    assert "sarah" not in winner.lower()
    assert "teacher" not in winner.lower()
    assert "horse" in winner.lower() or "person" in winner.lower()


# --- 15. Repetition candidate cleaned / loses ---
def test_repetition_candidate_controlled() -> None:
    repeated = (
        "The person is standing near a table. The person is standing beside the table."
    )
    cleaned = sanitize_caption(repeated)
    assert cleaned.lower().count("standing") <= 1 or cleaned.lower().count("table") <= 1
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "table_1", "table", 0.12, "bottom-center"),
        ),
        relations=(Relation(0, 1, "near", 0.8),),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
    )
    verified = build_verified_scene_evidence(ctx)
    natural = (
        "A person stands beside a table in an indoor room, with the table visible nearby."
    )
    winner = arbitrate_captions([repeated, natural], verified)
    assert "standing near" not in winner.lower() or winner == natural or len(winner.split(".")) <= 2


# --- 16. Factual but too short loses to richer equal-factual ---
def test_short_factual_loses_to_richer_coverage() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "horse_1", "horse", 0.22, "middle-right"),
            SceneNode(2, "horse_2", "horse", 0.15, "middle-left"),
            SceneNode(3, "tree_1", "tree", 0.08, "top-center"),
        ),
        relations=(Relation(0, 1, "leading", 0.80),),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "shirt_color", "black"),
            Attribute(1, "confidence", "88%"),
            Attribute(2, "confidence", "85%"),
            Attribute(3, "confidence", "75%"),
        ),
        env=_env(setting="field", scene_type="outdoor", indoor_outdoor="outdoor"),
    )
    verified = build_verified_scene_evidence(ctx)
    detailed = (
        "In a grassy outdoor field, a person wearing a black shirt leads a horse, "
        "while another horse stands nearby and trees rise in the background under daylight."
    )
    short = "A person is outside."
    stub = "A person stands near a horse."
    winner = arbitrate_captions([short, stub, detailed], verified)
    assert len(winner.split()) >= 20
    assert "horse" in winner.lower()
    assert "black" in winner.lower() or "leads" in winner.lower() or "lead" in winner.lower()


def test_thin_proximity_stub_penalized() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "table_1", "table", 0.12, "bottom-center"),
            SceneNode(2, "laptop_1", "laptop", 0.08, "bottom-center"),
            SceneNode(3, "chair_1", "chair", 0.07, "bottom-right"),
        ),
        relations=(Relation(0, 1, "near", 0.8), Relation(0, 2, "near", 0.75)),
        attrs=tuple(Attribute(i, "confidence", "85%") for i in range(4)),
    )
    verified = build_verified_scene_evidence(ctx)
    stub = "A person stands near a table."
    richer = (
        "A person stands beside a table in an indoor room, with a laptop and a chair "
        "also visible nearby."
    )
    score_stub = score_caption_candidate(stub, verified)
    score_rich = score_caption_candidate(richer, verified)
    assert score_rich.total > score_stub.total
    winner = arbitrate_captions([stub, richer], verified)
    assert "laptop" in winner.lower() or "chair" in winner.lower()


def test_florence_narrate_prompt_is_task_token_only() -> None:
    from language.vlm.prompt_builders import VisionPromptBuilder

    brief = language_understanding_from_verified(
        build_verified_scene_evidence(
            _ctx((SceneNode(0, "person_1", "person", 0.3, "middle-center"),))
        )
    )
    prompt = VisionPromptBuilder().narrate_prompt("florence2", brief)
    assert prompt == "<MORE_DETAILED_CAPTION>"


def test_writers_cannot_bypass_verified_for_talking_to() -> None:
    """Architectural: talking_to never becomes narrative-safe from proximity alone."""
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-left"),
            SceneNode(1, "person_2", "person", 0.25, "middle-right"),
        ),
        relations=(Relation(0, 1, "talking_to", 0.9), Relation(0, 1, "near", 0.85)),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "88%")),
    )
    verified = build_verified_scene_evidence(ctx)
    assert not any(
        r.relation_type == "talking_to" and r.narrative_safe for r in verified.relations
    )
    caption = _synth_caption(ctx)
    assert "talking to a person" not in caption.lower()


def test_unsupported_claim_sentence_marked() -> None:
    ctx = _ctx(
        (SceneNode(0, "person_1", "person", 0.3, "middle-center"),),
        attrs=(Attribute(0, "confidence", "90%"),),
    )
    verified = build_verified_scene_evidence(ctx)
    verdict = classify_sentence_against_verified(
        "A person talking to a person stands in the frame.",
        verified,
    )
    assert verdict.status == ClaimSupport.UNSUPPORTED


def test_filter_drops_robotic_talking_stub() -> None:
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-left"),
            SceneNode(1, "person_2", "person", 0.2, "middle-right"),
        ),
        attrs=(Attribute(0, "confidence", "90%"), Attribute(1, "confidence", "85%")),
    )
    verified = build_verified_scene_evidence(ctx)
    text = (
        "A person talking to a person. Two people share an indoor room with natural light."
    )
    cleaned = filter_unsupported_claims_verified(text, verified)
    assert "talking to a person" not in cleaned.lower()


def test_prominent_verified_hazard_survives_caption_arbitration() -> None:
    """Prominent verified scene element must survive final caption arbitration."""
    from language.refinement.caption_coverage import ensure_salient_verified_coverage

    env = EnvironmentInfo(
        scene_type="outdoor",
        setting="field",
        time_of_day="day",
        weather="unknown",
        indoor_outdoor="outdoor",
        social_context="",
        crowd_level="few",
        scene_complexity="medium",
        evidence=("Hazard detected: fire (confidence: 88%)",),
    )
    ctx = _ctx(
        (
            SceneNode(0, "person_1", "person", 0.3, "middle-center"),
            SceneNode(1, "horse_1", "horse", 0.22, "middle-right"),
        ),
        relations=(Relation(0, 1, "leading", 0.80),),
        attrs=(
            Attribute(0, "confidence", "90%"),
            Attribute(0, "shirt_color", "black"),
            Attribute(1, "confidence", "88%"),
        ),
        env=env,
    )
    verified = build_verified_scene_evidence(ctx)
    assert any("fire" in (line or "").lower() for line in verified.scene.evidence)

    with_fire = (
        "In a grassy field, a person wearing black clothing leads a horse by a rope. "
        "A fire is burning nearby."
    )
    without_fire = (
        "In a grassy field, a person wearing black clothing leads a horse by a rope."
    )
    # Shorter candidate omitting verified fire must not win solely on length/fluency.
    winner = arbitrate_captions([without_fire, with_fire], verified)
    assert "fire" in winner.lower()

    # Coverage repair restores fire when arbitration somehow left it out.
    repaired = ensure_salient_verified_coverage(without_fire, verified=verified)
    assert "fire" in repaired.lower()
    assert "entities" not in repaired.lower()
    assert "attributes" not in repaired.lower()
    # Do not invent smoke from fire alone.
    assert "smoke" not in repaired.lower()

    # Unverified smoke wording must not survive coverage repair.
    with_unverified_smoke = (
        "In a grassy field, a person leads a horse. "
        "In the foreground, a fire burns, sending smoke into the air."
    )
    cleaned_smoke = ensure_salient_verified_coverage(with_unverified_smoke, verified=verified)
    assert "fire" in cleaned_smoke.lower()
    assert "smoke" not in cleaned_smoke.lower()
