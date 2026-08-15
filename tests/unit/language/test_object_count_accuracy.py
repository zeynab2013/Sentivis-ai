"""General object-count + cross-sentence evidence consistency regressions."""

from __future__ import annotations

import re

from core.contracts.reasoning import EvidenceFact, SceneUnderstanding
from core.contracts.verified_evidence import VerifiedEntity, VerifiedSceneContext, VerifiedSceneEvidence
from language.validation.caption_factuality import (
    clamp_caption_object_counts,
    label_counts_from_verified,
    _verified_label_counts,
)


def _understanding(subjects: tuple[str, ...]) -> SceneUnderstanding:
    facts = tuple(
        EvidenceFact(s, "is", s.split("#")[0].replace("_", " ").strip(), 0.9, "yolo")
        for s in subjects
    )
    return SceneUnderstanding(
        facts=facts,
        ranked_subjects=subjects,
        environment_keys=(),
        activity_keys=(),
        ocr_text=(),
        evidence_brief=" ".join(subjects),
        overall_confidence=0.9,
        discarded_count=0,
        contradictions_resolved=0,
    )


def _verified(labels: list[tuple[str, str]]) -> VerifiedSceneEvidence:
    entities = tuple(
        VerifiedEntity(
            entity_id=eid,
            object_index=i,
            label=lab,
            confidence=0.9,
            narrative_safe=True,
        )
        for i, (eid, lab) in enumerate(labels)
    )
    return VerifiedSceneEvidence(
        entities=entities,
        attributes=(),
        relations=(),
        activities=(),
        scene=VerifiedSceneContext(confidence=0.5),
        ocr_text=(),
        evidence_brief="",
        overall_confidence=0.9,
    )


def test_a_one_object_cannot_inflate() -> None:
    u = _understanding(("refrigerator #1",))
    assert _verified_label_counts(u).get("refrigerator") == 1
    out = clamp_caption_object_counts(
        "4 brown refrigerators appear farther back.",
        u,
    ).lower()
    assert "4" not in out
    assert "refrigerator" in out
    assert not re.search(r"\b[2-9]\s+\w*\s*refrigerators?\b", out)


def test_b_multiple_same_class_keeps_exact_count() -> None:
    u = _understanding(("chair #1", "chair #2", "chair #3", "chair #4"))
    out = clamp_caption_object_counts("4 brown chairs surround the table.", u)
    assert "4 brown chairs" in out.lower()


def test_c_nested_duplicate_subject_ids_count_once() -> None:
    u = _understanding(("chair #1", "chair #1", "chair"))
    assert _verified_label_counts(u).get("chair") == 1


def test_d_overlapping_legitimate_instances_preserved() -> None:
    u = _understanding(("chair #1", "chair #2", "chair #3", "chair #4"))
    assert _verified_label_counts(u).get("chair") == 4


def test_e_same_object_not_counted_twice_across_sentences() -> None:
    u = _understanding(("refrigerator #1", "chair #1", "chair #2", "chair #3", "chair #4"))
    text = (
        "A beige refrigerator is visible behind them. "
        "4 beige refrigerators and a clock appear farther back. "
        "2 brown chairs surround the table. "
        "4 brown chairs appear nearby."
    )
    out = clamp_caption_object_counts(text, u).lower()
    assert out.count("refrigerator") == 1
    assert "4 brown chairs" in out
    assert "2 brown chairs" not in out
    assert not re.search(r"\b[2-9]\s+\w*\s*refrigerators?\b", out)


def test_f_color_modifiers_do_not_block_count_lookup() -> None:
    u = _understanding(("horse #1",))
    out = clamp_caption_object_counts("Three brown horses stand nearby.", u).lower()
    assert "three" not in out
    assert "horse" in out


def test_g_multiple_people_census_preserved_with_activity_head() -> None:
    u = _understanding(("person #1", "person #2", "bicycle #1"))
    out = clamp_caption_object_counts(
        "A person is riding a bicycle. Two people are visible in the scene.",
        u,
    )
    lower = out.lower()
    assert "a person is riding a bicycle" in lower
    assert "2 people" in lower


def test_h_multiple_animals_not_collapsed() -> None:
    u = _understanding(("horse #1", "horse #2"))
    out = clamp_caption_object_counts("Several horses stand in the field. A horse is nearby.", u)
    lower = out.lower()
    assert "2 horses" in lower
    assert lower.count("horse") >= 1
    assert "a horse is nearby" not in lower


def test_i_vehicle_scene_exact_count() -> None:
    u = _understanding(("car #1", "car #2", "car #3"))
    out = clamp_caption_object_counts("5 cars are parked on the street.", u)
    assert "3 cars" in out.lower()
    assert "5 cars" not in out.lower()


def test_j_verified_evidence_is_authoritative() -> None:
    verified = _verified(
        [("fridge_1", "refrigerator"), ("chair_1", "chair"), ("chair_2", "chair")]
    )
    assert label_counts_from_verified(verified) == {"refrigerator": 1, "chair": 2}
    out = clamp_caption_object_counts(
        "4 refrigerators and 5 chairs appear farther back.",
        verified=verified,
    ).lower()
    assert "4 refrigerators" not in out
    assert "5 chairs" not in out
    assert "2 chairs" in out


def test_k_uncertain_empty_counts_do_not_invent() -> None:
    out = clamp_caption_object_counts("4 dragons appear nearby.", counts={})
    assert "4 dragons" in out.lower()


def test_l_under_and_over_counts_both_repaired() -> None:
    u = _understanding(("bottle #1", "bottle #2", "bottle #3"))
    assert "3 bottles" in clamp_caption_object_counts("2 bottles are nearby.", u).lower()
    assert "3 bottles" in clamp_caption_object_counts("9 bottles are nearby.", u).lower()


def test_m_stop_sign_regression_still_clamps() -> None:
    understanding = SceneUnderstanding(
        facts=(EvidenceFact("stop_sign_1", "label", "stop sign", 0.9, "yolo"),),
        ranked_subjects=("stop_sign_1",),
        environment_keys=(),
        activity_keys=(),
        ocr_text=("STOP",),
        evidence_brief="stop_sign_1",
        overall_confidence=0.9,
        discarded_count=0,
        contradictions_resolved=0,
    )
    cleaned = clamp_caption_object_counts("4 stop signs are nearby.", understanding)
    assert "4 stop signs" not in cleaned.lower()
    assert "stop sign" in cleaned.lower()
