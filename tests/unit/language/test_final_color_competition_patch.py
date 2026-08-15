"""FINAL COLOR competition regressions A–G (entity-bound color only)."""

from __future__ import annotations

import numpy as np

from analysis.common.color_utils import dominant_color_for_entity, normalize_simple_color_name
from core.contracts.detection import BoundingBox
from language.assistant.evidence_packet import AssistantEvidencePacket, EvidenceItem
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever


def _packet(items: tuple[EvidenceItem, ...], *, objects: tuple[str, ...] = (), caption: str = "") -> AssistantEvidencePacket:
    return AssistantEvidencePacket(
        objects=objects,
        attributes=(),
        relations=(),
        activities=(),
        environment=(),
        ocr=(),
        evidence_brief="",
        canonical_caption_en=caption,
        items=items,
        from_verified=True,
    )


def test_A_person1_red_not_person2_blue() -> None:
    items = (
        EvidenceItem("object", "person", "is", "person", 0.95, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("object", "person", "is", "person", 0.94, entity_id="person_2", object_index=1, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "shirt_color", "red", 0.93, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "clothing_color", "red", 0.93, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "shirt_color", "blue", 0.93, entity_id="person_2", object_index=1, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "clothing_color", "blue", 0.93, entity_id="person_2", object_index=1, claim_status="OBSERVED"),
    )
    packet = _packet(
        items,
        objects=("person_1: person zone=middle-left area=0.2", "person_2: person zone=middle-right area=0.2"),
    )
    ans = VisualEvidenceRetriever().try_direct_answer(
        packet, "What color clothing is the first person wearing?"
    )
    assert "red" in ans.lower()
    assert "blue" not in ans.lower()


def test_B_white_sports_ball_not_beige_from_grass() -> None:
    pixels = np.full((80, 80, 3), (50, 140, 55), dtype=np.uint8)
    pixels[28:52, 28:52] = (242, 242, 238)
    name = dominant_color_for_entity(pixels, BoundingBox(20, 20, 60, 60), None, label="sports ball")
    assert name == "white"
    assert name not in {"beige", "green", "olive"}


def test_C_bicycle_never_grass_green() -> None:
    pixels = np.full((100, 100, 3), (45, 150, 55), dtype=np.uint8)
    pixels[32:68, 38:62] = (18, 18, 22)
    name = dominant_color_for_entity(pixels, BoundingBox(28, 28, 72, 72), None, label="bicycle")
    assert name not in {"green", "olive", "dark green", "forest green", "olive green"}
    assert name in {"black", "gray", "unknown", "charcoal"}


def test_D_brown_horse_near_grass() -> None:
    pixels = np.full((100, 100, 3), (45, 150, 55), dtype=np.uint8)
    pixels[25:75, 30:70] = (110, 70, 40)
    name = dominant_color_for_entity(pixels, BoundingBox(22, 22, 78, 78), None, label="horse")
    assert name == "brown"
    assert name not in {"green", "olive"}


def test_E_motorcycle_shirt_red_pants_brown_entity_bound() -> None:
    items = (
        EvidenceItem("object", "person", "is", "person", 0.95, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "shirt_color", "red", 0.93, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "clothing_color", "red", 0.93, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "pants_color", "brown", 0.93, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "color", "brown", 0.90, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
    )
    packet = _packet(items, objects=("person_1: person zone=middle-center area=0.3",), caption="A person riding a motorcycle.")
    shirt = VisualEvidenceRetriever().try_direct_answer(packet, "What color shirt is the person wearing?")
    pants = VisualEvidenceRetriever().try_direct_answer(packet, "What color pants is the person wearing?")
    assert "red" in shirt.lower()
    assert "brown" in pants.lower()
    assert "blue" not in shirt.lower()


def test_F_multi_person_clothing_stays_entity_bound() -> None:
    items = (
        EvidenceItem("object", "person", "is", "person", 0.95, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("object", "person", "is", "person", 0.94, entity_id="person_2", object_index=1, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "shirt_color", "red", 0.93, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "clothing_color", "red", 0.93, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "shirt_color", "blue", 0.93, entity_id="person_2", object_index=1, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "clothing_color", "blue", 0.93, entity_id="person_2", object_index=1, claim_status="OBSERVED"),
    )
    packet = _packet(
        items,
        objects=("person_1: person zone=middle-left area=0.25", "person_2: person zone=middle-right area=0.22"),
    )
    first = VisualEvidenceRetriever().try_direct_answer(packet, "What color clothing is the first person wearing?")
    second = VisualEvidenceRetriever().try_direct_answer(packet, "What color clothing is the second person wearing?")
    assert "red" in first.lower() and "blue" not in first.lower()
    assert "blue" in second.lower() and "red" not in second.lower()


def test_G_ambiguous_the_person_does_not_steal_color() -> None:
    items = (
        EvidenceItem("object", "person", "is", "person", 0.95, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("object", "person", "is", "person", 0.94, entity_id="person_2", object_index=1, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "shirt_color", "red", 0.93, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "clothing_color", "red", 0.93, entity_id="person_1", object_index=0, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "shirt_color", "blue", 0.93, entity_id="person_2", object_index=1, claim_status="OBSERVED"),
        EvidenceItem("attribute", "person", "clothing_color", "blue", 0.93, entity_id="person_2", object_index=1, claim_status="OBSERVED"),
    )
    packet = _packet(
        items,
        objects=("person_1: person zone=middle-left area=0.30", "person_2: person zone=middle-right area=0.20"),
    )
    ans = VisualEvidenceRetriever().try_direct_answer(packet, "What color clothing is the person wearing?")
    lower = ans.lower()
    assert "which person" in lower
    assert "red" not in lower
    assert "blue" not in lower


def test_vocabulary_preserves_cream_olive_burgundy() -> None:
    assert normalize_simple_color_name("cream") == "cream"
    assert normalize_simple_color_name("olive") == "olive"
    assert normalize_simple_color_name("burgundy") == "burgundy"
    assert normalize_simple_color_name("navy") == "dark blue"
    assert normalize_simple_color_name("olive green") == "olive"
