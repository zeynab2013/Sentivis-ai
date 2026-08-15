"""Deterministic person/entity indexing shared by QA and suggested questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from language.assistant.evidence_packet import AssistantEvidencePacket, EvidenceItem

_PERSON_LABELS = frozenset({"person", "man", "woman", "child", "people", "skier", "rider"})

_ORDINAL_WORDS = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
}

_ZONE_LEFT = frozenset(
    {
        "left",
        "middle-left",
        "top-left",
        "bottom-left",
        "far-left",
    }
)
_ZONE_RIGHT = frozenset(
    {
        "right",
        "middle-right",
        "top-right",
        "bottom-right",
        "far-right",
    }
)
_ZONE_BACK = frozenset(
    {
        "background",
        "far",
        "top-center",
        "top-left",
        "top-right",
    }
)


@dataclass(frozen=True)
class IndexedPerson:
    """One person entity in stable QA order (1-based ordinal)."""

    ordinal: int
    entity_id: str
    object_index: int
    label: str
    confidence: float
    position_zone: str = ""
    area_ratio: float = 0.0


def _parse_zone_and_area(packet: AssistantEvidencePacket, entity_id: str) -> tuple[str, float]:
    """Extract zone/area from packet.objects lines when available."""
    prefix = f"{(entity_id or '').lower()}:"
    for line in packet.objects:
        if not line.lower().startswith(prefix):
            continue
        zone = ""
        area = 0.0
        zm = re.search(r"zone=([^,\s)]+)", line, flags=re.I)
        if zm:
            zone = zm.group(1).strip().lower()
        am = re.search(r"area=([0-9.]+)", line, flags=re.I)
        if am:
            try:
                area = float(am.group(1))
            except ValueError:
                area = 0.0
        return zone, area
    return "", 0.0


def ordered_people(packet: AssistantEvidencePacket) -> list[IndexedPerson]:
    """Stable person order used by both suggestion and answering.

    Order:
    1) entity_id numeric suffix (person_1, person_2, …) when present
    2) otherwise detection object_index
    Within ties, larger area then higher confidence.
    """
    people: list[EvidenceItem] = []
    for item in packet.items:
        if item.kind != "object":
            continue
        lab = (item.subject or "").lower().strip()
        eid = (item.entity_id or "").lower().strip()
        if lab not in _PERSON_LABELS and not eid.startswith("person"):
            continue
        if item.confidence < 0.42:
            continue
        # Align with caption narrative_safe gate (UNCERTAIN = not narrative_safe).
        if (item.claim_status or "").upper() == "UNCERTAIN":
            continue
        people.append(item)

    def _sort_key(item: EvidenceItem) -> tuple[int, int, float, float]:
        eid = (item.entity_id or "").lower()
        m = re.search(r"_(\d+)$", eid)
        eid_ord = int(m.group(1)) if m else 10_000
        idx = item.object_index if item.object_index >= 0 else 10_000
        zone, area = _parse_zone_and_area(packet, eid)
        _ = zone
        # Prefer larger / more confident detections when eid order ties.
        return (eid_ord, idx, -area, -item.confidence)

    people.sort(key=_sort_key)
    indexed: list[IndexedPerson] = []
    for i, item in enumerate(people, start=1):
        eid = (item.entity_id or f"person_{i}").lower()
        zone, area = _parse_zone_and_area(packet, eid)
        indexed.append(
            IndexedPerson(
                ordinal=i,
                entity_id=eid,
                object_index=item.object_index,
                label=(item.subject or "person").lower(),
                confidence=item.confidence,
                position_zone=zone,
                area_ratio=area,
            )
        )
    return indexed


def resolve_person_reference(
    question: str,
    packet: AssistantEvidencePacket,
) -> IndexedPerson | None:
    """Map first/second/left/right/background person phrases to one IndexedPerson.

    Returns None when the question does not target a specific person index,
    or when the requested index/side does not exist.
    """
    q = (question or "").lower()
    people = ordered_people(packet)
    if not people:
        return None

    # Explicit ordinal: "second person", "person 2", "the 2nd person"
    for word, ord_n in _ORDINAL_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\s+person\b", q) or re.search(
            rf"\bperson\s*{ord_n}\b", q
        ):
            for person in people:
                if person.ordinal == ord_n:
                    return person
            return None

    # Spatial side / depth references.
    if re.search(r"\bperson\b.*\b(on the )?left\b|\bleft[- ]hand person\b|\bleft person\b", q):
        left = [p for p in people if p.position_zone in _ZONE_LEFT or "left" in p.position_zone]
        if left:
            return left[0]
        # Fallback: leftmost by zone token order, else first person.
        return people[0]
    if re.search(r"\bperson\b.*\b(on the )?right\b|\bright[- ]hand person\b|\bright person\b", q):
        right = [p for p in people if p.position_zone in _ZONE_RIGHT or "right" in p.position_zone]
        if right:
            return right[0]
        return people[-1] if len(people) > 1 else people[0]
    if re.search(
        r"\bperson\b.*\b(background|farther back|behind)\b|\bbackground person\b",
        q,
    ):
        back = [p for p in people if p.position_zone in _ZONE_BACK or "back" in p.position_zone]
        if back:
            # Smallest area among background-tagged people, else last ordinal.
            return min(back, key=lambda p: (p.area_ratio or 1.0, -p.ordinal))
        if len(people) >= 2:
            return people[-1]
        return people[0]

    # Unindexed "the person" / "person wearing" → prefer the most salient actor:
    # person with CONFIRMED activity, else largest area, else person_1.
    if len(people) == 1:
        return people[0]
    if re.search(r"\b(the )?person\b", q) and not any(
        tok in q for tok in ("people", "persons", "both", "other person", "another person")
    ):
        confirmed_actors: set[str] = set()
        for item in packet.items:
            if item.kind != "activity":
                continue
            if (item.evidence_level or "").upper() != "CONFIRMED":
                continue
            eid = (item.entity_id or "").lower()
            if eid.startswith("person"):
                confirmed_actors.add(eid)
        if confirmed_actors:
            actors = [p for p in people if p.entity_id in confirmed_actors]
            if actors:
                return max(actors, key=lambda p: (p.area_ratio, p.confidence))
        # Prefer the visually dominant person when no activity binding exists.
        return max(people, key=lambda p: (p.area_ratio, p.confidence))
    return None


def person_phrase(person: IndexedPerson | None, *, total: int = 0) -> str:
    """Natural subject phrase for answers."""
    if person is None:
        return "The person"
    if total <= 1 and person.ordinal == 1:
        return "The person"
    ordinal_names = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
    }
    name = ordinal_names.get(person.ordinal, str(person.ordinal))
    return f"The {name} person"


def find_person_attribute(
    packet: AssistantEvidencePacket,
    person: IndexedPerson,
    *,
    predicates: tuple[str, ...],
    require_reliable: bool = True,
    min_confidence: float = 0.55,
) -> EvidenceItem | None:
    """Best attribute for one indexed person across candidate predicates.

    Precedence:
    1) OBSERVED over INFERRED/UNCERTAIN
    2) Earlier predicate in ``predicates`` (e.g. shirt_color before clothing_color)
    3) Higher confidence
    Never binds clothing attrs from non-person entities.
    """
    best: EvidenceItem | None = None
    best_key: tuple[int, int, float] | None = None
    pred_rank = {p.lower(): i for i, p in enumerate(predicates)}
    preds = set(pred_rank)
    for item in packet.items:
        if item.kind != "attribute":
            continue
        pred = item.predicate.lower()
        if pred not in preds:
            continue
        if require_reliable and not item.reliable:
            continue
        if item.confidence < min_confidence:
            continue
        eid = (item.entity_id or "").lower()
        same_entity = eid and eid == person.entity_id
        same_index = item.object_index >= 0 and item.object_index == person.object_index
        if not (same_entity or same_index):
            continue
        # Never accept clothing attrs leaked onto non-person entities.
        subj = (item.subject or "").lower()
        if subj and subj not in _PERSON_LABELS and not eid.startswith("person"):
            continue
        status = (item.claim_status or "").upper()
        observed = 1 if status == "OBSERVED" else 0
        # Lower rank number = preferred predicate.
        rank = pred_rank.get(pred, 99)
        key = (observed, -rank, float(item.confidence))
        if best is None or best_key is None or key > best_key:
            best = item
            best_key = key
    return best
