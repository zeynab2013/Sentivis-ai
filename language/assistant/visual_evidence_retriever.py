"""VisualEvidenceRetriever — question → SceneContext evidence (never caption-only)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from language.assistant.evidence_packet import (
    AssistantEvidencePacket,
    EvidenceItem,
    find_attribute,
)
from language.assistant.entity_indexing import (
    find_person_attribute,
    ordered_people,
    person_phrase,
    resolve_person_reference,
)

_ATTR_CONF_MIN = 0.55
_DETECT_CONF_MIN = 0.50
_COLOR_ATTRS = {
    "clothing_color",
    "shirt_color",
    "pants_color",
    "shoes_color",
    "hair_color",
    "dominant_color",
    "secondary_color",
    "color",
}

_CLOTHING_COLOR_PREDS = frozenset(
    {
        "clothing_color",
        "shirt_color",
        "pants_color",
        "shoes_color",
        "hair_color",
    }
)

_OBJECT_COLOR_LABELS = frozenset(
    {
        "tree",
        "bus",
        "truck",
        "bag",
        "phone",
        "keyboard",
        "chair",
        "couch",
        "cup",
        "vase",
        "refrigerator",
        "clock",
        "potted plant",
        "sports ball",
        "ball",
        "motorcycle",
        "bicycle",
        "skateboard",
        "surfboard",
        "tennis racket",
        "baseball bat",
        "frisbee",
        "kite",
        "handbag",
        "backpack",
        "suitcase",
        "umbrella",
        "bottle",
        "bowl",
        "wine glass",
        "fork",
        "knife",
        "spoon",
        "banana",
        "apple",
        "orange",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "bench",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "book",
        "teddy bear",
        "hair drier",
        "toothbrush",
        "car",
        "van",
        "boat",
        "train",
        "airplane",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
    }
)

_PERSON = {"person", "man", "woman", "child", "people", "skier", "rider"}
_EQUIPMENT = {
    "skis",
    "ski",
    "snowboard",
    "pole",
    "poles",
    "racket",
    "bat",
    "bicycle",
    "surfboard",
    "skateboard",
    "tennis racket",
    "baseball bat",
}
_VEHICLE = {"car", "bus", "truck", "motorcycle", "bicycle", "van", "taxi"}
_ANIMALS = {
    "horse",
    "dog",
    "cat",
    "cow",
    "sheep",
    "bird",
    "goat",
    "pony",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}
_FIRE_LABELS = {"fire", "flame", "campfire", "bonfire"}
_SMOKE_LABELS = {"smoke"}
# Fashion-ambiguous colors — require high confidence or refuse.
_AMBIGUOUS_CLOTHING_COLORS = {
    "khaki",
    "olive",
    "beige",
    "tan",
    "cream",
    "champagne",
    "taupe",
    "sand",
    "mustard",
    "coral",
}
_SAFE_CAPTION_CLOTHING_COLORS = (
    "black",
    "white",
    "red",
    "blue",
    "green",
    "gray",
    "grey",
    "navy",
    "brown",
    "yellow",
    "pink",
    "purple",
    "orange",
    "charcoal",
    "maroon",
)
_CLOTHING_CUE_RE = re.compile(
    r"\b(?:wearing|wear|jacket|clothing|outfit|dressed|shirts?|t-shirts?|"
    r"pants|dress|dresses|coats?|hats?)\b",
    flags=re.I,
)
_INTERACTION_PREDICATES = {
    "holding",
    "leading",
    "guiding",
    "riding",
    "carrying",
    "using",
    "sitting_on",
}


@dataclass(frozen=True)
class EvidenceRetrievalResult:
    """Structured retrieval result for one user question."""

    question: str
    selected: tuple[EvidenceItem, ...]
    prompt_block: str
    direct_answer_en: str = ""
    has_reliable_match: bool = False


class VisualEvidenceRetriever:
    """Retrieve and optionally answer from structured visual evidence only."""

    def retrieve(self, packet: AssistantEvidencePacket, question: str) -> EvidenceRetrievalResult:
        q = " ".join((question or "").split()).strip()
        selected = self._select(packet, q)
        direct = self.try_direct_answer(packet, q, selected=selected)
        prompt = self._format_prompt(packet, q, selected, direct)
        return EvidenceRetrievalResult(
            question=q,
            selected=tuple(selected),
            prompt_block=prompt,
            direct_answer_en=direct,
            has_reliable_match=bool(direct) or any(item.reliable for item in selected),
        )

    def try_direct_answer(
        self,
        packet: AssistantEvidencePacket,
        question: str,
        *,
        selected: list[EvidenceItem] | None = None,
    ) -> str:
        """Deterministic grounded answer when evidence clearly supports it.

        Returns empty string when the question needs open-ended Gemma reasoning
        or when evidence is insufficient (caller should refuse appropriately).
        """
        q = (question or "").lower()
        items = selected if selected is not None else self._select(packet, question)
        answer = self._try_direct_answer_raw(packet, q, items=items)
        if not answer:
            return ""
        if not self._answer_consistent_with_evidence(packet, q, answer):
            if any(tok in q for tok in ("doing", "activity", "action", "happening")):
                return (
                    "I can't determine the person's exact activity "
                    "from the available visual evidence."
                )
            if "how many" in q or "number of" in q:
                # Prefer verified count over a contradictory draft.
                recount = self._count_answer(packet, q)
                if recount and self._answer_consistent_with_evidence(packet, q, recount):
                    return recount
                return (
                    "I can't reliably determine an exact count "
                    "from the available visual evidence."
                )
            return (
                "The available visual evidence does not confirm that detail "
                "clearly enough to answer confidently."
            )
        return self._strip_appended_caption(answer, packet)

    def _try_direct_answer_raw(
        self,
        packet: AssistantEvidencePacket,
        q: str,
        *,
        items: list[EvidenceItem],
    ) -> str:
        """Internal direct-answer handlers (pre-consistency gate)."""

        # Age is never answerable.
        if any(tok in q for tok in ("exact age", "how old", "person's age", "their age")):
            return "The person's exact age cannot be determined from the visual evidence."

        # Names are never answerable from visual evidence alone.
        if any(
            phrase in q
            for phrase in (
                "person's name",
                "their name",
                "his name",
                "her name",
                "what is the name",
                "what's the name",
                "who is named",
            )
        ) or (re.search(r"\bname\b", q) and "person" in q):
            return "The person's name cannot be determined from the visual evidence."

        # Fire / smoke presence — never dump the evidence brief.
        asks_fire_presence = any(
            tok in q
            for tok in (
                "is there",
                "are there",
                "visible",
                "present",
                "in the scene",
                "in the image",
                "can you see",
                "do you see",
                "any fire",
                "any smoke",
            )
        )
        if any(tok in q for tok in ("fire", "smoke", "flame")) and asks_fire_presence:
            fire_answer = self._fire_smoke_answer(packet, q)
            if fire_answer:
                return fire_answer

        # Animals visible.
        if any(
            phrase in q
            for phrase in (
                "what animals",
                "which animals",
                "any animals",
                "animals are visible",
                "animals visible",
                "animal is visible",
            )
        ) or ("animal" in q and any(tok in q for tok in ("what", "which", "visible", "see"))):
            animals = self._animals_answer(packet)
            if animals:
                return animals

        # Presence: "is there a horse", "is a refrigerator visible"
        # Do not steal "how many … are there" count questions.
        if "how many" not in q and "number of" not in q and (
            re.search(r"\b(?:is|are)\s+there\b", q)
            or (
                "visible" in q
                and re.search(r"\b(?:is|are)\s+(?:a|an|any|the)\b", q)
            )
        ):
            presence = self._presence_answer(packet, q)
            if presence:
                return presence

        # Shoe color — only if reliable; otherwise explicit unknown.
        if any(tok in q for tok in ("shoe", "shoes", "sneaker", "footwear")) and "color" in q:
            person = resolve_person_reference(q, packet)
            shoe = None
            if person is not None:
                shoe = find_person_attribute(
                    packet,
                    person,
                    predicates=("shoes_color",),
                    require_reliable=True,
                    min_confidence=0.60,
                )
            if shoe is None and person is None:
                shoe = find_attribute(packet, predicate="shoes_color", require_reliable=True)
            if shoe is None or shoe.confidence < 0.60:
                return "The shoes are not clearly visible enough to determine their color."
            who = person_phrase(person, total=len(ordered_people(packet)))
            return f"{who}'s shoes appear {shoe.value}."

        # Continue with remaining handlers via legacy path body below.
        return self._try_direct_answer_continued(packet, q, items=items)

    def _answer_consistent_with_evidence(
        self,
        packet: AssistantEvidencePacket,
        question: str,
        answer: str,
    ) -> bool:
        """Reject confident answers that contradict verified evidence facts."""
        q = (question or "").lower()
        a = (answer or "").lower()
        if not a:
            return False
        # Uncertainty answers are always consistent.
        if any(
            tok in a
            for tok in (
                "can't",
                "cannot",
                "could not",
                "not clearly",
                "not confirm",
                "insufficient",
                "does not confirm",
                "no people are clearly",
                "no horses are clearly",
                "no vehicles are clearly",
            )
        ):
            return True

        people = ordered_people(packet)
        verified_people = len(people)

        # Count questions must match verified narrative-safe people count.
        if ("how many" in q or "number of" in q) and any(
            tok in q for tok in ("people", "person", "persons")
        ):
            stated = self._extract_stated_count(a)
            if stated is not None and stated != verified_people:
                return False

        # Activity answers must cite a CONFIRMED/SUPPORTED activity in the packet.
        if any(tok in q for tok in ("doing", "activity", "action")) or (
            "happening" in q and any(tok in q for tok in ("person", "people"))
        ):
            acts = [
                (item.value or "").lower()
                for item in packet.items
                if item.kind == "activity"
                and item.reliable
                and item.confidence >= 0.58
                and (item.claim_status or "").upper() != "UNCERTAIN"
                and (item.evidence_level or "").upper() in {"CONFIRMED", "SUPPORTED", ""}
            ]
            if not acts:
                # No verified activity → any concrete activity claim is inconsistent.
                if any(
                    tok in a
                    for tok in (
                        "playing",
                        "cooking",
                        "riding",
                        "driving",
                        "skiing",
                        "working",
                        "running",
                        "office",
                        "tennis",
                    )
                ):
                    return False
                return True
            # Answer should mention at least one verified activity token.
            if not any(
                any(tok in a for tok in re.findall(r"[a-z]{4,}", act) if tok not in {"with", "from", "scene"})
                for act in acts
            ):
                return False
            # Reject answers that invent a competing performance not in evidence.
            for banned in ("office work", "playing tennis", "driving"):
                if banned in a and not any(banned in act for act in acts):
                    return False

        # Object mentions in "what objects" should not invent labels absent from evidence.
        if "object" in q and "visible" in q:
            known = {
                (item.subject or "").lower()
                for item in packet.items
                if item.kind == "object" and item.claim_status != "UNCERTAIN" and item.confidence >= 0.42
            }
            # Soft check only for a few high-risk invented venue words.
            for invented in ("highway", "restaurant", "classroom", "office building"):
                if invented in a and invented not in " ".join(known) and invented not in (
                    " ".join(
                        (item.value or "").lower()
                        for item in packet.items
                        if item.kind == "environment"
                    )
                ):
                    return False

        return True

    @staticmethod
    def _extract_stated_count(answer: str) -> int | None:
        text = (answer or "").lower()
        words = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
        }
        for word, n in words.items():
            if re.search(rf"\b{word}\b", text):
                return n
        m = re.search(r"\b(\d+)\b", text)
        if m:
            return int(m.group(1))
        if "no people" in text or "no person" in text:
            return 0
        return None

    def _try_direct_answer_continued(
        self,
        packet: AssistantEvidencePacket,
        q: str,
        *,
        items: list[EvidenceItem],
    ) -> str:
        """Remainder of deterministic handlers after early presence/age gates."""
        # Wearing / jacket / clothing / clothing color (word-boundary cues only).
        if _CLOTHING_CUE_RE.search(q):
            answer = self._clothing_answer(packet, q)
            if answer:
                return answer

        # Color of a named object (car, horse, etc.) — never invent garment type.
        if "color" in q or "colour" in q:
            color_answer = self._color_of_object_answer(packet, q)
            if color_answer:
                return color_answer

        # Equipment (not bare "holding" — holding uses verified interaction relations).
        if any(tok in q for tok in ("equipment", "gear")) or (
            "ski" in q and any(tok in q for tok in ("pole", "equipment", "what", "using"))
        ) or ("using" in q and "holding" not in q and "leading" not in q):
            equip = self._equipment_answer(packet)
            if equip:
                return equip

        # Holding / leading / riding — verified INTERACTION relations only.
        if any(tok in q for tok in ("holding", "leading", "riding", "guiding")):
            held = self._interaction_answer(packet, q)
            if held:
                return held

        # Counts.
        if "how many" in q or "number of" in q or "count" in q:
            count_answer = self._count_answer(packet, q)
            if count_answer:
                return count_answer

        # Where is X — zone / spatial evidence first.
        if q.startswith("where is") or q.startswith("where's") or "where is the" in q:
            where = self._where_is_answer(packet, q)
            if where:
                return where

        # Activity / doing — select strongest verified activity, never packet order.
        person_activity_ask = any(tok in q for tok in ("doing", "activity", "action")) or (
            "happening" in q
            and any(tok in q for tok in ("person", "people", "they", "he", "she"))
        )
        if person_activity_ask:
            activity_answer = self._activity_answer(packet, q)
            if activity_answer:
                return activity_answer
            return (
                "I can't determine the person's exact activity "
                "from the available visual evidence."
            )
        if "happening" in q:
            activity_answer = self._activity_answer(packet, q)
            if activity_answer:
                return activity_answer
            # Otherwise fall through to environment / setting handlers.

        # Objects near / beside / next to a person (person-anchored only).
        # Do NOT route "near the fire/smoke" here — that invents person-centric layout.
        if any(
            phrase in q
            for phrase in (
                "near the person",
                "next to the person",
                "beside the person",
                "objects near",
                "objects are visible near",
                "positioned near the person",
                "positioned next to the person",
                "is the person near",
            )
        ) or (
            any(tok in q for tok in ("near", "beside", "next to", "close to"))
            and any(tok in q for tok in ("object", "objects", "what", "positioned"))
            and "person" in q
            and not any(h in q for h in ("fire", "smoke", "flame"))
        ):
            near_answer = self._near_objects_answer(packet, q)
            if near_answer:
                return near_answer

        # NOTE: remaining object/environment handlers.
        return self._try_direct_answer_tail(packet, q, items=items)

    def _try_direct_answer_tail(
        self,
        packet: AssistantEvidencePacket,
        q: str,
        *,
        items: list[EvidenceItem],
    ) -> str:
        if any(h in q for h in ("near the fire", "beside the fire", "near the smoke")):
            return (
                "The image does not provide enough verified spatial evidence "
                "to describe what is near the fire or smoke."
            )

        # Relative spatial ("where is X relative to Y").
        if "relative to" in q or ("where is" in q and any(tok in q for tok in ("near", "beside", "behind", "front"))):
            spatial = self._spatial_relation_answer(packet, q)
            if spatial:
                return spatial

        # Indoor setting kind.
        if "indoor" in q and any(tok in q for tok in ("space", "kind", "setting", "room", "type")):
            env_answer = self._indoor_setting_answer(packet)
            if env_answer:
                return env_answer

        # OCR / text.
        if any(tok in q for tok in ("text", "read", "sign", "writing", "says")):
            ocr = [item.value for item in packet.items if item.kind == "ocr" and item.value]
            if ocr:
                joined = ", ".join(f'"{t}"' for t in ocr[:3])
                return f"Readable text in the image includes {joined}."
            return "No reliable readable text was detected in the image."

        # Background / environment.
        if "background" in q or "environment" in q or "setting" in q:
            env = [
                item
                for item in packet.items
                if item.kind == "environment" and item.reliable
            ]
            bits = []
            for item in env:
                if item.predicate in {"setting", "scene_type", "weather", "indoor_outdoor"}:
                    bits.append(item.value.replace("_", " "))
                elif item.predicate == "evidence" and item.value.lower() not in {
                    b.lower() for b in bits
                }:
                    bits.append(item.value)
            if bits:
                return f"The setting appears to include {', '.join(bits[:4])}."

        _ = items
        return ""

    def _activity_answer(self, packet: AssistantEvidencePacket, q: str) -> str:
        """Answer activity questions by evidence-level priority.

        Priority:
        1. CONFIRMED_ACTIVITY
        2. SUPPORTED_ACTION
        3. Caption-explicit action evidence
        4. Unknown (empty → caller returns uncertainty)
        """
        q_l = (q or "").lower()
        people = ordered_people(packet)
        person = resolve_person_reference(q_l, packet)
        asks_plural = any(
            tok in q_l for tok in ("people", "they", "them", "everyone", "persons")
        )

        def _level(item: EvidenceItem) -> str:
            return (item.evidence_level or "").upper()

        confirmed = [
            item
            for item in packet.items
            if item.kind == "activity"
            and item.reliable
            and item.confidence >= 0.62
            and _level(item) == "CONFIRMED"
            and (item.claim_status or "").upper() != "UNCERTAIN"
        ]
        supported = [
            item
            for item in packet.items
            if item.kind == "activity"
            and item.reliable
            and item.confidence >= 0.58
            and _level(item) == "SUPPORTED"
            and (item.claim_status or "").upper() != "UNCERTAIN"
        ]

        scene_blob = " ".join(
            (item.value or "").lower()
            for item in packet.items
            if item.kind == "environment"
            and item.predicate in {"scene_type", "setting", "indoor_outdoor"}
        )
        object_blob = " ".join(
            (item.subject or "").lower()
            for item in packet.items
            if item.kind == "object" and item.claim_status != "UNCERTAIN"
        )

        def _score(item: EvidenceItem) -> tuple[int, int, int, float]:
            status = (item.claim_status or "").upper()
            observed = 1 if status == "OBSERVED" else 0
            linked = 0
            if person is not None:
                eid = (item.entity_id or "").lower()
                if eid == person.entity_id or (
                    item.object_index >= 0 and item.object_index == person.object_index
                ):
                    linked = 2
                elif eid.startswith("person"):
                    linked = 1
            elif (item.entity_id or "").lower().startswith("person"):
                linked = 1
            act = (item.value or "").lower()
            act_tokens = [
                t
                for t in re.findall(r"[a-z]{4,}", act)
                if t not in {"with", "from", "scene", "appears", "person"}
            ]
            scene_hit = 0
            if act_tokens and (
                any(t in object_blob for t in act_tokens)
                or any(t in scene_blob for t in act_tokens)
            ):
                scene_hit = 1
            if "tennis" in act and "racket" not in object_blob and "tennis" not in scene_blob:
                scene_hit = -1
            if "office" in act and "laptop" not in object_blob and "keyboard" not in object_blob:
                scene_hit = -1
            # Prefer action-grade verbs over mere possession when both are CONFIRMED.
            action_grade = 1
            if any(
                tok in act
                for tok in (
                    "riding",
                    "leading",
                    "guiding",
                    "pushing",
                    "skiing",
                    "skating",
                    "cycling",
                    "driving",
                )
            ):
                action_grade = 3
            elif any(tok in act for tok in ("holding", "carrying", "wearing", "looking")):
                action_grade = 0
            return (scene_hit, linked, action_grade, observed, item.confidence)

        def _pick(pool: list[EvidenceItem]) -> EvidenceItem | None:
            ranked = [c for c in pool if _score(c)[0] >= 0]
            if not ranked:
                return None
            ranked.sort(key=_score, reverse=True)
            return ranked[0]

        def _format(best: EvidenceItem, *, hedged: bool) -> str:
            value = best.value
            if asks_plural and len(people) >= 2:
                return f"The people appear to be {value}."
            if person is not None and len(people) >= 2:
                who = person_phrase(person, total=len(people))
                eid = (best.entity_id or "").lower()
                linked = eid == person.entity_id or (
                    best.object_index >= 0 and best.object_index == person.object_index
                )
                if linked and not hedged:
                    return f"{who} is {value}."
                return f"{who} appears to be {value}."
            if person is not None or len(people) == 1:
                who = person_phrase(
                    person if person is not None else people[0],
                    total=len(people),
                )
                if hedged:
                    return f"{who} appears to be {value}."
                return f"{who} is {value}."
            if hedged:
                return f"The main activity appears to be {value}."
            return f"The main activity appears to be {value}."

        # 1) CONFIRMED — never drop a CONFIRMED activity solely for weak scene_hit.
        def _pick_confirmed(pool: list[EvidenceItem]) -> EvidenceItem | None:
            if not pool:
                return None
            ranked = sorted(pool, key=_score, reverse=True)
            # Prefer linked person activities when a person is referenced.
            if person is not None:
                linked = [
                    c
                    for c in ranked
                    if (c.entity_id or "").lower() == person.entity_id
                    or (c.object_index >= 0 and c.object_index == person.object_index)
                ]
                if linked:
                    return linked[0]
            return ranked[0]

        best_confirmed = _pick_confirmed(confirmed)
        if best_confirmed is not None:
            if asks_plural and len(people) >= 2:
                by_person: dict[str, EvidenceItem] = {}
                for item in sorted(confirmed, key=_score, reverse=True):
                    eid = (item.entity_id or "").lower()
                    if eid.startswith("person") and eid not in by_person:
                        by_person[eid] = item
                if len(by_person) >= 2:
                    ordered = []
                    for p in people:
                        if p.entity_id in by_person:
                            ordered.append((p, by_person[p.entity_id]))
                    if len(ordered) >= 2:
                        a0 = ordered[0][1].value
                        a1 = ordered[1][1].value
                        if a0.lower() != a1.lower():
                            return f"One person is {a0}, while the other is {a1}."
                        return f"The people appear to be {a0}."
            # Singular "they" with one person (or plural wording with one actor).
            return _format(best_confirmed, hedged=False)

        # 2) SUPPORTED
        best_supported = _pick(supported)
        if best_supported is not None:
            return _format(best_supported, hedged=True)

        # 3) Caption-explicit action verbs (conservative extraction).
        caption_action = self._caption_explicit_action(packet, person=person, people=people)
        if caption_action:
            return caption_action

        # 4) Unknown — empty triggers caller uncertainty message.
        return ""

    def _caption_explicit_action(
        self,
        packet: AssistantEvidencePacket,
        *,
        person,
        people: list,
    ) -> str:
        """Use only clear action phrases already stated in the caption."""
        caption = (packet.canonical_caption_en or "").strip()
        if not caption:
            return ""
        lower = caption.lower()
        # Require an explicit progressive / activity cue — not mere location.
        patterns = [
            r"\b(?:is|are|appears to be|appears)\s+([a-z]+(?:ing)(?:\s+[a-z]+){0,4})",
            r"\bobserved activity:\s*([a-z ]{3,40})",
        ]
        match = None
        for pat in patterns:
            match = re.search(pat, lower)
            if match:
                break
        if not match:
            return ""
        action = " ".join(match.group(1).split()).strip(" .,")
        # Reject weak/location-only leftovers.
        if not action or action in {
            "standing",
            "sitting",
            "visible",
            "present",
            "in",
            "near",
        }:
            return ""
        if any(tok in action for tok in ("office work", "playing tennis", "driving", "shopping", "cooking")):
            # Never revive hallucinated captions as QA authority without packet support.
            return ""
        # Caption-only actions are not authoritative — require a matching verified activity.
        act_blob = " ".join(
            (item.value or "").lower()
            for item in packet.items
            if item.kind == "activity"
            and item.reliable
            and (item.evidence_level or "").upper() in {"CONFIRMED", "SUPPORTED"}
        )
        action_tokens = [
            t for t in re.findall(r"[a-z]{4,}", action) if t not in {"with", "from", "appears"}
        ]
        if not action_tokens or not any(t in act_blob for t in action_tokens):
            return ""
        who = person_phrase(
            person if person is not None else (people[0] if people else None),
            total=len(people),
        )
        if who.lower().startswith("the person") or who.lower().startswith("one person") or "person" in who.lower():
            return f"{who} appears to be {action}."
        return f"{who} appears to be {action}."

    def _fire_smoke_answer(self, packet: AssistantEvidencePacket, q: str) -> str:
        has_fire = False
        has_smoke = False
        for item in packet.items:
            lab = (item.subject or "").lower().strip()
            val = (item.value or "").lower()
            blob = f"{lab} {val} {item.predicate}"
            if item.kind == "object" and item.confidence >= 0.50:
                if any(tok in lab for tok in _FIRE_LABELS) or lab in _FIRE_LABELS:
                    has_fire = True
                if any(tok in lab for tok in _SMOKE_LABELS):
                    has_smoke = True
            if item.kind == "environment" and item.confidence >= 0.50:
                if any(tok in blob for tok in _FIRE_LABELS):
                    has_fire = True
                if "smoke" in blob:
                    has_smoke = True
        caption = (packet.canonical_caption_en or "").lower()
        if not has_fire and re.search(r"\b(fire|flame|campfire)\b", caption):
            has_fire = True
        if not has_smoke and re.search(r"\bsmoke\b", caption):
            has_smoke = True

        asks_fire = "fire" in q or "flame" in q
        asks_smoke = "smoke" in q
        if has_fire and has_smoke and asks_fire and asks_smoke:
            return "Yes. A fire and visible smoke can be seen in the scene."
        if has_fire and asks_smoke and not has_smoke:
            return "Yes. A fire is clearly visible. Smoke cannot be confirmed."
        if has_fire and asks_fire:
            return "Yes. A fire is clearly visible."
        if has_smoke and asks_smoke:
            return "Yes. Smoke is visible in the scene."
        if asks_fire or asks_smoke:
            return "I can't reliably confirm fire or smoke from the available visual evidence."
        return ""

    def _animals_answer(self, packet: AssistantEvidencePacket) -> str:
        counts: dict[str, int] = {}
        for item in packet.items:
            if item.kind != "object" or item.confidence < _DETECT_CONF_MIN:
                continue
            lab = item.subject.lower().strip()
            if lab in _ANIMALS:
                counts[lab] = counts.get(lab, 0) + 1
        if not counts:
            return (
                "I can't reliably determine which animals are visible "
                "from the available visual evidence."
            )
        parts: list[str] = []
        for lab, n in counts.items():
            if n == 1:
                parts.append(f"a {lab}")
            else:
                plural = lab if lab.endswith("s") else f"{lab}s"
                parts.append(f"{n} {plural}")
        if len(parts) == 1:
            if counts[next(iter(counts))] == 1:
                return f"A {next(iter(counts))} is visible in the scene."
            # Prefer natural "Two horses…" over "2 horses…".
            raw = parts[0]
            spelled = (
                raw.replace("2 ", "Two ")
                .replace("3 ", "Three ")
                .replace("4 ", "Four ")
            )
            if spelled[0].isdigit():
                spelled = spelled[0].upper() + spelled[1:]
            else:
                spelled = spelled[0].upper() + spelled[1:]
            return f"{spelled} are visible in the scene."
        if len(parts) == 2:
            return f"Visible animals include {parts[0]} and {parts[1]}."
        return f"Visible animals include {', '.join(parts[:-1])}, and {parts[-1]}."

    def _strip_appended_caption(
        self,
        answer: str,
        packet: AssistantEvidencePacket,
    ) -> str:
        """Keep only the direct answer — never append the scene caption."""
        text = (answer or "").strip()
        if not text:
            return text
        caption = (packet.canonical_caption_en or "").strip()
        if caption and len(caption) >= 24:
            # Exact / near-exact caption pasted after the answer.
            if caption in text and not text.startswith(caption[:40]):
                text = text.replace(caption, "").strip()
            # Case-insensitive containment of a long caption span.
            cap_l = caption.lower()
            text_l = text.lower()
            if cap_l in text_l and not text_l.startswith(cap_l[: min(40, len(cap_l))]):
                idx = text_l.find(cap_l)
                if idx > 0:
                    text = text[:idx].strip()
        # Multi-paragraph: keep the first answer paragraph when later ones look
        # like a restated scene caption (person/clothing/leading prose).
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(parts) >= 2:
            first = parts[0]
            rest = " ".join(parts[1:]).lower()
            caption_like = any(
                tok in rest
                for tok in (
                    "wearing a",
                    "leading",
                    "riding",
                    "khaki-colored",
                    "beside a",
                    "outdoors",
                    "in the scene,",
                    "observed activity",
                )
            )
            if caption_like and len(first.split()) <= 40:
                text = first
        # Drop internal pipeline labels if somehow leaked into an answer.
        text = re.sub(r"(?i)\bobserved activity\s*:\s*", "", text).strip()
        return text

    def _safe_person_clothing_color(
        self,
        packet: AssistantEvidencePacket,
        person=None,
        *,
        question: str = "",
        predicates: tuple[str, ...] | None = None,
    ) -> tuple[str, float, str] | None:
        """Return (color, confidence, strength) for one person when verified.

        strength: 'high' | 'medium'
        """
        from language.assistant.entity_indexing import IndexedPerson

        preds = predicates or self._clothing_color_predicates_for_question(question)
        target: IndexedPerson | None = person
        people = ordered_people(packet)
        if target is None and len(people) == 1:
            target = people[0]
        # Never pick an arbitrary person with a color when the referent is unset —
        # that steals another person's clothing in multi-person scenes.
        if target is None:
            return None

        color = find_person_attribute(
            packet,
            target,
            predicates=preds,
            require_reliable=True,
            min_confidence=0.55,
        )
        if color is None:
            # Caption fallback only for the primary/unindexed ask — never for
            # second/third person (prevents cross-person color transfer).
            if person is None or target.ordinal == 1:
                if "pants" not in " ".join(preds) and "shoes" not in " ".join(preds):
                    caption_color = self._clothing_color_from_caption(packet)
                    if caption_color and self._caption_clothing_color_ok_for_person(
                        packet, target, caption_color
                    ):
                        return caption_color, 0.80, "high"
            return None

        value = (color.value or "").strip().lower()
        status = (color.claim_status or "").upper()
        conf = float(color.confidence)

        if value in _AMBIGUOUS_CLOTHING_COLORS:
            caption_color = ""
            if person is None or target.ordinal == 1:
                caption_color = self._clothing_color_from_caption(packet)
            # Prefer a clearer sibling OBSERVED shirt/clothing color when present.
            clearer = self._clearer_observed_clothing_color(packet, target, exclude=value)
            if clearer:
                return clearer
            # Uniform muted outdoor colors across shirt/pants/dominant often mean
            # grass/background bleed — refuse rather than assert khaki/olive.
            if value in {"olive", "khaki", "tan", "beige", "cream"} and self._likely_background_bleed(
                packet, target, value
            ):
                # After bleed refusal, caption may salvage ONLY when it is not
                # another person's clothing color (entity-bound gate).
                if (
                    caption_color
                    and (person is None or target.ordinal == 1)
                    and self._caption_clothing_color_ok_for_person(
                        packet, target, caption_color
                    )
                ):
                    return caption_color, 0.80, "high"
                return None
            # Caption may override ambiguous fashion labels only when the color
            # is not bound to a different person in the same scene.
            if (
                caption_color
                and caption_color != value
                and (person is None or target.ordinal == 1)
                and self._caption_clothing_color_ok_for_person(packet, target, caption_color)
            ):
                return caption_color, 0.80, "high"
            # Trust OBSERVED clothing analyzer colors at solid confidence
            # only when the caption does not contradict with a safer color.
            if status == "OBSERVED" and conf >= 0.62 and not caption_color:
                return value, conf, "high" if conf >= 0.75 else "medium"
            if conf >= 0.90:
                return value, conf, "high"
            if caption_color and caption_color == value:
                return value, max(conf, 0.80), "high"
            # Do not guess ambiguous fashion labels from weak/inferred evidence.
            return None

        if conf < 0.55:
            return None
        strength = "high" if conf >= 0.62 else "medium"
        return value, conf, strength

    def _likely_background_bleed(
        self,
        packet: AssistantEvidencePacket,
        person,
        value: str,
    ) -> bool:
        """True when muted color repeats across clothing+body attrs (grass/horse bleed)."""
        muted = value.lower().strip()
        hits = 0
        checked = 0
        for pred in ("shirt_color", "clothing_color", "pants_color", "dominant_color", "color"):
            item = find_person_attribute(
                packet,
                person,
                predicates=(pred,),
                require_reliable=False,
                min_confidence=0.50,
            )
            if item is None:
                continue
            checked += 1
            if (item.value or "").strip().lower() == muted:
                hits += 1
        return checked >= 3 and hits >= 3

    def _clearer_observed_clothing_color(
        self,
        packet: AssistantEvidencePacket,
        person,
        *,
        exclude: str,
    ) -> tuple[str, float, str] | None:
        """If ambiguous khaki/olive/tan was selected, prefer a clear OBSERVED shirt color."""
        clear = {
            "black",
            "white",
            "red",
            "blue",
            "green",
            "gray",
            "grey",
            "navy",
            "brown",
            "yellow",
            "pink",
            "purple",
            "orange",
            "charcoal",
            "maroon",
            "cyan",
            "light blue",
            "sky blue",
            "navy blue",
            "royal blue",
            "burgundy",
        }
        for pred in ("shirt_color", "clothing_color"):
            item = find_person_attribute(
                packet,
                person,
                predicates=(pred,),
                require_reliable=True,
                min_confidence=0.62,
            )
            if item is None:
                continue
            val = (item.value or "").strip().lower()
            if not val or val == exclude or val in _AMBIGUOUS_CLOTHING_COLORS:
                continue
            if (item.claim_status or "").upper() != "OBSERVED":
                continue
            # Match clear palette or multi-word blues already listed.
            if val in clear or any(c in val for c in ("blue", "red", "black", "white", "green")):
                return val, float(item.confidence), "high" if item.confidence >= 0.75 else "medium"
        return None

    def _clothing_color_predicates_for_question(self, q: str) -> tuple[str, ...]:
        """Keep shirt/pants/shoes as separate attributes — never collapse all clothing."""
        q_l = (q or "").lower()
        if any(tok in q_l for tok in ("shoe", "shoes", "sneaker", "footwear")):
            return ("shoes_color",)
        if any(tok in q_l for tok in ("pants", "trousers", "jeans", "shorts", "skirt")):
            return ("pants_color",)
        if any(
            tok in q_l
            for tok in (
                "shirt",
                "t-shirt",
                "tee",
                "jersey",
                "blouse",
                "top",
                "hoodie",
                "sweater",
                "jacket",
                "coat",
            )
        ):
            # Prefer entity-bound shirt/top over aggregate clothing_color.
            return ("shirt_color", "clothing_color")
        # Generic clothing color ask — never use person.color / dominant_color.
        return ("shirt_color", "clothing_color")
    def _clothing_color_from_caption(self, packet: AssistantEvidencePacket) -> str:
        caption = (packet.canonical_caption_en or "").lower()
        if not caption:
            return ""
        garment = (
            r"(sweatshirt|hoodie|jacket|coat|shirt|t-shirt|tee|clothing|sweater|"
            r"top|blouse|jersey)"
        )
        for color in _SAFE_CAPTION_CLOTHING_COLORS:
            if re.search(rf"\b{re.escape(color)}\s+{garment}\b", caption):
                return color
            if re.search(rf"\bin a {re.escape(color)}\b", caption):
                return color
        return ""

    def _caption_clothing_color_ok_for_person(
        self,
        packet: AssistantEvidencePacket,
        person,
        caption_color: str,
    ) -> bool:
        """True when caption color is safe to bind to this person.

        Blocks cross-person bleed: if another verified person already owns this
        color as clothing/body color, do not answer it for a different person.
        """
        color = (caption_color or "").strip().lower()
        if not color:
            return False
        # Caption color that already appears on THIS person is always ok.
        for pred in ("shirt_color", "clothing_color", "pants_color"):
            item = find_person_attribute(
                packet,
                person,
                predicates=(pred,),
                require_reliable=False,
                min_confidence=0.50,
            )
            if item is None:
                continue
            val = (item.value or "").strip().lower()
            if val == color or color in val or val in color:
                return True
        people = ordered_people(packet)
        if len(people) <= 1:
            return True
        # Multi-person: refuse if another person owns this color.
        for other in people:
            if other.entity_id == getattr(person, "entity_id", None):
                continue
            for pred in ("shirt_color", "clothing_color", "color", "dominant_color"):
                item = find_person_attribute(
                    packet,
                    other,
                    predicates=(pred,),
                    require_reliable=False,
                    min_confidence=0.50,
                )
                if item is None:
                    continue
                val = (item.value or "").strip().lower()
                if val == color or color in val:
                    return False
        return True

    def _verified_garment_type(self, packet: AssistantEvidencePacket, person=None) -> str:
        people = ordered_people(packet)
        target = person
        if target is None and len(people) == 1:
            target = people[0]
        if target is None:
            ctype = find_attribute(
                packet,
                predicate="clothing_type",
                subject_tokens=tuple(_PERSON),
                require_reliable=True,
            )
            jacket_flag = find_attribute(
                packet,
                predicate="jacket",
                subject_tokens=tuple(_PERSON),
                require_reliable=False,
            )
        else:
            ctype = find_person_attribute(
                packet,
                target,
                predicates=("clothing_type",),
                require_reliable=True,
            )
            jacket_flag = find_person_attribute(
                packet,
                target,
                predicates=("jacket",),
                require_reliable=False,
            )
        if (
            ctype
            and ctype.value.lower() not in {"unknown", "unlikely", "casual"}
            and ctype.confidence >= 0.70
        ):
            garment = ctype.value.replace("_", " ").strip().lower()
            # Generic shirt/t-shirt labels are frequently guessed from color alone.
            if garment in {"t-shirt", "tshirt", "tee", "shirt", "top"} and ctype.confidence < 0.88:
                return ""
            return garment
        if jacket_flag and jacket_flag.value.lower() == "likely" and jacket_flag.confidence >= 0.70:
            return "jacket"
        return ""

    def _multi_person_clothing_referent_ambiguous(
        self,
        packet: AssistantEvidencePacket,
        q: str,
        person,
    ) -> bool:
        """True when clothing color cannot be bound to one person safely."""
        people = ordered_people(packet)
        if len(people) <= 1:
            return False
        q_l = (q or "").lower()
        # Explicit ordinal / side / depth — referent is resolved by the question.
        if re.search(
            r"\b(first|second|third|fourth|1st|2nd|3rd|4th|left|right|background)\b",
            q_l,
        ) and "person" in q_l:
            return False
        # Unique CONFIRMED activity actor — safe to bind clothing to that actor.
        confirmed: set[str] = set()
        for item in packet.items:
            if item.kind != "activity":
                continue
            if (item.evidence_level or "").upper() != "CONFIRMED":
                continue
            eid = (item.entity_id or "").lower()
            if eid.startswith("person"):
                confirmed.add(eid)
        if len(confirmed) == 1 and person is not None and person.entity_id in confirmed:
            return False
        return True

    def _clothing_answer(self, packet: AssistantEvidencePacket, q: str = "") -> str:
        q_l = (q or "").lower()
        people = ordered_people(packet)
        person = resolve_person_reference(q_l, packet)
        asks_indexed = bool(
            re.search(
                r"\b(first|second|third|1st|2nd|3rd|left|right|background)\b",
                q_l,
            )
            and "person" in q_l
        )
        if asks_indexed and person is None:
            return (
                "I can't reliably match that person index "
                "from the available visual evidence."
            )

        # Multi-person + unindexed "the person" clothing ask: never steal colors.
        if self._multi_person_clothing_referent_ambiguous(packet, q_l, person):
            if any(
                tok in q_l
                for tok in (
                    "color",
                    "colour",
                    "clothing",
                    "wearing",
                    "wear",
                    "shirt",
                    "pants",
                    "jacket",
                )
            ):
                return (
                    "I can't determine which person's clothing you mean "
                    "from the available visual evidence."
                )

        color_pair = self._safe_person_clothing_color(packet, person, question=q_l)
        garment = self._verified_garment_type(packet, person)
        caption = (packet.canonical_caption_en or "").lower()
        who = person_phrase(person, total=len(people))
        preds = self._clothing_color_predicates_for_question(q_l)
        asks_pants = any(tok in q_l for tok in ("pants", "trousers", "jeans", "shorts", "skirt"))
        asks_shoes = any(tok in q_l for tok in ("shoe", "shoes", "sneaker", "footwear"))
        asks_shirt = any(
            tok in q_l
            for tok in (
                "shirt",
                "t-shirt",
                "tee",
                "jersey",
                "blouse",
                "top",
                "hoodie",
                "sweater",
                "jacket",
                "coat",
            )
        )

        # Prefer safe color-only wording for color questions.
        if color_pair and ("color" in q_l or "colour" in q_l):
            color, _conf, strength = color_pair
            if asks_pants:
                if strength == "medium":
                    return f"{who} appears to be wearing {color} pants."
                return f"{who} is wearing {color} pants."
            if asks_shoes:
                if strength == "medium":
                    return f"{who}'s shoes appear {color}."
                return f"{who}'s shoes are {color}."
            for named in ("sweatshirt", "hoodie", "jacket", "coat", "sweater"):
                if named in caption and color in caption and (person is None or person.ordinal == 1):
                    return f"{who} is wearing a {color} {named}."
            if strength == "medium":
                return f"{who} appears to be wearing {color} clothing."
            if asks_shirt:
                return f"{who} is wearing a {color} shirt."
            if garment and garment not in {"t-shirt", "tshirt", "tee", "shirt", "top", "unknown"}:
                return f"{who} is wearing a {color} {garment}."
            return f"{who} is wearing {color} clothing."

        # Color question with no reliable entity-bound color: refuse (do not
        # answer with a bare garment type from clothing_type alone).
        if ("color" in q_l or "colour" in q_l) and color_pair is None:
            return (
                "I can't reliably determine the clothing color "
                "from the available visual evidence."
            )

        for named in ("sweatshirt", "hoodie", "jacket", "coat", "sweater"):
            if named in caption and (not color_pair or color_pair[0] in caption):
                # Multi-person + no entity color: do not bind a caption garment
                # that may describe a different person.
                if color_pair is None and len(people) > 1:
                    continue
                if color_pair and (person is None or person.ordinal == 1):
                    return f"{who} is wearing a {color_pair[0]} {named}."
                if person is None or person.ordinal == 1:
                    return f"{who} is wearing a {named}."
        if color_pair and garment and garment not in {"t-shirt", "tshirt", "tee", "shirt", "top"}:
            return f"{who} is wearing a {color_pair[0]} {garment}."
        if color_pair:
            color, _conf, strength = color_pair
            if strength == "medium":
                return f"{who} appears to be wearing {color} clothing."
            return f"{who} is wearing {color} clothing."
        if garment and garment not in {"t-shirt", "tshirt", "tee", "shirt", "top"}:
            return f"{who} is wearing a {garment}."
        if any(tok in q_l for tok in ("clothing", "wearing", "wear", "shirt", "color", "colour")):
            return (
                "I can't reliably determine the clothing color "
                "from the available visual evidence."
            )
        return ""

    def _presence_answer(self, packet: AssistantEvidencePacket, q: str) -> str:
        """Answer yes/no object presence from verified detections."""
        labels = sorted(
            {
                item.subject.lower().strip()
                for item in packet.items
                if item.kind == "object" and item.confidence >= _DETECT_CONF_MIN
            },
            key=len,
            reverse=True,
        )
        for label in labels:
            if label in _PERSON:
                continue
            if not re.search(rf"\b{re.escape(label)}\b", q):
                # Multi-word / dining table variants
                if label == "dining table" and "table" in q and "dining" in q:
                    pass
                elif label.replace(" ", "") in q.replace(" ", ""):
                    pass
                else:
                    continue
            n = sum(
                1
                for item in packet.items
                if item.kind == "object"
                and item.subject.lower() == label
                and item.confidence >= _DETECT_CONF_MIN
            )
            art = "an" if label[:1] in "aeiou" else "a"
            if n >= 1:
                return f"Yes, {art} {label} is visible."
        # Explicit negative when a clear noun is asked and missing.
        m = re.search(r"\b(?:a|an|any|the)\s+([a-z][a-z\s]{1,40}?)\b(?:\s+visible|\s+in\b|\?|$)", q)
        if m:
            asked = m.group(1).strip()
            asked = re.sub(r"\b(visible|in the scene|in the image)\b", "", asked).strip()
            if asked and asked not in {"person", "people"}:
                return f"No reliable evidence shows a {asked} in the image."
        return ""

    def _where_is_answer(self, packet: AssistantEvidencePacket, q: str) -> str:
        """Locate an object via zone strings and verified spatial relations."""
        labels = sorted(
            {
                item.subject.lower().strip()
                for item in packet.items
                if item.kind == "object" and item.confidence >= _DETECT_CONF_MIN
            },
            key=len,
            reverse=True,
        )
        target = ""
        for label in labels:
            if re.search(rf"\b{re.escape(label)}\b", q):
                target = label
                break
        if not target and "table" in q:
            target = "dining table" if "dining table" in labels else (
                "table" if "table" in labels else ""
            )
        if not target:
            return ""

        # Prefer verified spatial relation involving the target.
        for item in packet.items:
            if item.kind != "relation" or item.confidence < 0.55:
                continue
            subj = (item.subject or "").lower()
            obj = (item.value or "").lower()
            pred = item.predicate.lower().replace("_", " ")
            if target in {subj, obj} or (target == "dining table" and "table" in f"{subj} {obj}"):
                other = obj if target in subj or (target == "dining table" and "table" in subj) else subj
                if other:
                    return f"The {target} is {pred} the {other}."

        # Fall back to zone from objects listing.
        zone = ""
        for line in packet.objects:
            low = line.lower()
            if target.replace(" ", "_") in low or f":{target}" in low or target in low:
                zm = re.search(r"zone=([^,\s)]+)", low)
                if zm:
                    zone = zm.group(1).replace("-", " ")
                    break
        if zone:
            return f"The {target} is in the {zone} of the scene."
        # Presence-only fallback.
        present = any(
            item.kind == "object"
            and item.subject.lower() == target
            and item.confidence >= _DETECT_CONF_MIN
            for item in packet.items
        )
        if present:
            return f"A {target} is visible in the scene."
        return ""

    def _color_of_object_answer(self, packet: AssistantEvidencePacket, q: str) -> str:
        q_l = (q or "").lower()
        # Clothing / person garment questions — never answer from object crops.
        if any(
            tok in q_l
            for tok in (
                "clothing",
                "wearing",
                "shirt",
                "t-shirt",
                "jacket",
                "outfit",
                "pants",
                "jeans",
                "shoes",
                "jersey",
            )
        ) or (
            "person" in q_l
            and any(tok in q_l for tok in ("color", "colour"))
        ):
            clothed = self._clothing_answer(packet, q)
            if clothed:
                return clothed

        labels = sorted(
            _VEHICLE | _PERSON | _ANIMALS | _OBJECT_COLOR_LABELS,
            key=len,
            reverse=True,
        )
        for label in labels:
            if label == "ball" and "sports ball" in q_l:
                continue  # prefer full "sports ball" match
            if not re.search(rf"\b{re.escape(label)}\b", q_l):
                continue

            if label in _PERSON:
                person = resolve_person_reference(q_l, packet)
                # Always use entity-bound clothing-safe color path (never raw khaki
                # from ambiguous aggregates or cross-entity caption bleed).
                safe = self._safe_person_clothing_color(packet, person, question=q_l)
                if safe is None:
                    return (
                        "I can't reliably determine the clothing color "
                        "from the available visual evidence."
                    )
                value, _conf, strength = safe
                who = person_phrase(person, total=len(ordered_people(packet)))
                if strength == "medium":
                    return f"{who} appears to be wearing {value} clothing."
                return f"{who} is wearing {value} clothing."

            # Object/animal color: bind strictly to that entity's own attributes.
            item = self._entity_bound_color_item(packet, label)
            if item is None:
                return (
                    f"I can't reliably determine the color of the {label} "
                    f"from the available visual evidence."
                )
            value = item.value.strip().lower()
            status = (item.claim_status or "").upper()
            label_l = label.lower()
            # Prefer OBSERVED entity colors for bleed-prone small outdoor objects.
            if label_l in {
                "sports ball",
                "ball",
                "frisbee",
                "bicycle",
                "motorcycle",
                "bike",
            } and status and status not in {"OBSERVED", ""} and item.confidence < 0.80:
                return (
                    f"I can't reliably determine the color of the {label} "
                    f"from the available visual evidence."
                )
            if self._unreliable_object_scene_bleed(label, value):
                return (
                    f"I can't reliably determine the color of the {label} "
                    f"from the available visual evidence."
                )
            if self._color_matches_other_entity_clothing(packet, label, value):
                return (
                    f"I can't reliably determine the color of the {label} "
                    f"from the available visual evidence."
                )
            if label in _ANIMALS and value in {"olive", "khaki", "burgundy", "navy", "maroon"}:
                from language.refinement.caption_sanity import normalize_animal_coat_color

                value = normalize_animal_coat_color(value)
            return f"The {label} appears {value}."
        return ""

    def _unreliable_object_scene_bleed(self, label: str, color: str) -> bool:
        """True when the color is a common background/ground bleed for this object."""
        value = (color or "").strip().lower()
        label_l = (label or "").lower()
        if not value:
            return True
        # Sports balls are almost never beige/tan/olive — those are ground/grass crops.
        if label_l in {"sports ball", "ball", "frisbee"}:
            return value in {
                "beige",
                "tan",
                "cream",
                "khaki",
                "olive",
                "sand",
                "brown",
            }
        # Bicycles/motorcycles: green/olive often come from grass behind the frame.
        if label_l in {"bicycle", "motorcycle", "bike"}:
            return value in {
                "olive",
                "khaki",
                "beige",
                "tan",
                "cream",
                "dark green",
                "green",
            }
        return False
    def _entity_bound_color_item(
        self,
        packet: AssistantEvidencePacket,
        label: str,
    ) -> EvidenceItem | None:
        """Return dominant/color attribute for the requested entity only."""
        label_l = (label or "").lower().strip()
        if not label_l:
            return None
        label_key = label_l.replace(" ", "_")
        aliases = {label_l, label_key}
        if label_l == "sports ball":
            aliases.add("ball")
        best: EvidenceItem | None = None
        for item in packet.items:
            if item.kind != "attribute":
                continue
            pred = item.predicate.lower()
            if pred in _CLOTHING_COLOR_PREDS:
                continue  # never transfer clothing color onto objects
            if pred not in {"dominant_color", "color"}:
                continue
            if not item.reliable or item.confidence < _ATTR_CONF_MIN:
                continue
            eid = (item.entity_id or "").lower()
            subj = (item.subject or "").lower().replace("_", " ")
            # Reject person entities for non-person labels.
            if eid.startswith("person") or subj in _PERSON:
                continue
            eid_norm = eid.replace("_", " ")
            matched = False
            if eid.startswith(label_key) or eid_norm.startswith(label_l):
                matched = True
            elif subj == label_l or subj in aliases:
                matched = True
            elif label_l == "sports ball" and (
                eid.startswith("sports_ball") or "sports ball" in eid_norm
            ):
                matched = True
            if not matched:
                continue
            if best is None or item.confidence > best.confidence:
                best = item
        return best

    def _color_matches_other_entity_clothing(
        self,
        packet: AssistantEvidencePacket,
        label: str,
        color: str,
    ) -> bool:
        """Detect likely clothing→object color bleed for small/ambiguous objects."""
        value = (color or "").strip().lower()
        if not value or value in {"unknown", "unclear"}:
            return False
        # Only guard objects that commonly inherit nearby clothing/scene crop colors.
        label_l = (label or "").lower()
        if label_l not in {
            "sports ball",
            "ball",
            "frisbee",
            "tennis racket",
            "baseball bat",
            "kite",
            "bottle",
            "cup",
            "wine glass",
            "bicycle",
            "motorcycle",
            "bike",
        }:
            return False
        # Fashion / camouflage / grass hues are the typical bleed victims.
        if value not in _AMBIGUOUS_CLOTHING_COLORS and value not in {
            "olive",
            "khaki",
            "beige",
            "tan",
            "cream",
            "maroon",
            "burgundy",
            "navy",
            "green",
            "dark green",
            "brown",
        }:
            return False
        label_key = label_l.replace(" ", "_")
        for item in packet.items:
            if item.kind != "attribute":
                continue
            pred = item.predicate.lower()
            if pred not in _CLOTHING_COLOR_PREDS and pred not in {
                "color",
                "dominant_color",
            }:
                continue
            eid = (item.entity_id or "").lower()
            if eid.startswith(label_key) or eid.startswith("sports_ball"):
                continue
            # Person / other-entity colors must not become this object's color.
            subj = (item.subject or "").lower()
            if not (
                eid.startswith("person")
                or subj in _PERSON
                or pred in _CLOTHING_COLOR_PREDS
            ):
                continue
            other = (item.value or "").strip().lower()
            if other == value and item.confidence >= 0.55:
                return True
            # Near-match greens (dark green ↔ green ↔ olive).
            greenish = {"green", "dark green", "olive", "olive green"}
            if value in greenish and other in greenish and item.confidence >= 0.55:
                return True
        return False
    def _interaction_answer(self, packet: AssistantEvidencePacket, q: str) -> str:
        wanted: list[str] = []
        if "holding" in q:
            wanted.append("holding")
        if "leading" in q or "guiding" in q:
            wanted.extend(["leading", "guiding"])
        if "riding" in q:
            wanted.append("riding")
        if not wanted:
            wanted = list(_INTERACTION_PREDICATES)

        matches: list[tuple[str, str]] = []
        for item in packet.items:
            if item.kind != "relation":
                continue
            pred = item.predicate.lower()
            if pred not in wanted:
                continue
            if item.confidence < 0.68 or item.claim_status == "UNCERTAIN":
                continue
            kind = (item.relation_kind or "").upper()
            if kind and kind not in {"INTERACTION", ""}:
                continue
            obj = (item.value or "").strip().lower()
            if obj:
                matches.append((pred, obj))

        if re.search(r"\b(is|are|does|do)\b", q) and any(
            tok in q for tok in ("holding", "leading", "riding")
        ):
            target = ""
            for animal in _ANIMALS:
                if animal in q:
                    target = animal
                    break
            if target:
                ok = any(obj == target or target in obj for _, obj in matches)
                if ok:
                    verb = matches[0][0].replace("_", " ")
                    return f"Yes. The person is {verb} a {target}."
                # Holding ≠ leading/riding — surface the verified verb when available.
                related: list[str] = []
                for item in packet.items:
                    if item.kind != "relation":
                        continue
                    pred = item.predicate.lower()
                    if pred not in {"leading", "guiding", "riding", "holding"}:
                        continue
                    if item.confidence < 0.68 or item.claim_status == "UNCERTAIN":
                        continue
                    kind = (item.relation_kind or "").upper()
                    if kind and kind not in {"INTERACTION", ""}:
                        continue
                    obj = (item.value or "").strip().lower()
                    if obj == target or target in obj:
                        related.append(pred)
                if "holding" in q and related and "holding" not in related:
                    verb = related[0].replace("_", " ")
                    return (
                        f"The person is {verb} the {target}, but holding "
                        f"the {target} itself cannot be confirmed."
                    )
                return (
                    "I can't reliably confirm that interaction "
                    "from the available visual evidence."
                )

        if not matches:
            return (
                "The image does not provide enough verified evidence "
                "to determine that interaction."
            )
        pred, obj = matches[0]
        return f"The person is {pred.replace('_', ' ')} a {obj}."

    def _holding_answer(self, packet: AssistantEvidencePacket) -> str:
        """Backward-compatible holding helper."""
        return self._interaction_answer(packet, "what is the person holding")

    def _equipment_answer(self, packet: AssistantEvidencePacket) -> str:
        found: list[str] = []
        for item in packet.items:
            if item.kind != "object" or item.confidence < 0.55:
                continue
            label = item.subject.lower()
            if any(tok in label for tok in _EQUIPMENT):
                if label not in found:
                    found.append(label)
        if not found:
            # Relations like person using skis
            for item in packet.items:
                if item.kind == "relation" and item.predicate.lower() in {"using", "holding", "carrying"}:
                    if any(tok in item.value.lower() for tok in _EQUIPMENT):
                        if item.value.lower() not in found:
                            found.append(item.value.lower())
        if not found:
            return ""
        if len(found) == 1:
            return f"Visible equipment includes {found[0]}."
        return f"Visible equipment includes {', '.join(found[:-1])}, and {found[-1]}."

    def _count_answer(self, packet: AssistantEvidencePacket, q: str) -> str:
        def _count(labels: set[str]) -> int:
            # Match caption: only narrative-safe / non-UNCERTAIN entities.
            return sum(
                1
                for item in packet.items
                if item.kind == "object"
                and item.subject.lower() in labels
                and item.confidence >= _DETECT_CONF_MIN
                and (item.claim_status or "").upper() != "UNCERTAIN"
            )

        def _num_word(n: int) -> str:
            words = {
                0: "zero",
                1: "one",
                2: "two",
                3: "three",
                4: "four",
                5: "five",
                6: "six",
                7: "seven",
                8: "eight",
                9: "nine",
                10: "ten",
                11: "eleven",
                12: "twelve",
            }
            return words.get(n, str(n))

        if "people" in q or "person" in q or "persons" in q:
            # Single source of truth: ordered narrative-safe people index.
            n = len(ordered_people(packet))
            if n == 0:
                return "No people are clearly visible."
            if n == 1:
                return "There is one person."
            return f"There are {_num_word(n)} people."
        if any(tok in q for tok in ("horse", "horses")):
            n = _count({"horse"})
            if n == 0:
                return "No horses are clearly visible."
            if n == 1:
                return "One horse is visible."
            return f"{_num_word(n).capitalize()} horses are visible."
        if any(tok in q for tok in ("vehicle", "car", "cars", "bus", "truck")):
            n = _count(_VEHICLE)
            if n == 0:
                return "No vehicles are clearly visible."
            if n == 1:
                return "One vehicle is visible."
            return f"{_num_word(n).capitalize()} vehicles are visible."
        # Count a named object class ("how many chairs").
        for label in sorted(
            {
                item.subject.lower()
                for item in packet.items
                if item.kind == "object" and (item.claim_status or "").upper() != "UNCERTAIN"
            },
            key=len,
            reverse=True,
        ):
            if label in _PERSON:
                continue
            if label in q or (label == "dining table" and "table" in q):
                n = _count({label})
                if not n:
                    continue
                plural = label if label.endswith("s") else f"{label}s"
                if n == 1:
                    return f"There is one {label}."
                return f"There are {_num_word(n)} {plural}."
        return ""

    def _near_objects_answer(self, packet: AssistantEvidencePacket, q: str) -> str:
        """Answer proximity from verified SPATIAL relations only.

        near ≠ holding. Never invent proximity from "any high-confidence object".
        Interaction relations answer separately as interactions, not as 'near'.
        """
        spatial_preds = {
            "near",
            "next_to",
            "beside",
            "left_of",
            "right_of",
            "above",
            "below",
            "behind",
            "in_front_of",
            "inside",
            "on",
            "overlapping",
        }
        related: list[str] = []
        for item in packet.items:
            if item.kind != "relation" or item.confidence < 0.68:
                continue
            pred = item.predicate.lower()
            kind = (item.relation_kind or "").upper()
            # Prefer explicit SPATIAL kind; allow known spatial predicates when kind empty (legacy).
            if kind and kind != "SPATIAL":
                continue
            if pred not in spatial_preds:
                continue
            subj = item.subject.lower()
            obj = (item.value or "").lower()
            if subj in _PERSON and obj and obj not in _PERSON:
                if obj not in related:
                    related.append(obj)
            elif obj in _PERSON and subj not in _PERSON:
                if subj not in related:
                    related.append(subj)
            elif item.entity_id.startswith("person") and obj and obj not in _PERSON:
                if obj not in related:
                    related.append(obj)
        if not related:
            return (
                "The image does not provide enough verified spatial evidence "
                "to list objects near the person."
            )
        for lab in related:
            if lab in q:
                return f"The person is near a {lab}."
        top = related[:4]
        if len(top) == 1:
            return f"The person is near a {top[0]}."
        if len(top) == 2:
            return f"The person is near a {top[0]} and a {top[1]}."
        return (
            f"The person is near a {top[0]}, a {top[1]}, and a {top[2]}"
            + (f", along with a {top[3]}" if len(top) > 3 else "")
            + "."
        )

    def _spatial_relation_answer(self, packet: AssistantEvidencePacket, q: str) -> str:
        objects = [
            item.subject.lower()
            for item in packet.items
            if item.kind == "object" and item.confidence >= 0.50
        ]
        for item in packet.items:
            if item.kind != "relation" or item.confidence < 0.55:
                continue
            subj = item.subject.lower()
            obj = (item.value or "").lower()
            pred = item.predicate.lower().replace("_", " ")
            if subj in q and obj and (obj in q or obj in objects or "person" in q):
                return f"The {subj} is {pred} the {obj}."
            if obj in q and subj and (subj in q or "person" in q):
                return f"The {subj} is {pred} the {obj}."
        # Dining table / person fallback from near-objects.
        if "dining table" in q or "table" in q:
            near = self._near_objects_answer(packet, "near the person dining table")
            if near and ("dining table" in near.lower() or "table" in near.lower()):
                return "The dining table is near the person."
        return ""

    def _indoor_setting_answer(self, packet: AssistantEvidencePacket) -> str:
        setting = ""
        indoor = ""
        for item in packet.items:
            if item.kind != "environment":
                continue
            if item.predicate == "setting" and item.value:
                setting = item.value
            if item.predicate == "scene_type" and item.value and not setting:
                setting = item.value
            if item.predicate == "indoor_outdoor" and item.value:
                indoor = item.value
        for line in packet.environment:
            low = line.lower()
            if low.startswith("setting=") and not setting:
                setting = line.split("=", 1)[-1]
            if low.startswith("indoor_outdoor=") and not indoor:
                indoor = line.split("=", 1)[-1]
            if low.startswith("scene_type=") and not setting:
                setting = line.split("=", 1)[-1]
        if setting and setting.lower() not in {"unknown", "general"}:
            place = setting.replace("_", " ")
            if indoor == "indoor" or "indoor" in " ".join(packet.environment).lower():
                return f"The image shows an indoor {place}."
            return f"The setting appears to be a {place}."
        if indoor == "indoor":
            return "The image shows an indoor space."
        return ""

    def _select(self, packet: AssistantEvidencePacket, question: str) -> list[EvidenceItem]:
        q = (question or "").lower()
        selected: list[EvidenceItem] = []

        def _push(item: EvidenceItem) -> None:
            if item not in selected:
                selected.append(item)

        # Wearing / appearance → always include all reliable person clothing attrs.
        if any(tok in q for tok in ("wear", "wearing", "jacket", "clothing", "outfit", "dressed", "color", "colour")):
            for item in packet.items:
                if item.kind == "attribute" and item.predicate.lower() in (
                    _COLOR_ATTRS
                    | {"clothing_type", "jacket", "hoodie", "coat", "footwear_type"}
                ):
                    _push(item)
                if item.kind == "object" and item.subject.lower() in _PERSON:
                    _push(item)

        if any(tok in q for tok in ("equipment", "gear", "using", "holding", "leading", "riding", "ski", "pole")):
            for item in packet.items:
                if item.kind == "object" and any(tok in item.subject.lower() for tok in _EQUIPMENT | _ANIMALS):
                    _push(item)
                if item.kind == "relation":
                    _push(item)
                if item.kind == "activity":
                    _push(item)

        if any(tok in q for tok in ("fire", "smoke", "flame")):
            for item in packet.items:
                lab = item.subject.lower()
                if item.kind == "object" and (
                    lab in _FIRE_LABELS or lab in _SMOKE_LABELS or "fire" in lab or "smoke" in lab
                ):
                    _push(item)
                if item.kind == "environment" and (
                    "fire" in (item.value or "").lower() or "smoke" in (item.value or "").lower()
                ):
                    _push(item)

        if "animal" in q:
            for item in packet.items:
                if item.kind == "object" and item.subject.lower() in _ANIMALS:
                    _push(item)

        if any(tok in q for tok in ("how many", "count", "number", "people", "person", "vehicle", "car")):
            for item in packet.items:
                if item.kind == "object":
                    _push(item)

        if any(tok in q for tok in ("doing", "happening", "activity", "action", "skiing", "crossing")):
            for item in packet.items:
                if item.kind == "activity":
                    _push(item)

        if any(tok in q for tok in ("text", "ocr", "sign", "read", "writing")):
            for item in packet.items:
                if item.kind == "ocr":
                    _push(item)

        if any(
            tok in q
            for tok in ("background", "environment", "weather", "snow", "street", "outdoor", "where", "setting")
        ):
            for item in packet.items:
                if item.kind in {"environment", "object"}:
                    _push(item)

        if any(tok in q for tok in ("near", "beside", "behind", "front", "relative", "close", "next")):
            for item in packet.items:
                if item.kind in {"relation", "object"}:
                    _push(item)

        # Prefer a tight match set — never dump the full packet by default.
        if not selected:
            selected = list(packet.reliable_items()[:12])

        if len(selected) < 3:
            for item in packet.reliable_items():
                _push(item)
                if len(selected) >= 8:
                    break
        return selected

    def _format_prompt(
        self,
        packet: AssistantEvidencePacket,
        question: str,
        selected: list[EvidenceItem],
        direct: str,
    ) -> str:
        lines = [
            "VISUAL EVIDENCE FACTS (answer the question only — never restate this as an inventory):",
        ]
        if direct:
            lines.append(f"GROUNDED ANSWER DRAFT FROM EVIDENCE: {direct}")
            lines.append(
                "Use this draft unless it conflicts with RELIABLE evidence lines below. "
                "Do not say the information is unavailable."
            )
        if not selected:
            lines.append("(no matching evidence items)")
        for item in selected[:12]:
            status = "RELIABLE" if item.reliable else "LOW_CONFIDENCE_DO_NOT_ASSERT"
            # Never expose entity IDs / packet structure to the LLM.
            lines.append(
                f"- [{status}] {item.kind}: {item.subject} {item.predicate}={item.value} "
                f"(conf={item.confidence:.2f})"
            )

        # Never include the scene caption in the assistant prompt — answers must
        # come only from verified evidence, not by restating the caption.
        q = question.lower()
        if any(tok in q for tok in ("age", "old", "years")):
            lines.append(
                "NOTE: Exact age cannot be determined. Refuse age estimates."
            )
        if any(tok in q for tok in ("shoe", "shoes", "sneaker")) and not find_attribute(
            packet, predicate="shoes_color", require_reliable=True
        ):
            lines.append(
                "NOTE: No reliable shoe-color evidence. "
                "Say shoes are not clearly visible enough to determine color."
            )
        lines.append(
            "OUTPUT RULES: Answer in 1–3 short sentences. "
            "Do NOT restate or append the scene caption. "
            "Do NOT output Entities/Attributes/Relationships sections, "
            "internal entity IDs, confidence tables, or analysis breakdowns."
        )
        return "\n".join(lines)


def retrieve_relevant_evidence(packet: AssistantEvidencePacket, question: str) -> str:
    """Backward-compatible wrapper returning the prompt block string."""
    return VisualEvidenceRetriever().retrieve(packet, question).prompt_block
