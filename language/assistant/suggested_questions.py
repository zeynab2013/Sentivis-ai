"""Generate a small set of image-specific, answerable suggested questions."""

from __future__ import annotations

import re

from language.assistant.evidence_packet import (
    AssistantEvidencePacket,
    find_attribute,
)
from language.assistant.entity_indexing import (
    find_person_attribute,
    ordered_people,
    resolve_person_reference,
)
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever

_LANGUAGE_NAMES = {
    "en": "English",
    "fa": "Persian",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
}

_COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "gray",
    "grey",
    "maroon",
    "navy",
    "brown",
    "orange",
    "pink",
    "purple",
    "beige",
    "cream",
    "charcoal",
    "silver",
    "gold",
    "tan",
    "olive",
    "burgundy",
    "light blue",
    "dark green",
}

_PERSON = {"person", "man", "woman", "child", "people", "skier", "rider"}
_VEHICLE = {"car", "bus", "truck", "motorcycle", "bicycle", "van", "taxi"}
_FIXTURES = {
    "refrigerator",
    "dining table",
    "chair",
    "tv",
    "couch",
    "bed",
    "oven",
    "sink",
    "table",
    "desk",
    "laptop",
    "monitor",
    "keyboard",
}
_MIN_SCORE = 60
_MAX_QUESTIONS = 5
_ANIMALS = {"horse", "dog", "cat", "cow", "sheep", "bird", "goat", "bear", "elephant", "zebra", "giraffe"}
_HAZARDS = {"fire", "smoke", "flame"}

# Partial detector labels → readable object names for questions.
_OBJECT_DISPLAY_ALIASES = {
    "potted": "potted plant",
    "dining": "dining table",
    "tv": "television",
    "cell": "cell phone",
}


def _object_display_name(label: str, label_set: set[str]) -> str:
    """Prefer full verified object labels in user-facing questions."""
    lab = (label or "").strip().lower()
    if not lab:
        return label
    if lab in label_set:
        return lab
    if lab in _OBJECT_DISPLAY_ALIASES:
        alias = _OBJECT_DISPLAY_ALIASES[lab]
        if alias in label_set or any(alias in other for other in label_set):
            return alias
    for full in sorted(label_set, key=len, reverse=True):
        if full == lab or full.startswith(f"{lab} ") or full.endswith(f" {lab}"):
            return full
    return _OBJECT_DISPLAY_ALIASES.get(lab, lab)


def _word_in(text: str, word: str) -> bool:
    """Whole-word match so color 'tan' does not hit 'stands'."""
    token = (word or "").strip().lower()
    if not token:
        return False
    return bool(re.search(rf"\b{re.escape(token)}\b", (text or "").lower()))

# Category tags for diversity (at most one kept per category unless score is very high).
_CATEGORY = {
    "activity": "activity",
    "spatial": "spatial",
    "appearance": "appearance",
    "ocr": "ocr",
    "count": "count",
    "environment": "environment",
    "equipment": "equipment",
    "object": "object",
    "relation": "relation",
}


def generate_suggested_questions(
    packet: AssistantEvidencePacket,
    *,
    language: str = "en",
    limit: int = 5,
    answered_questions: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """Return up to ``limit`` high-value, answerable, non-redundant questions.

    Questions are built from SceneGraph evidence and filtered against:
    - caption overlap (already answered by caption)
    - answerability (evidence path exists)
    - semantic novelty vs other candidates and prior answered questions
    - category diversity
    """
    from core.logging import get_logger

    logger = get_logger(__name__)
    cap = max(0, min(int(limit or 5), _MAX_QUESTIONS))
    caption = packet.canonical_caption_en or ""
    caption_l = caption.lower()
    prior = tuple((q or "").lower().strip() for q in (answered_questions or ()) if q)
    candidates = _build_candidates(packet, caption_l)
    retriever = VisualEvidenceRetriever()
    logger.info("SuggestedQ candidates=%s limit=%s", len(candidates), cap)

    ranked: list[tuple[int, str, str]] = []  # score, question, category
    seen_keys: set[str] = set()
    used_categories: set[str] = set()

    for score, question, cues, category in candidates:
        key = question.lower().strip()
        reason = None
        if score < _MIN_SCORE:
            reason = f"score<{_MIN_SCORE}"
        elif key in seen_keys or _semantic_duplicate(key, seen_keys | set(prior)):
            reason = "semantic_duplicate"
        elif _is_caption_duplicate(question, caption, cues):
            reason = "caption_duplicate"
        elif not _simulate_answerable(packet, question, retriever):
            reason = "not_answerable"
        elif category in used_categories and score < 80:
            reason = f"category_dup={category}"
        if reason:
            logger.info("SuggestedQ rejected score=%s q=%r reason=%s", score, question, reason)
            continue
        seen_keys.add(key)
        used_categories.add(category)
        ranked.append((score, question, category))
        logger.info("SuggestedQ kept score=%s cat=%s q=%r", score, category, question)

    if not ranked:
        # Soft fallback: keep the best answerable non-activity candidates.
        soft: list[tuple[int, str, str]] = []
        for score, question, cues, category in candidates:
            key = question.lower().strip()
            if score < 50:
                continue
            if category == "activity":
                continue
            if key in seen_keys or _semantic_duplicate(key, seen_keys | set(prior)):
                continue
            if _is_caption_duplicate(question, caption, cues):
                continue
            if not _simulate_answerable(packet, question, retriever):
                continue
            soft.append((score, question, category))
        if soft:
            soft.sort(key=lambda item: item[0], reverse=True)
            selected = [q for _s, q, _c in soft[: max(1, min(3, cap))]]
            logger.info("SuggestedQ soft-fallback: %s", selected)
            lang = (language or "en").lower()
            if lang != "en":
                localized = _localize_questions(selected, lang)
                return localized[:cap] if localized else selected
            return selected
        logger.info("SuggestedQ final: none")
        return []
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [q for _s, q, _c in ranked[:cap]]
    logger.info("SuggestedQ final: %s", selected)
    lang = (language or "en").lower()
    if lang != "en":
        localized = _localize_questions(selected, lang)
        return localized[:cap] if localized else selected
    return selected


def _build_candidates(
    packet: AssistantEvidencePacket,
    caption_l: str,
) -> list[tuple[int, str, tuple[str, ...], str]]:
    """Evidence-conditioned candidates with category tags for diversity."""
    present_objects = [
        item
        for item in packet.items
        if item.kind == "object" and item.confidence >= 0.42
    ]
    labels = [item.subject.lower() for item in present_objects]
    label_set = set(labels)
    people = sum(1 for lab in labels if lab in _PERSON)
    cars = sum(1 for lab in labels if lab in _VEHICLE)
    has_skis = any("ski" in lab for lab in labels)
    has_snowboard = any("snowboard" in lab for lab in labels)
    env_text = " ".join(packet.environment).lower()
    ocr_values = [item.value for item in packet.items if item.kind == "ocr" and item.value]
    if packet.ocr:
        ocr_values.extend(packet.ocr)
    candidates: list[tuple[int, str, tuple[str, ...], str]] = []

    # --- Equipment (ski scenes) ---
    if (has_skis or has_snowboard) and people >= 1 and not _caption_mentions_equipment(caption_l):
        poles = any("pole" in lab for lab in labels)
        if poles:
            candidates.append(
                (94, "What equipment is the skier using besides the skis?", ("equipment", "poles"), "equipment")
            )
        else:
            candidates.append(
                (88, "What ski equipment is clearly visible?", ("equipment", "skis"), "equipment")
            )

    # --- Activity ---
    activities = [
        item
        for item in packet.items
        if item.kind == "activity"
        and item.confidence >= 0.58
        and item.reliable
        and (item.claim_status or "").upper() != "UNCERTAIN"
        and (item.evidence_level or "").upper() in {"CONFIRMED", "SUPPORTED"}
    ]
    if people >= 1 and activities:
        # Prefer CONFIRMED over SUPPORTED; never packet[0] blindly.
        activities.sort(
            key=lambda it: (
                2 if (it.evidence_level or "").upper() == "CONFIRMED" else 1,
                1 if (it.claim_status or "").upper() == "OBSERVED" else 0,
                it.confidence,
            ),
            reverse=True,
        )
        act_item = activities[0]
        act = act_item.value.lower()
        person_in_caption = any(tok in caption_l for tok in ("person", "man", "woman", "people", "child"))
        act_tokens = [t for t in re.findall(r"[a-z]{4,}", act) if t not in {"with", "from", "scene", "standing"}]
        act_covered = bool(act_tokens) and any(t in caption_l for t in act_tokens)
        weak_activity = act in {
            "standing",
            "present",
            "visible",
            "general",
            "unknown",
            "sitting",
            "walking",
        }
        if not weak_activity and (not person_in_caption or not act_covered):
            candidates.append(
                (90, "What is the person doing in this scene?", (act.split()[0] if act else "activity",), "activity")
            )

    # --- Verified person–object spatial layout (SPATIAL relations only) ---
    spatial_near = [
        item
        for item in packet.items
        if item.kind == "relation"
        and item.confidence >= 0.68
        and item.claim_status != "UNCERTAIN"
        and (
            (item.relation_kind or "").upper() == "SPATIAL"
            or item.predicate.lower()
            in {
                "near",
                "beside",
                "next_to",
                "in_front_of",
                "behind",
                "above",
                "below",
                "on",
                "inside",
            }
        )
    ]
    omitted_fixtures = [
        lab
        for lab in labels
        if lab in _FIXTURES and lab not in caption_l and not (lab == "table" and "dining" in caption_l)
    ]
    if people >= 1 and spatial_near:
        # Prefer concrete object names when caption omitted a specific fixture.
        if omitted_fixtures:
            top = omitted_fixtures[0]
            if top == "dining table" and "table" not in caption_l:
                candidates.append(
                    (
                        87,
                        "Where is the dining table relative to the person?",
                        ("dining table", "near"),
                        "spatial",
                    )
                )
            candidates.append(
                (
                    85,
                    "What objects are positioned near the person?",
                    (omitted_fixtures[0],),
                    "spatial",
                )
            )
        else:
            candidates.append(
                (84, "What is positioned next to the person?", ("near", "beside"), "spatial")
            )
    elif people >= 1 and omitted_fixtures:
        # Co-presence without verified spatial edges → inventory, not proximity.
        top = omitted_fixtures[0]
        candidates.append(
            (
                83,
                f"Is there a {top} visible in the scene?",
                (top,),
                "object",
            )
        )

    # --- Appearance (only when reliable color exists and caption omitted it) ---
    shirt = find_attribute(
        packet,
        predicate="shirt_color",
        subject_tokens=tuple(_PERSON),
        require_reliable=True,
    )
    clothing = find_attribute(
        packet,
        predicate="clothing_color",
        subject_tokens=tuple(_PERSON),
        require_reliable=True,
    )
    color_item = shirt or clothing
    if (
        people >= 1
        and color_item is not None
        and color_item.confidence >= 0.62
        and not _word_in(caption_l, color_item.value.lower())
    ):
        # Still ask about color even if the caption already says "wearing …"
        # without naming the color.
        candidates.append(
            (
                82,
                "What color clothing is the person wearing?",
                (color_item.value.lower(),),
                "appearance",
            )
        )

    # Object color for a high-confidence non-person object omitted from caption.
    for item in packet.items:
        if item.kind != "attribute":
            continue
        if item.predicate.lower() not in {"color", "dominant_color"}:
            continue
        if item.confidence < 0.62 or not item.reliable:
            continue
        subj = item.subject.lower()
        if subj in _PERSON:
            continue
        if _word_in(caption_l, item.value.lower()):
            continue
        if subj in {"tv", "monitor", "screen", "laptop"}:
            continue  # display colors are unreliable
        display = _object_display_name(subj, label_set)
        if len(display.split()) < 2 and display.endswith(("ed", "ing")) and display not in label_set:
            continue  # skip truncated labels like bare "potted"
        # If caption already named one coat color for this animal, prefer the
        # "other {animal}" question below instead of a redundant primary ask.
        if subj in _ANIMALS and any(
            re.search(rf"\b{re.escape(color)}\s+{re.escape(subj)}\b", caption_l)
            for color in _COLOR_WORDS
        ):
            continue
        # Subject may already appear in the caption without its color — still ask.
        candidates.append(
            (
                int(78 + min(10, item.confidence * 10)),
                f"What color is the {display}?",
                (display, item.value.lower()),
                "appearance",
            )
        )
        break

    # Secondary animal color when caption named one coat color but evidence has another.
    animal_colors = [
        item
        for item in packet.items
        if item.kind == "attribute"
        and item.predicate.lower() in {"color", "dominant_color"}
        and item.subject.lower() in _ANIMALS
        and item.reliable
        and item.confidence >= 0.62
        and not _word_in(caption_l, item.value.lower())
    ]
    if animal_colors and any(_word_in(caption_l, a) for a in _ANIMALS):
        top = animal_colors[0]
        candidates.append(
            (
                84,
                f"What color is the other {top.subject.lower()}?",
                (top.subject.lower(), top.value.lower(), "other"),
                "appearance",
            )
        )

    # --- OCR ---
    if ocr_values:
        if not any(snip.lower() in caption_l for snip in ocr_values if len(snip) > 2):
            candidates.append(
                (93, "What readable text appears in the scene?", ("text", "reads"), "ocr")
            )

    # --- Counts ---
    if people >= 1 and not re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(people|persons|men|women|person|man|woman)\b",
        caption_l,
    ):
        candidates.append(
            (86, "How many people are visible in total?", ("how many people",), "count")
        )
    if cars >= 1 and "vehicle" not in caption_l and "car" not in caption_l and "bus" not in caption_l:
        if cars >= 2:
            candidates.append((84, "How many vehicles are visible?", ("how many", "vehicle"), "count"))
        else:
            candidates.append((78, "What vehicle is visible in the scene?", ("vehicle",), "object"))

    # --- Background / environment ---
    if "indoor_outdoor=outdoor" in env_text and "background" not in caption_l:
        bg_objs = [
            lab
            for lab in labels
            if lab in {"mountain", "tree", "building", "sky", "road", "snow", "hill"}
        ]
        if bg_objs:
            candidates.append(
                (78, "What is visible in the background of the scene?", ("background",), "environment")
            )
    if "indoor_outdoor=indoor" in env_text and people >= 1:
        if not any(tok in caption_l for tok in ("kitchen", "office", "room", "indoor", "workspace")):
            candidates.append(
                (76, "What kind of indoor space is shown?", ("indoor", "setting"), "environment")
            )

    # --- Omitted secondary object (image-specific) ---
    omitted = [
        item
        for item in present_objects
        if item.subject.lower() not in _PERSON
        and item.subject.lower() not in {"tree", "sky", "ground"}
        and item.subject.lower() not in caption_l
        and item.confidence >= 0.55
    ]
    if omitted:
        top = max(omitted, key=lambda i: i.confidence)
        lab = top.subject.lower()
        if not any(tok in lab for tok in ("ski", "snowboard")):
            synonyms = {
                "keyboard": ("computer", "laptop", "typing", "working"),
                "mouse": ("computer", "laptop", "keyboard"),
                "monitor": ("computer", "tv", "screen"),
                "tv": ("computer", "monitor", "screen"),
                "dining table": ("dining",),
            }
            syns = synonyms.get(lab, ())
            covered = any(s in caption_l for s in syns) if syns else False
            if lab == "dining table":
                covered = "dining table" in caption_l or "dining" in caption_l
            if not covered and lab not in _FIXTURES:
                candidates.append(
                    (
                        int(72 + min(15, top.confidence * 15)),
                        f"What other details are near the {lab}?",
                        (lab,),
                        "object",
                    )
                )

    # Vehicle-person spatial — categorical near only (no invented distance).
    strong_near_vehicle = any(
        item.kind == "relation"
        and item.predicate.lower() in {"near", "beside", "next_to", "in_front_of"}
        and item.confidence >= 0.60
        and (
            (item.relation_kind or "").upper() == "SPATIAL"
            or any(v in (item.value or "").lower() for v in _VEHICLE)
        )
        for item in packet.items
    )
    if (
        strong_near_vehicle
        and people
        and cars
        and not any(tok in caption_l for tok in ("near", "beside", "next to", "close"))
    ):
        candidates.append(
            (80, "Is the person near a vehicle?", ("near", "beside", "vehicle"), "spatial")
        )

    # --- Relationship graph (holding / sitting / using) not covered by caption ---
    # Only INTERACTION relations — never ask holding from spatial near.
    relation_qs: list[tuple[int, str, tuple[str, ...], str]] = []
    for item in packet.items:
        if item.kind != "relation" or item.confidence < 0.68:
            continue
        if item.claim_status == "UNCERTAIN":
            continue
        kind = (item.relation_kind or "").upper()
        pred = item.predicate.lower().replace(" ", "_")
        obj = (item.value or "").lower()
        if kind == "SPATIAL" or pred in {
            "near",
            "beside",
            "next_to",
            "left_of",
            "right_of",
            "behind",
            "in_front_of",
        }:
            continue
        if pred == "holding" and "hold" not in caption_l and obj:
            relation_qs.append(
                (88, "What is the person holding?", (obj, "holding"), "relation")
            )
        elif pred == "sitting_on" and "sit" not in caption_l and obj:
            relation_qs.append(
                (86, "What is the person sitting on?", (obj, "sitting"), "relation")
            )
        elif pred == "using" and "using" not in caption_l and obj:
            relation_qs.append(
                (85, f"What is the person using?", (obj, "using"), "relation")
            )
        elif pred == "leading" and "lead" in caption_l and obj in _ANIMALS:
            # Caption already covers leading — prefer a complementary count/appearance ask.
            pass
    # Keep up to two strong unique relation questions (different cues).
    if relation_qs:
        relation_qs.sort(key=lambda row: row[0], reverse=True)
        seen_preds: set[str] = set()
        for row in relation_qs:
            cue_key = row[2][0] if row[2] else row[1]
            if cue_key in seen_preds:
                continue
            seen_preds.add(cue_key)
            candidates.append(row)
            if len(seen_preds) >= 2:
                break

    # --- Fire / smoke hazards ---
    hazard_labels = [lab for lab in labels if lab in _HAZARDS]
    if hazard_labels and not any(tok in caption_l for tok in ("fire", "smoke", "burning", "flame")):
        candidates.append(
            (89, "Is there fire or smoke visible in the scene?", ("fire", "smoke"), "object")
        )
    elif hazard_labels and "smoke" in caption_l and "fire" not in caption_l:
        candidates.append(
            (79, "What is producing the smoke nearby?", ("smoke", "fire"), "object")
        )
    # Do not ask "What is near the fire?" — that routes poorly and invents layout.
    # --- Secondary person clothing / appearance beyond caption ---
    people_idx = ordered_people(packet)
    if len(people_idx) >= 2:
        secondary = people_idx[1]
        color_item = find_person_attribute(
            packet,
            secondary,
            predicates=("clothing_color", "shirt_color"),
            require_reliable=True,
            min_confidence=0.62,
        )
        if color_item is not None:
            value = (color_item.value or "").lower()
            # Skip ambiguous fashion labels unless strongly observed.
            ambiguous = {
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
            status = (color_item.claim_status or "").upper()
            if value in ambiguous and not (
                status == "OBSERVED" and color_item.confidence >= 0.62
            ):
                color_item = None
            elif _word_in(caption_l, value):
                color_item = None
        if color_item is not None:
            candidates.append(
                (
                    83,
                    "What color clothing is the second person wearing?",
                    (color_item.value.lower(),),
                    "appearance",
                )
            )

    # --- Holding: only when a verified INTERACTION holding relation exists ---
    if people >= 1 and "hold" not in caption_l:
        holding_rel = any(
            item.kind == "relation"
            and item.predicate.lower() == "holding"
            and item.confidence >= 0.68
            and item.claim_status != "UNCERTAIN"
            and (item.relation_kind or "").upper() in {"", "INTERACTION"}
            for item in packet.items
        )
        if holding_rel:
            candidates.append(
                (86, "What is the person holding?", ("holding",), "relation")
            )

    # --- Secondary person / background activity ---
    if people >= 2 and "second person" not in caption_l:
        second_activity = any(
            item.kind == "activity" and item.reliable and item.confidence >= 0.55
            for item in packet.items
        )
        if second_activity and not any(
            tok in caption_l for tok in ("other person", "farther back", "another person")
        ):
            # Ask a safer inventory-style question — activity may not be person-bound.
            candidates.append(
                (83, "How many people are visible?", ("people", "how many"), "count")
            )
        elif "farther back" in caption_l and "background" not in caption_l:
            candidates.append(
                (80, "What is happening in the background?", ("background",), "environment")
            )

    # --- Other animals beyond the lead interaction ---
    animal_labels = [lab for lab in labels if lab in _ANIMALS]
    animal_count = len(animal_labels)
    if animal_count >= 2:
        named = animal_labels[0]
        if f"other {named}" not in caption_l and "another horse" not in caption_l:
            if "horses" not in caption_l and named == "horse":
                candidates.append(
                    (85, "What other animals are visible?", ("horse", "animals"), "count")
                )
            else:
                candidates.append(
                    (84, f"How many {named}s are visible?", (named, "how many"), "count")
                )
    elif animal_count == 1 and animal_labels[0] not in caption_l:
        named = animal_labels[0]
        candidates.append(
            (82, f"What animals are visible besides people?", (named,), "object")
        )

    # Dense captions can suppress every specialty cue — still offer answerable layout asks.
    if len(candidates) < 2 and present_objects:
        bg_labels: list[str] = []
        for line in packet.objects:
            low = (line or "").lower()
            if "zone=background" not in low:
                continue
            match = re.search(r":([a-z ]+?)\s*\(", low)
            if not match:
                continue
            lab = match.group(1).strip()
            if lab and lab not in _PERSON:
                bg_labels.append(lab)
        if bg_labels and "background" not in caption_l:
            display = _object_display_name(bg_labels[0], label_set)
            candidates.append(
                (
                    74,
                    f"What is visible in the background near the {display}?",
                    (display, "background"),
                    "environment",
                )
            )
        if people >= 1 and any(lab in _FIXTURES for lab in label_set):
            fixture = next(lab for lab in label_set if lab in _FIXTURES)
            display = _object_display_name(fixture, label_set)
            candidates.append(
                (
                    71,
                    f"Where is the {display} located in the scene?",
                    (display, "where"),
                    "spatial",
                )
            )
        # Always try a verified object inventory ask when sparse.
        non_person = [
            item
            for item in present_objects
            if item.subject.lower() not in _PERSON
            and (item.claim_status or "").upper() != "UNCERTAIN"
            and item.confidence >= 0.50
        ]
        if non_person and not any("what objects" in c[1].lower() for c in candidates):
            top = max(non_person, key=lambda i: i.confidence)
            display = _object_display_name(top.subject.lower(), label_set)
            if display not in caption_l:
                candidates.append(
                    (
                        72,
                        f"Is a {display} visible in the scene?",
                        (display, "visible"),
                        "object",
                    )
                )
            elif people >= 1:
                candidates.append(
                    (
                        70,
                        f"Where is the {display} relative to the person?",
                        (display, "where"),
                        "spatial",
                    )
                )
        if "kitchen" in env_text and "refrigerator" in label_set:
            candidates.append(
                (
                    70,
                    "What else is visible near the refrigerator?",
                    ("refrigerator",),
                    "object",
                )
            )
        # Natural / outdoor environment asks when scene evidence exists.
        if any(
            "natural environment" in (item.value or "").lower()
            or "outdoor" in (item.value or "").lower()
            for item in packet.items
            if item.kind == "environment"
        ) and "background" not in caption_l:
            candidates.append(
                (
                    73,
                    "What is visible in the background of the scene?",
                    ("background",),
                    "environment",
                )
            )

    _ = label_set
    return candidates


def _semantic_duplicate(candidate: str, existing: set[str]) -> bool:
    """Reject questions that ask essentially the same thing."""
    cand_tokens = set(re.findall(r"[a-z]{4,}", candidate)) - {
        "what",
        "which",
        "where",
        "how",
        "many",
        "visible",
        "scene",
        "image",
        "person",
        "objects",
        "object",
        "this",
        "that",
        "with",
        "from",
        "near",
        "beside",
        "next",
        "close",
        "color",
        "other",
        "second",
    }
    if not cand_tokens:
        return False
    for prior in existing:
        # Distinguish primary vs secondary subject questions.
        if ("second" in candidate) != ("second" in prior):
            continue
        if ("other" in candidate) != ("other" in prior):
            continue
        prior_tokens = set(re.findall(r"[a-z]{4,}", prior)) - {
            "what",
            "which",
            "where",
            "how",
            "many",
            "visible",
            "scene",
            "image",
            "person",
            "objects",
            "object",
            "this",
            "that",
            "with",
            "from",
            "near",
            "beside",
            "next",
            "close",
            "color",
            "other",
            "second",
        }
        if not prior_tokens:
            continue
        overlap = cand_tokens & prior_tokens
        if len(overlap) >= max(2, min(len(cand_tokens), len(prior_tokens)) - 1):
            return True
        # Near/beside/next-to variants
        if {"positioned", "objects"} & cand_tokens and {"positioned", "objects", "beside", "next"} & prior_tokens:
            if ("person" in candidate and "person" in prior) or ("near" in candidate and "near" in prior):
                return True
    return False


def _simulate_answerable(
    packet: AssistantEvidencePacket,
    question: str,
    retriever: VisualEvidenceRetriever,
) -> bool:
    """Reject questions the assistant cannot confidently answer from SceneContext."""
    q = question.lower()
    if "shoe" in q:
        shoe = find_attribute(packet, predicate="shoes_color", require_reliable=True)
        return shoe is not None and shoe.confidence >= 0.60
    if "hair" in q:
        hair = find_attribute(packet, predicate="hair_color", require_reliable=True)
        return hair is not None and hair.confidence >= 0.60
    if "equipment" in q or ("ski" in q and "using" in q):
        return any(
            item.kind == "object"
            and item.confidence >= 0.55
            and any(tok in item.subject.lower() for tok in ("ski", "snowboard", "pole"))
            for item in packet.items
        )
    if "text" in q or "sign" in q or "readable" in q:
        return bool(packet.ocr) or any(
            item.kind == "ocr" and item.value for item in packet.items
        ) or any(
            item.kind == "object" and "sign" in item.subject.lower() and item.confidence >= 0.55
            for item in packet.items
        )
    if "how many people" in q:
        return (
            sum(
                1
                for item in packet.items
                if item.kind == "object"
                and item.subject.lower() in _PERSON
                and item.confidence >= 0.42
            )
            >= 2
        )
    if "how many vehicles" in q:
        return (
            sum(
                1
                for item in packet.items
                if item.kind == "object"
                and item.subject.lower() in _VEHICLE
                and item.confidence >= 0.55
            )
            >= 1
        )
    if "what is the person doing" in q or ("doing" in q and "person" in q):
        return any(
            item.kind == "activity"
            and item.confidence >= 0.58
            and item.reliable
            and (item.claim_status or "").upper() != "UNCERTAIN"
            and (item.evidence_level or "").upper() in {"CONFIRMED", "SUPPORTED"}
            and item.value.lower()
            not in {"standing", "sitting", "present", "visible", "walking"}
            for item in packet.items
        )
    if "near the person" in q or "next to the person" in q or "relative to the person" in q or (
        "positioned near" in q or "positioned next to" in q
    ):
        return any(
            item.kind == "relation"
            and item.confidence >= 0.60
            and item.claim_status != "UNCERTAIN"
            and (
                (item.relation_kind or "").upper() == "SPATIAL"
                or item.predicate.lower()
                in {
                    "near",
                    "beside",
                    "next_to",
                    "in_front_of",
                    "behind",
                    "above",
                    "below",
                    "on",
                    "inside",
                }
            )
            for item in packet.items
        )
    if "is the person near a vehicle" in q:
        return any(
            item.kind == "relation"
            and item.predicate.lower() in {"near", "beside", "next_to", "in_front_of"}
            and item.confidence >= 0.55
            for item in packet.items
        ) and any(
            item.kind == "object" and item.subject.lower() in _VEHICLE for item in packet.items
        )
    if "is there a " in q and "visible" in q:
        return any(item.kind == "object" and item.confidence >= 0.50 for item in packet.items)
    if "indoor space" in q or "kind of indoor" in q:
        return any(
            item.kind == "environment"
            and item.predicate in {"setting", "scene_type", "indoor_outdoor"}
            for item in packet.items
        ) or bool(packet.environment)
    if "color" in q and "clothing" in q:
        if "second person" in q or "first person" in q or "third person" in q:
            person = resolve_person_reference(q, packet)
            if person is None:
                return False
            color = find_person_attribute(
                packet,
                person,
                predicates=("clothing_color", "shirt_color"),
                require_reliable=True,
                min_confidence=0.62,
            )
            if color is None:
                return False
            # Mirror retriever: ambiguous colors need OBSERVED evidence.
            ambiguous = {
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
            value = (color.value or "").lower()
            status = (color.claim_status or "").upper()
            if value in ambiguous and not (
                status == "OBSERVED" and color.confidence >= 0.62
            ):
                return False
            return True
        return (
            find_attribute(packet, predicate="clothing_color", require_reliable=True) is not None
            or find_attribute(packet, predicate="shirt_color", require_reliable=True) is not None
        )
    if "color is the" in q:
        # Object-specific color question
        for item in packet.items:
            if item.kind == "attribute" and item.predicate.lower() in {"color", "dominant_color"}:
                if item.subject.lower() in q and item.reliable:
                    return True
        return False
    if "color is the other" in q:
        return any(
            item.kind == "attribute"
            and item.predicate.lower() in {"color", "dominant_color"}
            and item.subject.lower() in q
            and item.reliable
            for item in packet.items
        )
    if "burning" in q or "producing the smoke" in q or "fire or smoke visible" in q:
        return any(
            item.kind == "object"
            and item.subject.lower() in _HAZARDS
            and item.confidence >= 0.50
            for item in packet.items
        )
    if "other animals" in q or ("animals are visible" in q):
        return any(
            item.kind == "object"
            and item.subject.lower() in _ANIMALS
            and item.confidence >= 0.50
            for item in packet.items
        )
    if "how many" in q and any(a in q for a in _ANIMALS):
        return (
            sum(
                1
                for item in packet.items
                if item.kind == "object"
                and item.subject.lower() in _ANIMALS
                and item.confidence >= 0.42
            )
            >= 1
        )
    if "second person" in q or "happening in the background" in q:
        if "color" in q and "clothing" in q:
            # Handled above with person-indexed clothing checks.
            pass
        elif len(ordered_people(packet)) < 2:
            return False
        else:
            return (
                sum(
                    1
                    for item in packet.items
                    if item.kind == "object"
                    and item.subject.lower() in _PERSON
                    and item.confidence >= 0.42
                )
                >= 2
            )
    if "fire relative" in q or "near the fire" in q:
        # Prefer not to suggest these; if asked, require a spatial edge involving a hazard.
        return any(
            item.kind == "relation"
            and item.confidence >= 0.60
            and (
                any(h in (item.value or "").lower() for h in _HAZARDS)
                or any(h in (item.subject or "").lower() for h in _HAZARDS)
            )
            for item in packet.items
        )

    result = retriever.retrieve(packet, question)
    if result.direct_answer_en:
        low = result.direct_answer_en.lower()
        if any(
            tok in low
            for tok in (
                "not clearly visible",
                "cannot be determined",
                "no reliable",
                "not enough",
            )
        ):
            return False
        return True
    return bool(result.has_reliable_match and result.selected)


def _caption_mentions_equipment(caption_l: str) -> bool:
    return bool(re.search(r"\b(skis|snowboard|ski poles?|equipment|gear)\b", caption_l))


def _is_caption_duplicate(question: str, caption: str, cues: tuple[str, ...]) -> bool:
    q = question.lower()
    c = (caption or "").lower()
    if not c:
        return False

    if any(tok in q for tok in ("wearing", "wear", "jacket", "clothing", "shirt")) or (
        "color" in q and "clothing" in q
    ):
        # Only treat as duplicate when the caption already states a clothing color
        # for the same kind of ask (not just any color word like "dark green pants"
        # blocking a later shirt-color question).
        if "clothing" in q or "wearing" in q:
            cues_hit = [cue for cue in cues if cue and cue.lower() in c]
            if cues_hit and any(g in c for g in ("jacket", "shirt", "coat", "hoodie", "wearing", "pants")):
                return True
            if any(color in c for color in _COLOR_WORDS) and any(
                g in c for g in ("jacket", "shirt", "coat", "hoodie", "wearing")
            ) and any(color in c for color in _COLOR_WORDS if color in " ".join(cues)):
                return True
    # Object color question — only duplicate when THAT object's color is already stated.
    # Object color question — only duplicate when THAT object's asked color cue
    # is already stated (do not let "brown horse" block a tan-horse question).
    if "color is the" in q:
        subject_match = re.search(r"color is the (?:other )?([a-z ]+?)\??$", q)
        subj = subject_match.group(1).strip() if subject_match else ""
        cue_colors = [cue for cue in cues if cue in _COLOR_WORDS]
        if subj and cue_colors:
            if any(
                re.search(rf"\b{re.escape(color)}\s+{re.escape(subj)}\b", c)
                or re.search(rf"\b{re.escape(subj)}\s+(?:is|are)\s+{re.escape(color)}\b", c)
                for color in cue_colors
            ):
                return True
        elif subj:
            # No color cue — duplicate only if subject already has some color adjective.
            if any(
                re.search(rf"\b{re.escape(color)}\s+{re.escape(subj)}\b", c) for color in _COLOR_WORDS
            ):
                return True
        # Plural chairs / objects already colored in caption.
        if subj:
            plural = subj if subj.endswith("s") else f"{subj}s"
            if any(
                re.search(rf"\b{re.escape(color)}\s+{re.escape(plural)}\b", c)
                for color in _COLOR_WORDS
            ):
                return True

    # Generic "what color are the X" when caption already states color+X.
    color_are = re.search(r"what color are the ([a-z ]+?)\??$", q)
    if color_are:
        subj = color_are.group(1).strip()
        if any(re.search(rf"\b{re.escape(color)}\s+{re.escape(subj)}\b", c) for color in _COLOR_WORDS):
            return True

    if "equipment" in q and _caption_mentions_equipment(c):
        return True

    if "what is the skier doing" in q or ("doing" in q and "ski" in q):
        if any(tok in c for tok in ("skiing", "skis", "snowboard")):
            return True

    if "what is the person doing" in q:
        if any(tok in c for tok in ("preparing", "working", "skiing", "sitting", "standing", "holding")):
            # Activity already in caption — still allow if caption omitted the person.
            if any(tok in c for tok in ("person", "man", "woman", "people", "child")):
                return True

    content = [
        cue
        for cue in cues
        if cue
        and cue
        not in {"background", "farther", "near", "close", "how many", "vehicle", "ski", "skis", "beside", "activity"}
    ]
    if content and all(re.search(rf"\b{re.escape(cue.lower())}\b", c) for cue in content):
        return True
    return False


def _localize_questions(questions: list[str], lang: str) -> list[str]:
    """Best-effort localization via Ollama; fall back to English on failure."""
    try:
        from analysis.activity.ollama_client import OllamaClient

        client = OllamaClient(model="gemma3:4b", timeout_seconds=60.0)
        language_name = _LANGUAGE_NAMES.get(lang, lang)
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        system = (
            "Translate each numbered question into the target language. "
            "Keep the same order and count. Return only the numbered list."
        )
        user = f"Target language: {language_name}\n\n{numbered}"
        response = client.generate_text(system=system, user=user, max_tokens=220, purpose="translation")
        lines = [ln.strip() for ln in (response.text or "").splitlines() if ln.strip()]
        out: list[str] = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\).\:\-]\s*", "", line).strip()
            if cleaned:
                out.append(cleaned)
        if out:
            return out[: len(questions)]
    except Exception:  # noqa: BLE001
        pass
    return questions
