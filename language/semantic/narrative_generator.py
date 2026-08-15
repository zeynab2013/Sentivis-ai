"""Convert verified evidence into competition-quality narrative captions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from core.contracts.analysis import ActivityEvidence, EnvironmentInfo, SceneContext, SceneGraph, SceneNode
from core.contracts.language import CaptionQualityReport, VisualObservations
from core.logging import get_logger
from language.semantic.scene_enrichment import SceneEnrichment, enrich_scene
from language.validation.caption_validator import CaptionEvidenceValidator
from language.validation.sentence_evidence import SentenceEvidenceAnalyzer

logger = get_logger(__name__)

_PERSON_LABELS = {"person", "people", "man", "woman", "child"}
_MIN_FULL_WORDS = 40
_MAX_FULL_WORDS = 200
_MAX_SHORT_WORDS = 48
_MIN_EXEC_WORDS = 20
_MAX_EXEC_WORDS = 80
_DANGLING_TAIL = frozenset(
    {
        "while",
        "and",
        "or",
        "but",
        "another",
        "because",
        "with",
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "from",
        "than",
        "that",
        "which",
        "who",
    }
)


def _split_complete_sentences(text: str) -> list[str]:
    """Split on sentence boundaries; keep non-empty clauses ending with .!? when present."""
    body = " ".join((text or "").split()).strip()
    if not body:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", body) if p.strip()]
    return parts


def short_caption_from_paragraph(
    paragraph: str,
    *,
    max_words: int = _MAX_SHORT_WORDS,
) -> str:
    """Build a short caption that always ends on a complete sentence boundary."""
    text = " ".join((paragraph or "").split()).strip()
    if not text:
        return text
    # Heal legacy mid-sentence cuts that used ellipsis.
    text = re.sub(r"\s*\.{2,}\s*", " ", text).strip()
    text = re.sub(r"\s+,", ",", text)
    sentences = _split_complete_sentences(text)
    if not sentences:
        tokens = text.rstrip(",;:.").split()
        while tokens and tokens[-1].lower().strip(",;:") in _DANGLING_TAIL:
            tokens.pop()
        healed = " ".join(tokens).strip(" ,;:")
        return f"{healed}." if healed else ""

    selected: list[str] = []
    for sentence in sentences:
        # Skip redundant count restatements in the short form.
        if re.match(
            r"(?i)^\s*(?:Two|There are two)\s+people\s+and\s+two\s+horses\b",
            sentence,
        ):
            continue
        if re.search(r"(?i)\bspread across the open outdoor setting\b", sentence):
            # Distribution closers belong to the full narrative, not Short Caption.
            continue
        candidate = " ".join([*selected, sentence]).strip()
        words = candidate.split()
        if selected and len(words) > max_words:
            break
        selected.append(sentence)
        # Prefer up to two complete sentences for Short Caption.
        if len(selected) >= 2:
            break
        if len(words) >= max_words:
            break
    if not selected:
        selected = [sentences[0]]

    # Fold a verified hazard sentence when Short still lacks it and room remains.
    if len(selected) < 2 and len(sentences) > 1:
        lower_sel = " ".join(selected).lower()
        hazard = next(
            (
                s
                for s in sentences[1:]
                if re.search(r"\b(fire|smoke|flame|campfire)\b", s, re.IGNORECASE)
                and s not in selected
            ),
            None,
        )
        if hazard and not re.search(r"\b(fire|smoke|flame|campfire)\b", lower_sel):
            trial = f"{selected[0]} {hazard}".strip()
            if len(trial.split()) <= max_words:
                selected.append(hazard)

    short = " ".join(selected).strip()
    # Never duplicate the full multi-sentence narrative as Short Caption.
    if short.rstrip(".") == text.rstrip(".") and len(sentences) >= 3:
        short = selected[0] if selected else sentences[0]
    # If the only available clause never had a terminator (truncated input),
    # drop dangling conjunctions before closing the sentence.
    if not short.endswith((".", "!", "?")):
        tokens = short.rstrip(",;:").split()
        while tokens and tokens[-1].lower().strip(",;:") in _DANGLING_TAIL:
            tokens.pop()
        short = " ".join(tokens).strip(" ,;:")
        if short:
            short += "."
    # Guard against dangling conjunction tails from any legacy cutters.
    tokens = short.rstrip(".!?").split()
    while tokens and tokens[-1].lower().strip(",;:") in _DANGLING_TAIL:
        tokens.pop()
    short = " ".join(tokens).strip(" ,;:")
    if short and not short.endswith((".", "!", "?")):
        short += "."
    if short.endswith("..."):
        short = short.rstrip(".") + "."
    return short


def executive_summary_from_paragraph(
    paragraph: str,
    *,
    max_words: int = _MAX_EXEC_WORDS,
) -> str:
    """Concise verified summary — must not simply duplicate the full narrative."""
    text = " ".join((paragraph or "").split()).strip()
    if not text:
        return text
    sentences = _split_complete_sentences(text)
    if not sentences:
        return short_caption_from_paragraph(text, max_words=max_words)
    if len(sentences) == 1:
        # Single-sentence captions: short/executive may match; narrative equals them.
        return sentences[0] if sentences[0].endswith((".", "!", "?")) else f"{sentences[0]}."

    hazard = [
        s
        for s in sentences[1:]
        if re.search(r"\b(fire|smoke|flame|campfire)\b", s, re.IGNORECASE)
    ]
    # Prefer primary action + hazard; avoid distribution closers and count lines.
    secondary = [
        s
        for s in sentences[1:]
        if s not in hazard
        and not re.match(
            r"(?i)^\s*(?:Two|There are two)\s+people\s+and\s+two\s+horses\b",
            s,
        )
        and not re.search(r"(?i)\bspread across the open outdoor setting\b", s)
    ]
    parts = [sentences[0]]
    if hazard:
        parts.append(hazard[0])
    elif secondary:
        parts.append(secondary[0])
    elif len(sentences) > 2:
        parts.append(sentences[1])

    executive = " ".join(parts).strip()
    if executive.rstrip(".") == text.rstrip("."):
        # Avoid identical copy of the narrative when multiple sentences exist.
        executive = sentences[0]
        if hazard and hazard[0] != sentences[0]:
            trial = f"{executive} {hazard[0]}".strip()
            if trial.rstrip(".") != text.rstrip("."):
                executive = trial
    if not executive.endswith((".", "!", "?")):
        executive = executive.rstrip(",;:") + "."
    words = executive.split()
    if len(words) > max_words:
        # Truncate only by dropping trailing whole sentences.
        kept: list[str] = []
        for sentence in _split_complete_sentences(executive):
            trial = " ".join([*kept, sentence])
            if kept and len(trial.split()) > max_words:
                break
            kept.append(sentence)
        executive = " ".join(kept) if kept else sentences[0]
        if not executive.endswith((".", "!", "?")):
            executive += "."
    return executive


_RELATION_NARRATIVES: dict[str, str] = {
    "inside": "{subject} is inside {obj}.",
    "holding": "{subject} is holding {obj}.",
    "playing_with": "{subject} is engaged with {obj}.",
    "looking_at": "{subject} is looking toward {obj}.",
    "sitting_on": "{subject} is seated on {obj}.",
    "standing_beside": "{subject} stands beside {obj}.",
    "riding": "{subject} is riding {obj}.",
    "leading": "{subject} is leading {obj}.",
    "carrying": "{subject} is carrying {obj}.",
    "using": "{subject} is using {obj}.",
}

_ZONE_PHRASES: dict[str, str] = {
    "top-left": "toward the upper-left area",
    "top-center": "near the top of the scene",
    "top-right": "toward the upper-right area",
    "middle-left": "on the left side",
    "middle-center": "near the center",
    "middle-right": "on the right side",
    "bottom-left": "toward the lower-left area",
    "bottom-center": "near the bottom of the scene",
    "bottom-right": "toward the lower-right area",
}

_FORBIDDEN_PATTERNS = (
    re.compile(r"\b\d+\s*%"),
    re.compile(r"\bconfidence\b", re.I),
    re.compile(r"\bobject\s*id\b", re.I),
    re.compile(r"\[\d+\]"),
    re.compile(r"\bgeneral scene\b", re.I),
    re.compile(r"\bspatial relations identified\b", re.I),
    re.compile(r"\bobjects include\b", re.I),
    re.compile(r"\binferred activity\b", re.I),
    re.compile(r"\bverified activity\b", re.I),
    re.compile(r"\bcrowd level\b", re.I),
)


@dataclass(frozen=True)
class NarrativeCaption:
    """Competition-facing narrative outputs."""

    full_caption: str
    short_caption: str
    executive_summary: str = ""


class NarrativeGenerator:
    """Evidence-only narrative synthesis — never invents facts."""

    def __init__(self) -> None:
        self._validator = CaptionEvidenceValidator()
        self._sentence_evidence = SentenceEvidenceAnalyzer()

    def from_natural_paragraph(self, paragraph: str) -> NarrativeCaption:
        """Build UI narrative fields from one fluent competition paragraph."""
        text = " ".join(paragraph.split()).strip()
        if not text:
            text = "The image content could not be described with sufficient confidence."
        short = short_caption_from_paragraph(text)
        executive = executive_summary_from_paragraph(text)
        return NarrativeCaption(
            full_caption=text,
            short_caption=short,
            executive_summary=executive,
        )

    def generate(
        self,
        context: SceneContext,
        *,
        observations: VisualObservations | None = None,
        semantic_summary: str = "",
        quality_report: CaptionQualityReport | None = None,
        ollama_caption: str = "",
        natural_paragraph: str = "",
    ) -> NarrativeCaption:
        from language.refinement.caption_sanity import choose_better_caption, has_awkward_filler

        natural = (natural_paragraph or "").strip()
        ollama = (ollama_caption or "").strip()
        if natural and ollama and has_awkward_filler(natural):
            preferred = choose_better_caption(ollama, natural)
            if preferred:
                return self.from_natural_paragraph(preferred)
        if natural:
            cleaned = choose_better_caption(natural)
            return self.from_natural_paragraph(cleaned or natural)
        sentences = self._build_sentences(
            context,
            observations=observations,
            semantic_summary=semantic_summary,
            ollama_caption=ollama_caption,
        )
        validated = self._validate_sentences(sentences, context)
        full_text = self._compose_full_caption(validated, context)
        short_text = self._compose_short_caption(validated, context)
        full_sentences = self._validate_sentences(self._split_sentences(full_text), context)
        short_sentences = self._validate_sentences(self._split_sentences(short_text), context)
        full_text = self._join_sentences(full_sentences)
        short_text = self._join_sentences(short_sentences)
        if not full_text:
            full_text = self._minimal_fallback(context)
        if not short_text:
            short_text = self._minimal_short(context)
        full_text = self._sentence_evidence.filter_supported(full_text, context)
        short_text = self._sentence_evidence.filter_supported(short_text, context)
        enrichment = enrich_scene(context)
        executive = self._compose_executive_summary(context, enrichment, short_text)
        executive = self._sentence_evidence.filter_supported(executive, context)
        full_text = self._dedupe_object_mentions(full_text, context)
        logger.info(
            "Narrative generated full_words=%d short_words=%d",
            len(full_text.split()),
            len(short_text.split()),
        )
        return NarrativeCaption(
            full_caption=full_text,
            short_caption=short_text,
            executive_summary=executive,
        )

    def _build_sentences(
        self,
        context: SceneContext,
        *,
        observations: VisualObservations | None,
        semantic_summary: str,
        ollama_caption: str,
    ) -> list[str]:
        env = context.environment
        graph = context.graph
        sentences: list[str] = []

        opening = self._opening_sentence(env, context)
        if opening:
            sentences.append(opening)

        activity_text = self._activity_paragraph(context)
        if activity_text:
            sentences.extend(self._split_sentences(activity_text))

        for sentence in self._relation_sentences(graph):
            if sentence not in sentences:
                sentences.append(sentence)

        for sentence in self._object_detail_sentences(context):
            if sentence not in sentences:
                sentences.append(sentence)

        atmosphere = self._atmosphere_sentence(env)
        if atmosphere:
            sentences.append(atmosphere)

        layout = self._layout_sentence(context)
        if layout:
            sentences.append(layout)

        for hint in self._blip_sentences(observations, context):
            if hint not in sentences:
                sentences.append(hint)

        for sentence in self._ollama_sentences(semantic_summary, ollama_caption, context):
            if sentence not in sentences:
                sentences.append(sentence)

        enrichment = enrich_scene(context)
        enrich_sentence = self._enrichment_sentence(enrichment, context)
        if enrich_sentence:
            sentences.append(enrich_sentence)

        return sentences

    def _opening_sentence(self, env: EnvironmentInfo, context: SceneContext) -> str:
        setting = self._readable_setting(env).strip()
        indoor_outdoor = env.indoor_outdoor
        person_count = sum(1 for node in context.graph.nodes if node.label.lower() in _PERSON_LABELS)

        if indoor_outdoor == "outdoor":
            prefix = (
                f"The photograph captures an outdoor {setting}"
                if setting and setting not in {"an outdoor area"}
                else "The photograph captures an outdoor scene"
            )
        elif indoor_outdoor == "indoor":
            prefix = (
                f"The scene takes place in {setting}"
                if setting
                else "The scene takes place indoors"
            )
        elif setting:
            prefix = f"The image shows {setting}"
        else:
            prefix = "The image shows a scene"

        if person_count >= 3:
            prefix += ", where a group of people gathers"
        elif person_count == 2:
            prefix += ", where two people appear together"
        elif person_count == 1:
            prefix += ", with one person visible"
        elif context.object_count:
            prefix += f", featuring {self._object_count_phrase(context.object_count)}"
        return f"{prefix}."

    def _activity_paragraph(self, context: SceneContext) -> str:
        activities = sorted(
            context.activities.activities,
            key=lambda item: item.confidence,
            reverse=True,
        )
        if not activities:
            return ""

        primary = self._select_primary_activity(activities)
        person_count = sum(1 for node in context.graph.nodes if node.label.lower() in _PERSON_LABELS)

        if not primary:
            return ""
        activity_phrase = self._human_activity(primary)
        if person_count >= 2:
            return f"Several people are {activity_phrase}."
        if person_count == 1:
            return f"A person is {activity_phrase}."
        return f"The scene shows {activity_phrase}."

    def _relation_sentences(self, graph: SceneGraph) -> list[str]:
        if not graph.relations:
            return []
        labels = {node.index: node.label for node in graph.nodes}
        sentences: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in graph.relations:
            if relation.relation_type not in _RELATION_NARRATIVES:
                continue
            if relation.confidence < 0.62:
                continue
            subject = labels.get(relation.subject_index, "object")
            obj = labels.get(relation.object_index, "object")
            template = _RELATION_NARRATIVES[relation.relation_type]
            phrase = self._article(subject)
            object_phrase = self._article(obj)
            sentence = template.format(subject=phrase, obj=object_phrase)
            key = (subject.lower(), relation.relation_type, obj.lower())
            if key in seen:
                continue
            seen.add(key)
            sentences.append(sentence)
            if len(sentences) >= 4:
                break
        return sentences

    def _object_detail_sentences(self, context: SceneContext) -> list[str]:
        nodes = context.graph.nodes
        if not nodes:
            return ["No distinct objects were detected in this image."]

        mentioned: set[str] = set()
        sentences: list[str] = []
        by_label: dict[str, list[SceneNode]] = {}
        for node in nodes:
            by_label.setdefault(node.label.lower(), []).append(node)

        for label, group in sorted(by_label.items(), key=lambda item: -len(item[1])):
            if label in _PERSON_LABELS:
                continue
            count = len(group)
            zone = _ZONE_PHRASES.get(group[0].position_zone, "in the scene")
            color = self._attribute_value(context, group[0].index, "dominant_color")
            color_clause = f" with {color} tones" if color not in {"", "unknown"} else ""
            if count == 1:
                sentences.append(f"A {label} is visible {zone}{color_clause}.")
            else:
                sentences.append(f"Multiple {label}s appear {zone}{color_clause}.")
            mentioned.add(label)
            if len(sentences) >= 5:
                break

        for node in nodes:
            if node.label.lower() not in _PERSON_LABELS:
                continue
            shirt = self._attribute_value(context, node.index, "shirt_color")
            hair = self._attribute_value(context, node.index, "hair_color")
            if shirt not in {"", "unknown"} or hair not in {"", "unknown"}:
                parts = []
                if hair not in {"", "unknown"}:
                    parts.append(f"{hair} hair")
                if shirt not in {"", "unknown"}:
                    parts.append(f"a {shirt} shirt")
                if parts:
                    sentences.append(f"A person shows {' and '.join(parts)}.")
            if len(sentences) >= 8:
                break

        dominant = [label for label in context.dominant_objects if label.lower() not in _PERSON_LABELS][:3]
        extras = [label for label in dominant if label.lower() not in mentioned]
        for label in extras[:2]:
            sentences.append(f"{self._article(label).capitalize()} stands out among the visible elements.")
        return sentences

    @staticmethod
    def _attribute_value(context: SceneContext, object_index: int, name: str) -> str:
        for attribute in context.attributes.attributes:
            if attribute.object_index == object_index and attribute.name == name:
                return attribute.value
        return ""

    def _atmosphere_sentence(self, env: EnvironmentInfo) -> str:
        parts: list[str] = []
        if env.weather not in {"", "unknown"}:
            if env.weather == "clear":
                parts.append("clear weather conditions")
            elif env.weather == "rainy":
                parts.append("rainy weather suggested by visible cues")
            elif env.weather == "snowy":
                parts.append("snowy conditions")
        if env.time_of_day not in {"", "unknown"}:
            if env.time_of_day == "daytime":
                parts.append("daylight")
            elif env.time_of_day == "night":
                parts.append("nighttime lighting")
        if not parts:
            return ""
        joined = " and ".join(parts)
        return f"The atmosphere suggests {joined}."

    def _layout_sentence(self, context: SceneContext) -> str:
        zones = sorted({node.position_zone for node in context.graph.nodes if node.position_zone})
        if len(zones) <= 1:
            return ""
        zone_text = ", ".join(_ZONE_PHRASES.get(zone, zone.replace("-", " ")) for zone in zones[:4])
        return f"Elements are distributed across the frame, with notable presence {zone_text}."

    def _blip_sentences(
        self,
        observations: VisualObservations | None,
        context: SceneContext,
    ) -> list[str]:
        if not observations:
            return []
        candidates: list[str] = []
        if observations.raw_caption.text.strip():
            candidates.append(observations.raw_caption.text.strip())
        candidates.extend(observations.observations[:2])
        kept: list[str] = []
        for text in candidates:
            for sentence in self._split_sentences(text):
                cleaned = sentence.strip()
                if cleaned and not self._looks_template(cleaned):
                    validated = self._validate_sentences([cleaned], context)
                    kept.extend(validated)
        return kept[:2]

    def _ollama_sentences(
        self,
        semantic_summary: str,
        ollama_caption: str,
        context: SceneContext,
    ) -> list[str]:
        combined = " ".join(part.strip() for part in (semantic_summary, ollama_caption) if part.strip())
        if not combined:
            return []
        sentences = self._split_sentences(combined)
        return self._validate_sentences(sentences, context)[:3]

    def _compose_full_caption(self, sentences: list[str], context: SceneContext) -> str:
        if not sentences:
            return self._minimal_fallback(context)
        unique: list[str] = []
        for sentence in sentences:
            normalized = sentence.strip()
            if normalized and normalized not in unique:
                unique.append(normalized)
        text = self._join_sentences(unique)
        words = text.split()
        if len(words) < _MIN_FULL_WORDS:
            extra = self._expand_sentences(context, unique)
            for sentence in extra:
                if sentence not in unique:
                    unique.append(sentence)
            text = self._join_sentences(unique)
            words = text.split()
        # Do not pad with filler sentences — brevity beats generic padding.
        if len(words) > _MAX_FULL_WORDS:
            trimmed: list[str] = []
            count = 0
            for sentence in unique:
                sentence_words = len(sentence.split())
                if count + sentence_words > _MAX_FULL_WORDS:
                    break
                trimmed.append(sentence)
                count += sentence_words
            text = self._join_sentences(trimmed)
        return text

    def _expand_sentences(self, context: SceneContext, existing: list[str]) -> list[str]:
        extras: list[str] = []
        env = context.environment
        graph = context.graph
        if env.social_context not in {"", "unknown", "no people detected"}:
            extras.append(f"The social context reads as {env.social_context.replace('_', ' ')}.")
        if env.scene_complexity not in {"", "unknown"}:
            extras.append(
                f"Interaction complexity in the scene appears {env.scene_complexity}, "
                f"with {len(graph.relations)} spatial relationships shaping the layout."
            )
        if env.crowd_level not in {"", "unknown"}:
            extras.append(f"The scene carries a {env.crowd_level.replace('_', ' ')} crowd character.")
        for sentence in self._object_detail_sentences(context):
            if sentence not in existing and sentence not in extras:
                extras.append(sentence)
        for sentence in self._relation_sentences(context.graph):
            if sentence not in existing and sentence not in extras:
                extras.append(sentence)
        for node in graph.nodes[:8]:
            zone = _ZONE_PHRASES.get(node.position_zone, "within the frame")
            sentence = f"The {node.label} occupies space {zone}, contributing to the overall composition."
            if sentence not in existing and sentence not in extras:
                extras.append(sentence)
        for activity in context.activities.activities[:3]:
            rationale = activity.rationale.rstrip(".")
            if "via " in rationale.lower():
                rationale = rationale.split("via ", maxsplit=1)[0].rstrip()
            sentence = f"The activity of {activity.activity} is supported by evidence that {rationale.lower()}."
            if sentence not in existing and sentence not in extras:
                extras.append(sentence)
        if env.setting and env.setting not in {"photographed scene", "unknown", "general scene"}:
            extras.append(f"The location is {env.setting}.")
        if env.weather not in {"", "unknown", "clear", "none"}:
            extras.append(f"Weather conditions appear {env.weather}.")
        if env.time_of_day not in {"", "unknown", "general", "day"}:
            extras.append(f"The time of day is {env.time_of_day}.")
        return extras

    def _filler_sentences(self, context: SceneContext, existing: list[str]) -> list[str]:
        """Disabled — filler padding reduces competition quality."""
        _ = context, existing
        return []

    def _compose_short_caption(self, sentences: list[str], context: SceneContext) -> str:
        env = context.environment
        setting = self._readable_setting(env)
        activities = context.activities.activities
        primary = self._select_primary_activity(activities)
        person_count = sum(1 for node in context.graph.nodes if node.label.lower() in _PERSON_LABELS)

        parts: list[str] = []
        if person_count >= 3:
            parts.append("Several people")
        elif person_count == 2:
            parts.append("Two people")
        elif person_count == 1:
            parts.append("A person")
        else:
            key_objects = [
                label
                for label in context.dominant_objects
                if label.lower() not in _PERSON_LABELS
            ][:2]
            if key_objects:
                label = key_objects[0]
                phrase = f"{label.capitalize()}s" if len(key_objects) > 1 else self._article(label).capitalize()
                parts.append(phrase)
            else:
                parts.append("The scene")

        if primary and primary not in {"static scene", "people present"}:
            if person_count == 1:
                parts.append(f"is {self._human_activity(primary)}")
            else:
                parts.append(f"are {self._human_activity(primary)}")
        elif person_count:
            parts.append("is present" if person_count == 1 else "are present")
        else:
            parts.append("shows")

        if setting:
            parts.append(f"in {setting}")

        candidate = " ".join(parts).strip()
        candidate = re.sub(r"\s+", " ", candidate)
        if not candidate.endswith("."):
            candidate += "."
        candidate = short_caption_from_paragraph(candidate, max_words=_MAX_SHORT_WORDS)
        validated = self._validate_sentences(self._split_sentences(candidate), context)
        if validated:
            return self._join_sentences(validated)
        if sentences:
            return short_caption_from_paragraph(sentences[0], max_words=_MAX_SHORT_WORDS)
        return self._minimal_short(context)

    def _validate_sentences(self, sentences: list[str], context: SceneContext) -> list[str]:
        kept: list[str] = []
        for sentence in sentences:
            cleaned = self._clean_sentence(sentence)
            if not cleaned:
                continue
            filtered = self._validator.filter_unsupported_sentences(cleaned, context)
            filtered = self._clean_sentence(filtered)
            if filtered and filtered not in kept:
                kept.append(filtered)
        return kept

    def _clean_sentence(self, sentence: str) -> str:
        text = sentence.strip()
        if not text:
            return ""
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(text):
                return ""
        text = re.sub(r"\s+", " ", text)
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def _readable_setting(self, env: EnvironmentInfo) -> str:
        setting = env.setting.strip()
        scene_type = env.scene_type.strip()
        blocked = {
            "general scene",
            "everyday environment",
            "unknown",
            "photographed scene",
            "object-focused scene",
        }
        if setting and setting not in blocked:
            return setting
        if scene_type and scene_type not in blocked:
            return scene_type
        if env.indoor_outdoor == "outdoor":
            return "an outdoor area"
        if env.indoor_outdoor == "indoor":
            return "an indoor space"
        return ""

    def _select_primary_activity(self, activities: Sequence[ActivityEvidence]) -> str:
        ranked = sorted(activities, key=lambda item: item.confidence, reverse=True)
        weak = {
            "people present",
            "static scene",
            "waiting",
            "having a conversation",
            "transportation scene",
            "standing",
            "sitting",
        }
        for item in ranked:
            if item.activity.lower() not in weak and item.confidence >= 0.65:
                return item.activity.lower()
        return ""

    def _human_activity(self, activity: str) -> str:
        mapping = {
            "playing tennis": "playing tennis",
            "playing baseball": "playing baseball",
            "playing basketball": "playing basketball",
            "playing soccer": "playing with a ball outdoors",
            "playing with a ball": "playing with a ball",
            "playing sports": "engaged in a sporting activity",
            "skateboarding": "skateboarding",
            "surfing": "surfing",
            "cycling": "cycling",
            "dining": "dining",
            "preparing food": "preparing food",
            "working": "working",
            "reading": "reading",
            "having a conversation": "having a conversation",
            "walking together": "walking together",
            "walking": "walking",
            "waiting": "waiting",
            "driving": "driving",
            "transportation scene": "near active transportation",
            "shopping": "shopping",
            "people present": "present in the scene",
            "static scene": "observing a quiet scene",
        }
        return mapping.get(activity.lower(), activity.lower())

    def _object_count_phrase(self, count: int) -> str:
        if count >= 10:
            return "numerous visible objects"
        if count >= 5:
            return "several distinct objects"
        if count >= 2:
            return "a few key objects"
        return "a primary object"

    def _article(self, label: str) -> str:
        lower = label.lower()
        if lower in {"person", "people"}:
            return "a person" if lower == "person" else "people"
        if lower[0] in "aeiou":
            return f"an {lower}"
        return f"a {lower}"

    def _split_sentences(self, text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        return [part.strip() for part in parts if part.strip()]

    def _join_sentences(self, sentences: list[str]) -> str:
        return " ".join(sentence.strip() for sentence in sentences if sentence.strip())

    def _looks_template(self, sentence: str) -> bool:
        lower = sentence.lower()
        markers = (
            "objects include",
            "spatial relations",
            "supported activity",
            "crowd level",
            "this appears to be",
            "key relations:",
        )
        return any(marker in lower for marker in markers)

    def _minimal_fallback(self, context: SceneContext) -> str:
        setting = self._readable_setting(context.environment)
        labels = [node.label for node in context.graph.nodes[:4]]
        if labels:
            objects = ", ".join(labels)
            return f"The image depicts {setting}, with {objects} visible among the main elements."
        return f"The image depicts {setting}."

    def _minimal_short(self, context: SceneContext) -> str:
        setting = self._readable_setting(context.environment)
        return f"A scene in {setting}."

    def _enrichment_sentence(self, enrichment: SceneEnrichment, context: SceneContext) -> str:
        # Decorative atmosphere inference is not competition-quality evidence.
        _ = enrichment, context
        return ""

    def _compose_executive_summary(
        self,
        context: SceneContext,
        enrichment: SceneEnrichment,
        short_caption: str,
    ) -> str:
        _ = enrichment
        s1 = short_caption.rstrip(".") + "."
        primary = self._select_primary_activity(context.activities.activities)
        parts = [s1]
        if primary:
            activity = self._human_activity(primary)
            if activity and activity.lower() not in s1.lower():
                # Only add when short caption did not already state the activity.
                parts.append(f"Activity: {activity}.")
        setting = self._readable_setting(context.environment)
        if setting and setting.lower() not in s1.lower():
            # Avoid "The location is indoor" restating an already-placed scene.
            if not any(tok in s1.lower() for tok in ("office", "kitchen", "street", "outdoor", "indoor")):
                parts.append(f"The location is {setting}.")
        validated = self._validate_sentences(parts, context)
        text = " ".join(validated) if validated else s1
        # Prefer paragraph-derived executive when short_caption is already a full narrative.
        derived = executive_summary_from_paragraph(
            " ".join(parts) if parts else short_caption,
            max_words=_MAX_EXEC_WORDS,
        )
        if derived and derived.rstrip(".") != short_caption.rstrip("."):
            text = derived
        else:
            text = executive_summary_from_paragraph(text, max_words=_MAX_EXEC_WORDS)
        return text

    def _dedupe_object_mentions(self, text: str, context: SceneContext) -> str:
        sentences = self._split_sentences(text)
        seen_labels: set[str] = set()
        kept: list[str] = []
        for sentence in sentences:
            lower = sentence.lower()
            labels_in = [node.label.lower() for node in context.graph.nodes if node.label.lower() in lower]
            if labels_in and all(label in seen_labels for label in labels_in):
                continue
            for label in labels_in:
                seen_labels.add(label)
            kept.append(sentence)
        return self._join_sentences(kept) if kept else text
