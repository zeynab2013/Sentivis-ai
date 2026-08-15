"""Caption quality evaluation."""

from __future__ import annotations

import re

from core.contracts.analysis import SceneContext
from core.contracts.language import CaptionQualityReport
from core.logging import get_logger
from language.validation.caption_validator import CaptionEvidenceValidator

logger = get_logger(__name__)

_RELATION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "holding": ("holding", "holds", "hold", "carrying", "carries", "carry", "with a", "with the"),
    "sitting_on": ("sitting on", "sits on", "seated on", "on the", "on a"),
    "standing_beside": ("beside", "next to", "alongside", "standing beside", "with"),
    "playing_with": ("playing", "play with", "sport", "sports"),
    "looking_at": ("looking at", "looking toward", "viewing", "reading", "watching"),
    "inside": ("inside", "within", "in the", "in a"),
    "outside": ("outside", "outdoors", "outdoor"),
    "near_vehicle": ("vehicle", "car", "bus", "truck", "near"),
    "near": ("near", "next to", "close to", "beside", "by"),
    "overlapping": ("overlapping", "overlap"),
    "behind": ("behind", "in back"),
    "in_front_of": ("in front", "ahead of"),
}

_ACTIVITY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "people present": ("people", "person", "individual", "someone", "present"),
    "static scene": ("scene", "static", "quiet", "still"),
    "playing sports": ("sport", "playing", "game", "athletic", "tennis", "baseball", "soccer", "ball"),
    "playing tennis": ("tennis", "playing", "racket"),
    "playing baseball": ("baseball", "bat", "playing"),
    "playing basketball": ("basketball", "playing"),
    "playing soccer": ("soccer", "ball", "playing"),
    "playing with a ball": ("ball", "playing"),
    "skateboarding": ("skateboard", "skating", "skate"),
    "surfing": ("surfboard", "surf"),
    "cycling": ("cycling", "bicycle", "bike"),
    "preparing food": ("preparing", "food", "kitchen", "cooking"),
    "having a conversation": ("conversation", "talking", "together", "speaking"),
    "walking together": ("walking", "together"),
    "dining": ("dining", "eating", "meal", "food"),
    "working": ("working", "work", "office", "technology"),
    "reading": ("reading", "read", "book"),
    "transportation scene": ("transport", "vehicle", "traffic", "driving"),
    "driving": ("driving", "drive", "vehicle"),
    "walking": ("walking", "walk", "pedestrian"),
    "waiting": ("waiting", "wait", "standing"),
}

# Activities that are placeholders — not real verified actions for coverage.
_WEAK_ACTIVITIES = frozenset({"people present", "static scene", "unknown", "none", ""})


class CaptionQualityEvaluator:
    """Evaluates caption quality against structured scene evidence."""

    def __init__(self) -> None:
        self._validator = CaptionEvidenceValidator()

    def evaluate(self, caption: str, context: SceneContext) -> CaptionQualityReport:
        text = caption.strip()
        notes: list[str] = []

        grammar_score = self._grammar_score(text)
        fluency_score = self._fluency_score(text)
        object_coverage = self._object_coverage(text, context)
        relationship_coverage = self._relationship_coverage(text, context)
        activity_coverage = self._activity_coverage(text, context)
        context_coverage = self._context_coverage(text, context)

        if object_coverage is None:
            notes.append("No verified objects — object coverage N/A.")
        if relationship_coverage is None:
            notes.append("No verified relationships — relationship coverage N/A.")
        if activity_coverage is None:
            notes.append("No verified activities — activity coverage N/A.")

        hallucination_risk = self._hallucination_risk(text, context)
        if hallucination_risk is None:
            notes.append("Hallucination safety not available.")
        elif hallucination_risk > 0:
            unsupported = self._validator.unsupported_object_tokens(text, context)
            if unsupported:
                notes.append(f"Potential unsupported tokens: {', '.join(unsupported)}")

        evidence_consistency = self._average(
            object_coverage,
            relationship_coverage,
            activity_coverage,
            context_coverage,
            None if hallucination_risk is None else max(0.0, 1.0 - hallucination_risk),
        )
        factual = 0.7 if hallucination_risk is None else max(0.0, 1.0 - hallucination_risk)
        obj_term = 0.5 if object_coverage is None else object_coverage
        overall_quality = (
            0.12 * grammar_score
            + 0.12 * fluency_score
            + 0.18 * evidence_consistency
            + 0.18 * obj_term
            + 0.40 * factual
        )
        # Serious unsupported semantic claims must not keep an artificially high score.
        if hallucination_risk is not None and hallucination_risk >= 0.35:
            overall_quality = min(overall_quality, max(0.0, 0.72 - 0.45 * hallucination_risk))
            notes.append("Unsupported semantic claims reduced overall quality.")
        if overall_quality < 0.5:
            notes.append("Overall caption quality below acceptable threshold.")

        report = CaptionQualityReport(
            grammar_score=grammar_score,
            fluency_score=fluency_score,
            evidence_consistency=evidence_consistency,
            object_coverage=object_coverage,
            relationship_coverage=relationship_coverage,
            activity_coverage=activity_coverage,
            context_coverage=context_coverage,
            hallucination_risk=hallucination_risk,
            overall_quality=overall_quality,
            notes=tuple(notes),
        )
        logger.info(
            "Caption quality overall=%.2f hallucination_risk=%s object=%s relation=%s activity=%s",
            overall_quality,
            "N/A" if hallucination_risk is None else f"{hallucination_risk:.2f}",
            "N/A" if object_coverage is None else f"{object_coverage:.2f}",
            "N/A" if relationship_coverage is None else f"{relationship_coverage:.2f}",
            "N/A" if activity_coverage is None else f"{activity_coverage:.2f}",
        )
        return report

    def _hallucination_risk(self, text: str, context: SceneContext) -> float | None:
        if not text.strip():
            return None
        # Without any graph evidence, unsupported-token checks are not meaningful.
        if not context.graph.nodes and not context.activities.activities:
            return None
        unsupported = self._validator.unsupported_object_tokens(text, context)
        risk = min(1.0, len(unsupported) * 0.25)
        risk = min(1.0, risk + self._unsupported_semantic_risk(text, context))
        return risk

    def _unsupported_semantic_risk(self, text: str, context: SceneContext) -> float:
        """Penalize strong venue/role/intent claims that lack graph-level support.

        Does not trust ``environment.setting`` alone — weak object→venue mappings
        must not launder unsupported caption claims into a zero hallucination score.
        """
        lower = text.lower()
        labels = {node.label.lower() for node in context.graph.nodes}
        activity_blob = " ".join(a.activity.lower() for a in context.activities.activities)
        relation_types = {r.relation_type for r in context.graph.relations}
        penalty = 0.0

        def _claim(*phrases: str) -> bool:
            return any(p in lower for p in phrases)

        # Specific venues require Tier-1 object evidence in the graph.
        if _claim("classroom", "school"):
            strong = (
                "backpack" in labels
                and "book" in labels
                and bool(labels & {"laptop", "keyboard", "chair"})
            )
            if not strong:
                penalty += 0.35
        if _claim("laboratory", " lab ", " lab.", "in a lab"):
            if len(labels & {"bottle", "bowl", "cup"}) < 3:
                penalty += 0.35
        if _claim("library"):
            if not ("book" in labels and len(labels & {"book", "chair", "laptop"}) >= 3):
                penalty += 0.30
        if _claim("restaurant"):
            if not (
                "dining table" in labels
                and labels & {"wine glass", "fork", "knife", "spoon"}
            ):
                penalty += 0.30
        if _claim("office") and not _claim("officer"):
            if len(labels & {"laptop", "keyboard", "mouse"}) < 2:
                penalty += 0.30

        # Occupational / academic intent claims.
        if _claim("student", "teacher", "schoolwork", "homework"):
            penalty += 0.40
        if _claim("studying", "attending a class", "attending class"):
            book_interact = "book" in labels and bool(
                relation_types & {"looking_at", "holding", "using", "reading"}
            )
            if not book_interact and "studying" not in activity_blob:
                penalty += 0.35
        if _claim("working") and not _claim("working with a horse", "network"):
            device_use = bool(labels & {"laptop", "keyboard", "mouse"}) and (
                "using" in relation_types or "using" in activity_blob
            )
            if not device_use and "working" not in activity_blob:
                penalty += 0.30

        return min(1.0, penalty)

    def _grammar_score(self, text: str) -> float:
        if not text:
            return 0.0
        score = 1.0
        if text[0].islower():
            score -= 0.2
        if text[-1] not in ".!?":
            score -= 0.15
        if re.search(r"\s{2,}", text):
            score -= 0.1
        if re.search(r"[.!?]{2,}", text):
            score -= 0.1
        # Broken auxiliary / determiner constructions (must not score 100%).
        broken = (
            r"\bwith one is\b",
            r"\ba other\b",
            r"\ban a\b",
            r"\bthe the\b",
            r"\bis is\b",
            r"\bare is\b",
            r"\bis are\b",
            r"\bthey is\b",
            r"\bpeople is\b",
            r"\bskis is\b",
        )
        lower = text.lower()
        for pattern in broken:
            if re.search(pattern, lower):
                score -= 0.35
        # Immediate word repeats ("the the", "person person").
        if re.search(r"\b([a-z]{2,})\s+\1\b", lower):
            score -= 0.2
        # Fragment openers mid-paragraph.
        if re.search(r"(?<=[.!?]\s)(?:Is|Are|With)\s+(?:also\s+)?visible\b", text):
            score -= 0.2
        return max(0.0, min(1.0, score))

    def _fluency_score(self, text: str) -> float:
        if not text:
            return 0.0
        lower = text.lower()
        template_markers = (
            "objects include",
            "spatial relations identified",
            "supported activity",
            "this appears to be",
            "crowd level",
        )
        if any(marker in lower for marker in template_markers):
            return 0.35
        sentences = [part for part in re.split(r"[.!?]+\s*", text) if part.strip()]
        if not sentences:
            return 0.4
        avg_len = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
        word_count = len(text.split())
        if 80 <= word_count <= 160 and 8 <= avg_len <= 32:
            return 0.96
        if 28 <= word_count <= 90 and 8 <= avg_len <= 28:
            return 0.93
        if 18 <= word_count <= 120 and 6 <= avg_len <= 32:
            return 0.88
        if word_count > 180:
            return 0.6
        if 8 <= avg_len <= 28:
            return 0.8
        if 4 <= avg_len <= 35:
            return 0.7
        return 0.55

    def _object_coverage(self, text: str, context: SceneContext) -> float | None:
        """Fraction of verified object *instances* accounted for in the caption.

        Denominator: number of verified graph nodes (distinct physical entities).
        Numerator: for each class, credit ``min(stated_quantity, verified_count)``
        when an explicit quantity appears; if the class is mentioned without an
        explicit quantity, credit all verified instances of that class.

        Returns None when no verified nodes exist — UI must show Unavailable.
        """
        from language.validation.caption_factuality import (
            _canonical_count_label,
            extract_caption_quantity_mentions,
        )

        by_label: dict[str, int] = {}
        for node in context.graph.nodes:
            lab = _canonical_count_label(node.label or "")
            if not lab:
                continue
            if lab in {"person", "people", "man", "woman", "child", "boy", "girl"}:
                lab = "person"
            by_label[lab] = by_label.get(lab, 0) + 1
        total = sum(by_label.values())
        if total <= 0:
            return None

        lower = text.lower()
        stated: dict[str, int] = {}
        for label, qty in extract_caption_quantity_mentions(text):
            if qty < 0:
                # Vague quantifier: do not over-credit; leave to mention handling.
                continue
            stated[label] = max(stated.get(label, 0), qty)

        covered = 0
        for label, verified_n in by_label.items():
            mentioned = self._label_in_text(label, lower)
            if label == "person":
                mentioned = mentioned or any(
                    tok in lower for tok in ("people", "person", "man", "woman", "child")
                )
            if not mentioned:
                continue
            if label in stated:
                covered += min(stated[label], verified_n)
            else:
                covered += verified_n
        return covered / total

    def _relationship_coverage(self, text: str, context: SceneContext) -> float | None:
        semantic = [
            relation
            for relation in context.graph.relations
            if relation.relation_type
            not in {"left_of", "right_of", "above", "below", "near", "far"}
        ]
        if not semantic:
            return None
        lower = text.lower()
        covered = 0
        for relation in semantic:
            phrase = relation.relation_type.replace("_", " ")
            if self._phrase_matches(lower, phrase, _RELATION_SYNONYMS.get(relation.relation_type, ())):
                covered += 1
        return covered / len(semantic)

    def _activity_coverage(self, text: str, context: SceneContext) -> float | None:
        activities = [
            item.activity.lower()
            for item in context.activities.activities
            if item.activity.lower() not in _WEAK_ACTIVITIES
        ]
        if not activities:
            return None
        lower = text.lower()
        covered = sum(
            1
            for activity in activities
            if self._phrase_matches(lower, activity, _ACTIVITY_SYNONYMS.get(activity, ()))
        )
        return covered / len(activities)

    def _context_coverage(self, text: str, context: SceneContext) -> float:
        env = context.environment
        markers = [
            env.scene_type,
            env.indoor_outdoor,
            env.social_context,
            env.weather,
            env.time_of_day,
        ]
        markers = [marker.lower() for marker in markers if marker and marker != "unknown"]
        if not markers:
            return 0.5
        lower = text.lower()
        covered = sum(1 for marker in markers if any(part in lower for part in marker.split()))
        return covered / len(markers)

    def _average(self, *values: float | None) -> float:
        present = [value for value in values if value is not None]
        if not present:
            return 0.0
        return sum(present) / len(present)

    def _label_in_text(self, label: str, lower_text: str) -> bool:
        if label in lower_text:
            return True
        if label == "person" and "people" in lower_text:
            return True
        if label.endswith("s") and label[:-1] in lower_text:
            return True
        return any(part in lower_text for part in label.split() if len(part) > 3)

    def _phrase_matches(self, lower_text: str, phrase: str, synonyms: tuple[str, ...]) -> bool:
        if phrase in lower_text:
            return True
        if any(token in lower_text for token in synonyms):
            return True
        # Honest coverage: "leading a horse" must match "leading the horse".
        norm_phrase = _strip_articles(phrase)
        norm_text = _strip_articles(lower_text)
        if norm_phrase and norm_phrase in norm_text:
            return True
        # Allow brief modifiers between tokens and light verb inflection:
        # "leading a horse" ↔ "leading a brown horse"
        # "holding a rope" ↔ "holds a rope"
        if _ordered_activity_tokens_match(norm_text, norm_phrase):
            return True
        return False


def _strip_articles(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\b(a|an|the)\b", " ", text)).strip()


def _verb_token_variants(token: str) -> tuple[str, ...]:
    """Deterministic light inflection set for activity verbs (no free synonym inventing)."""
    tok = (token or "").lower().strip()
    if not tok:
        return ()
    forms = {tok}
    if tok.endswith("ing") and len(tok) > 4:
        base = tok[:-3]  # holding→hold, leading→lead
        forms.update({base, f"{base}s", f"{base}ed"})
    elif tok.endswith("s") and len(tok) > 3 and not tok.endswith(("ss", "us", "is")):
        base = tok[:-1]
        forms.update({base, f"{base}ing", f"{base}ed"})
    elif len(tok) > 2:
        forms.update({f"{tok}s", f"{tok}ing", f"{tok}ed"})
    return tuple(sorted(forms))


def _ordered_activity_tokens_match(norm_text: str, norm_phrase: str) -> bool:
    """True when phrase content tokens appear in order, allowing up to 3 modifiers between."""
    tokens = [t for t in (norm_phrase or "").split() if t]
    if len(tokens) < 2:
        return bool(tokens) and tokens[0] in (norm_text or "")
    parts: list[str] = []
    for index, token in enumerate(tokens):
        alts = _verb_token_variants(token)
        parts.append("(?:" + "|".join(re.escape(a) for a in alts) + ")")
        if index < len(tokens) - 1:
            parts.append(r"(?:\s+\w+){0,3}\s+")
    return re.search("".join(parts), norm_text or "") is not None
