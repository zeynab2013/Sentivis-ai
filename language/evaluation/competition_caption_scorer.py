"""Score caption candidates with human readability as the primary NLG target."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.contracts.analysis import SceneContext
from core.contracts.language import CaptionQualityReport
from core.contracts.reasoning import SceneUnderstanding
from language.evaluation.caption_quality_evaluator import CaptionQualityEvaluator


@dataclass(frozen=True)
class CandidateScore:
    """Composite score for one caption candidate."""

    text: str
    overall: float
    object_coverage: float
    clothing_accuracy: float
    color_accuracy: float
    relationship_accuracy: float
    ocr_consistency: float
    hallucination_rate: float
    grammar: float
    fluency: float
    human_readability: float
    source: str


class CompetitionCaptionScorer:
    """Rank candidates — readability first, without sacrificing factual accuracy."""

    def __init__(self, evaluator: CaptionQualityEvaluator | None = None) -> None:
        self._evaluator = evaluator or CaptionQualityEvaluator()

    def score(
        self,
        caption: str,
        context: SceneContext,
        understanding: SceneUnderstanding,
        *,
        source: str = "candidate",
    ) -> CandidateScore:
        text = caption.strip()
        base: CaptionQualityReport = self._evaluator.evaluate(text, context)
        clothing = self._clothing_accuracy(text, understanding)
        color = self._color_accuracy(text, understanding)
        ocr = self._ocr_consistency(text, understanding)
        readability = self._human_readability(text)
        robotic_penalty = self._robotic_penalty(text)
        words = len(text.split())
        # Do not punish long, information-rich captions — only true stubs / runaway text.
        if words < 28:
            length_penalty = 0.18
        elif words < 45:
            length_penalty = 0.06
        elif words > 200:
            length_penalty = 0.04
        else:
            length_penalty = 0.0
        # N/A coverages must not crash scoring — treat as neutral, not perfect.
        object_cov = 0.5 if base.object_coverage is None else base.object_coverage
        relationship_cov = 0.5 if base.relationship_coverage is None else base.relationship_coverage
        hall_risk = 0.0 if base.hallucination_risk is None else base.hallucination_risk
        factual = 1.0 - hall_risk
        evidence_coverage = self._evidence_coverage(text, understanding)
        recall = self._clamp(
            0.35 * object_cov + 0.25 * clothing + 0.2 * color + 0.2 * evidence_coverage
        )
        precision = factual
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        narrative_bonus = 0.04 if source.startswith("narrative_") else 0.0
        if source == "narrative_complete":
            narrative_bonus += 0.06
        if source == "vlm_narrate":
            narrative_bonus += 0.10  # image-first preference
        if words >= 55:
            narrative_bonus += 0.06
        elif words >= 35:
            narrative_bonus += 0.03
        # Information preservation first; fluency second. Never optimize for thin captions.
        overall = self._clamp(
            0.22 * evidence_coverage
            + 0.14 * object_cov
            + 0.12 * relationship_cov
            + 0.12 * factual
            + 0.10 * readability
            + 0.08 * clothing
            + 0.06 * color
            + 0.06 * f1
            + 0.05 * base.fluency_score
            + 0.05 * ocr
            + narrative_bonus
            - robotic_penalty
            - length_penalty
        )
        # Never promote a fluent hallucination over a factual caption.
        if hall_risk >= 0.5:
            overall = min(overall, 0.45)
        return CandidateScore(
            text=text,
            overall=overall,
            object_coverage=object_cov,
            clothing_accuracy=clothing,
            color_accuracy=color,
            relationship_accuracy=relationship_cov,
            ocr_consistency=ocr,
            hallucination_rate=hall_risk,
            grammar=base.grammar_score,
            fluency=base.fluency_score,
            human_readability=readability,
            source=source,
        )

    def rank(
        self,
        candidates: list[tuple[str, str]],
        context: SceneContext,
        understanding: SceneUnderstanding,
    ) -> list[CandidateScore]:
        scored = [
            self.score(text, context, understanding, source=source)
            for text, source in candidates
            if text.strip()
        ]
        scored.sort(key=lambda item: (-item.overall, -item.human_readability, -item.object_coverage))
        return scored

    def _evidence_coverage(self, text: str, understanding: SceneUnderstanding) -> float:
        lower = text.lower()
        values = [
            fact.value.lower().replace("_", " ")
            for fact in understanding.facts
            if fact.confidence >= 0.55
            and fact.subject != "vlm"
            and fact.predicate != "is"
            and fact.value not in {"unknown", "unlikely", "none detected", "not_applicable", "possible"}
        ]
        if not values:
            return 0.5
        sample = values[:12]
        hits = sum(1 for value in sample if value in lower or value.split()[0] in lower)
        return self._clamp(hits / max(1, len(sample)))

    def _clothing_accuracy(self, text: str, understanding: SceneUnderstanding) -> float:
        lower = text.lower()
        expected = [
            f.value.lower().replace("_", " ")
            for f in understanding.facts
            if f.predicate in {"clothing_type", "footwear_type", "sleeve_length"}
            and f.confidence >= 0.55
            and f.value not in {"unknown", "unlikely", "none detected"}
        ]
        if not expected:
            return 0.55 if any(w in lower for w in ("wearing", "dressed", "outfit")) else 0.45
        hits = sum(1 for value in expected if value in lower or value.split()[0] in lower)
        return self._clamp(hits / max(1, len(expected)))

    def _color_accuracy(self, text: str, understanding: SceneUnderstanding) -> float:
        lower = text.lower()
        expected = [
            f.value.lower()
            for f in understanding.facts
            if (
                (f.predicate.endswith("_color") or f.predicate in {"dominant_color", "secondary_color"})
                and f.confidence >= 0.55
                and f.value not in {"unknown", "unlikely"}
            )
        ]
        if not expected:
            return 0.5
        hits = sum(1 for value in expected if value in lower)
        return self._clamp(hits / max(1, min(6, len(expected))))

    def _ocr_consistency(self, text: str, understanding: SceneUnderstanding) -> float:
        if not understanding.ocr_text:
            return 0.6
        lower = text.lower()
        hits = sum(1 for item in understanding.ocr_text[:3] if item.lower() in lower)
        return self._clamp(0.4 + 0.2 * hits)

    def _human_readability(self, text: str) -> float:
        if not text:
            return 0.0
        lower = text.lower().strip()
        words = text.split()
        score = 0.35
        # Prefer information-rich human paragraphs over thin stubs.
        if 70 <= len(words) <= 170:
            score += 0.30
        elif 50 <= len(words) < 70 or 170 < len(words) <= 200:
            score += 0.18
        elif 35 <= len(words) < 50:
            score += 0.10
        elif 22 <= len(words) < 35:
            score += 0.04
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if 3 <= len(sentences) <= 7:
            score += 0.12
        elif 2 <= len(sentences) <= 8:
            score += 0.08
        # Who/what/where opening (not clothing/color/list).
        first = sentences[0].lower() if sentences else ""
        if first.startswith(
            (
                "a person",
                "a man",
                "a woman",
                "a child",
                "in this",
                "in a",
                "on a",
                "on the",
                "in the",
                "a dark",
                "a white",
                "a black",
                "a brown",
            )
        ) or any(
            cue in first
            for cue in (
                " sits",
                " stands",
                " plays",
                " holds",
                " wearing",
                " leading",
                " dominates",
                " defines",
                " anchors",
            )
        ):
            score += 0.12
        if re.search(r"\bpeople are visible\b", first) or first.startswith(
            ("two people are visible", "the image shows", "the image depicts")
        ):
            score -= 0.18
        if first.startswith(
            (
                "one person appears",
                "outfit details",
                "around them",
                "the scene includes",
                "nearby is",
                "nearby are",
                "close by",
                "the person is wearing",
            )
        ):
            score -= 0.25
        if any(p in lower for p in ("appears to", "seems to", "there is ", "there are ")):
            score -= 0.15
        # Clothing woven into the subject clause beats a bolted-on wear sentence.
        if re.search(r"\b(person|man|woman|child)\b.+\bwearing\b", first):
            score += 0.08
        elif "wearing" in lower or "outfit details" in lower:
            score += 0.04
        if re.search(r"\b(person #\d+|object #\d+)\b", lower):
            score -= 0.2
        if text.count(";") <= 1 and "\n" not in text:
            score += 0.06
        if any(
            p in lower
            for p in ("around the subject", "complete the arrangement", "shares the frame", "just beyond")
        ):
            score += 0.05
        return self._clamp(score)

    def _robotic_penalty(self, text: str) -> float:
        lower = text.lower()
        markers = (
            "notable relationships include",
            "nearby objects include",
            "overall activity cues suggest",
            "the scene includes",
            "around them are",
            "one person appears to be wearing",
            "appears to be wearing",
            "the moment feels like",
            "person #",
            "verified evidence",
            "confidence",
            "they appear to be",
            "is present in the scene",
            "rests nearby",
            "overall, the scene reads",
            "nearby are",
            "nearby is",
            "also visible are",
            "are part of the scene",
            "is part of the scene",
        )
        hits = sum(1 for marker in markers if marker in lower)
        return min(0.35, 0.08 * hits)

    def _clamp(self, value: float) -> float:
        return float(max(0.0, min(1.0, value)))
