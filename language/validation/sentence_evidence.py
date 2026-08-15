"""Sentence-level evidence scoring for generated captions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.contracts.analysis import SceneContext
from language.validation.caption_validator import CaptionEvidenceValidator

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SentenceEvidence:
    """Evidence metadata for one caption sentence."""

    sentence: str
    confidence: float
    sources: tuple[str, ...]


class SentenceEvidenceAnalyzer:
    """Score each sentence against verified pipeline evidence."""

    def __init__(self) -> None:
        self._validator = CaptionEvidenceValidator()

    def analyze(self, caption: str, context: SceneContext) -> tuple[SentenceEvidence, ...]:
        supported = self._validator.filter_unsupported_sentences(caption, context)
        sentences = [part.strip() for part in _SENTENCE_SPLIT.split(supported) if part.strip()]
        results: list[SentenceEvidence] = []
        for sentence in sentences:
            sources = self._sources_for_sentence(sentence, context)
            confidence = self._confidence_for(sentence, context, sources)
            results.append(SentenceEvidence(sentence=sentence, confidence=confidence, sources=sources))
        return tuple(results)

    def filter_supported(self, caption: str, context: SceneContext) -> str:
        """Remove unsupported sentences and return validated caption text."""
        analyzed = self.analyze(caption, context)
        kept = [item.sentence for item in analyzed if item.confidence >= 0.45]
        return " ".join(kept)

    def _sources_for_sentence(self, sentence: str, context: SceneContext) -> tuple[str, ...]:
        lower = sentence.lower()
        sources: list[str] = []
        if any(node.label.lower() in lower for node in context.graph.nodes):
            sources.append("YOLO")
        if any(
            relation.relation_type.replace("_", " ") in lower
            for relation in context.graph.relations
        ):
            sources.append("Scene Graph")
        if any(item.activity.lower() in lower for item in context.activities.activities):
            sources.append("Activity")
        env = context.environment
        env_tokens = {
            env.setting,
            env.weather,
            env.time_of_day,
            env.indoor_outdoor,
            env.scene_type,
        }
        if any(token and token.lower() in lower for token in env_tokens):
            sources.append("Environment")
        for attribute in context.attributes.attributes:
            if attribute.value.lower() in lower or attribute.name.replace("_", " ") in lower:
                sources.append("Attributes")
                break
        if not sources:
            sources.append("Template")
        return tuple(dict.fromkeys(sources))

    def _confidence_for(self, sentence: str, context: SceneContext, sources: tuple[str, ...]) -> float:
        unsupported = self._validator.unsupported_object_tokens(sentence, context)
        base = 0.82 if not unsupported else max(0.35, 0.82 - 0.18 * len(unsupported))
        if "Scene Graph" in sources:
            base += 0.05
        if "Activity" in sources:
            base += 0.04
        if "Attributes" in sources:
            base += 0.03
        return min(0.98, base)
