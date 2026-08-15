"""Internal scene reasoning contracts (never shown directly in UI)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceFact:
    """One grounded fact with confidence and provenance."""

    subject: str
    predicate: str
    value: str
    confidence: float
    source: str
    processing_time_ms: float = 0.0

    def as_clause(self) -> str:
        if self.predicate in {"is", "appears"}:
            return f"{self.subject} {self.predicate} {self.value}"
        if self.predicate in {
            "wearing",
            "holding",
            "sitting_on",
            "beside",
            "looking_at",
            "leading",
            "riding",
            "carrying",
            "using",
            "talking_to",
            "guiding",
            "playing_with",
        }:
            return f"{self.subject} {self.predicate.replace('_', ' ')} {self.value}"
        return f"{self.subject}: {self.predicate}={self.value}"


@dataclass(frozen=True)
class SceneUnderstanding:
    """Merged semantic understanding produced by SceneReasoner (internal only)."""

    facts: tuple[EvidenceFact, ...]
    ranked_subjects: tuple[str, ...]
    environment_keys: tuple[str, ...]
    activity_keys: tuple[str, ...]
    ocr_text: tuple[str, ...]
    evidence_brief: str
    overall_confidence: float
    discarded_count: int
    contradictions_resolved: int
