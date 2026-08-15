"""Language pipeline DTOs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RawCaption:
    """Caption produced by a vision-language or reasoning model."""

    text: str
    source: str
    confidence: float


@dataclass(frozen=True)
class Prompt:
    """Structured prompt for reasoning model."""

    system: str
    user: str


@dataclass(frozen=True)
class VisualObservations:
    """BLIP visual description output — observations only, not final semantics."""

    observations: tuple[str, ...]
    object_attributes: tuple[str, ...]
    candidate_descriptions: tuple[str, ...]
    confidence: float
    raw_caption: RawCaption


@dataclass(frozen=True)
class RefinedCaption:
    """Final polished caption for presentation.

    canonical_caption_en is the immutable English source of truth.
    Display language variants are derived via translation — never by re-analysis.
    """

    text: str
    sources: tuple[str, ...]
    narrative_full: str = ""
    narrative_short: str = ""
    executive_summary: str = ""
    canonical_caption_en: str = ""


@dataclass(frozen=True)
class CaptionQualityReport:
    """Internal quality evaluation accompanying the final caption.

    Coverage fields use None when the denominator is empty (N/A — not 100%).
    hallucination_risk uses None when risk cannot be measured meaningfully.
    """

    grammar_score: float
    fluency_score: float
    evidence_consistency: float
    object_coverage: float | None
    relationship_coverage: float | None
    activity_coverage: float | None
    context_coverage: float
    hallucination_risk: float | None
    overall_quality: float
    notes: tuple[str, ...]
