"""Post-process, polish, and localize captions for the active UI language."""

from __future__ import annotations

import re
from functools import lru_cache

from core.contracts.analysis import SceneContext
from core.contracts.language import RawCaption, RefinedCaption
from core.logging import get_logger
from language.validation.caption_validator import CaptionEvidenceValidator

logger = get_logger(__name__)

SUPPORTED_UI_LANGUAGES: tuple[str, ...] = ("en", "fa", "de", "es", "zh")
_DEFAULT_UI_LANGUAGE = "en"

_ROBOTIC_REPLACEMENTS = (
    # Strip detector-inventory openers — do NOT rewrite them into "Nearby are".
    (r"\bNotable relationships include\b", ""),
    (r"\bNearby objects include\b", ""),
    (r"\bAround them are\b", ""),
    (r"\bThe scene includes\b", ""),
    (r"\bClose by are\b", ""),
    (r"\bClose by is\b", ""),
    (r"\bAlso visible are\b", ""),
    (r"\bAlso visible is\b", ""),
    (r"\bNearby are\b", ""),
    (r"\bNearby is\b", ""),
    (r"\bis close by\b", "is visible"),
    (r"\bare close by\b", "are visible"),
    (r"\bWhat stands out is\b", ""),
    (r"\bThe main focus is\b", ""),
    (r"\bOverall activity cues suggest\b", ""),
    (r"\bThe setting appears\b", "The location is"),
    (r"\bThe moment feels like\b", "The location is"),
    (r"\bappears to be wearing\b", "is wearing"),
    (r"\bappears to be\b", "is"),
    (r"\bseems to be\b", "is"),
    (r"\blikely\b", ""),
    (r"\bpossibly\b", ""),
    (r"\bdressed in dressed in\b", "dressed in"),
    (r"\bphotographed scene(?: setting)?\b", ""),
    (r"\bquiet observational detail\b", ""),
    (r"\bcalm and observational\b", ""),
    (r"\bcalm and lived-in\b", ""),
    (r"\bminimal interaction evidence\b", ""),
    (r"\bOne person\b", "A person"),
    # Keep mid-sentence "another person" natural — never force capitalized "A second person".
    (r"(?<=[.!?]\s)\banother person\b", "A second person"),
    (r"^\banother person\b", "A second person"),
    (r"\bSomeone nearby farther back\b", "a second person farther back"),
    (r"\bperson #(\d+)\b", r"someone"),
    (r"\bsits close enough to matter(?: to the action)?\b", "is nearby"),
    (r"\bstays? close to the main action\b", "is nearby"),
    (r"\bVerified evidence:?\b", ""),
    (r"\bconfidence(?: scores?)?\b", ""),
    (r"\blower clothing\b", "pants"),
    (r"\bupper clothing\b", "shirt"),
    (r"\bwearing backpack\b", "wearing a backpack"),
    (r"\bwearing hat\b", "wearing a hat"),
    (r"\bplays a clear role in the moment\b", ""),
    (r"\bbelongs among the defining details of the view\b", ""),
    (r"\bmatters in the moment\b", ""),
    (r"\bstands out in the view\b", ""),
    (r"\bThe activity centers on\b", ""),
    (r"\bThe key interaction is\b", ""),
    (r"\bThere is\b", ""),
    (r"\bThere are\b", ""),
)

_PHRASE_KEYS: tuple[tuple[str, str], ...] = (
    ("working at a computer", "term.working_at_a_computer"),
    ("standing beside a horse", "term.standing_beside_a_horse"),
    ("leading a horse", "term.leading_a_horse"),
    ("riding a horse", "term.riding_a_horse"),
    ("riding a bicycle", "term.riding_a_bicycle"),
    ("playing with a ball", "term.playing_with_a_ball"),
    ("playing soccer", "term.playing_soccer"),
    ("playing football", "term.playing_football"),
    ("playing tennis", "term.playing_tennis"),
    ("playing basketball", "term.playing_basketball"),
    ("walking a dog", "term.walking_a_dog"),
    ("using a phone", "term.using_a_phone"),
    ("sports ball", "term.sports_ball"),
    ("tennis racket", "term.tennis_racket"),
    ("dining table", "term.dining_table"),
    ("traffic light", "term.traffic_light"),
    ("stop sign", "term.stop_sign"),
    ("cell phone", "term.cell_phone"),
    ("shopping cart", "term.shopping_cart"),
    ("narrative caption unavailable", "msg.narrative_unavailable"),
    ("no objects detected", "msg.no_objects"),
    ("no relationships inferred", "msg.no_relationships"),
    ("no activities inferred", "msg.no_activities"),
    ("no attributes extracted", "msg.no_attributes"),
    ("no image quality report available", "msg.no_image_quality"),
    ("no dominant colors extracted", "msg.no_colors"),
    ("none detected", "msg.none_detected"),
    ("recovered via fallback", "msg.qa_recovered"),
    ("time of day", "label.time_of_day"),
    ("indoor/outdoor", "label.indoor_outdoor"),
    ("social context", "label.social_context"),
    ("crowd level", "label.crowd_level"),
    ("scene type", "label.scene_type"),
    ("overall quality", "label.overall_quality"),
    ("hallucination risk", "label.hallucination_risk"),
    ("evidence consistency", "label.evidence_consistency"),
    ("object coverage", "label.object_coverage"),
    ("relationship coverage", "label.relationship_coverage"),
    ("activity coverage", "label.activity_coverage"),
    ("caption quality", "label.caption_quality"),
    ("activity confidence", "label.activity_confidence"),
    ("competition mode", "label.competition_mode"),
    ("stage timings", "label.stage_timings"),
    ("enhancement applied", "label.enhancement_applied"),
    ("super resolution", "label.super_resolution"),
    ("executive summary", "label.executive_summary"),
    ("short caption", "label.short_caption"),
    ("full caption", "label.full_caption"),
)


@lru_cache(maxsize=8)
def _load_ui_catalog(language: str) -> dict[str, str]:
    from core.resources import load_translation_catalog

    return load_translation_catalog(language)


def active_ui_language() -> str:
    """UI language selector is the single source of truth for display language.

    Domain code must not import Streamlit/Qt. UI layers register a provider via
    ``core.config.ui_language.register_ui_language_provider``.
    Priority: SENTIVIS_UI_LANGUAGE → registered provider → English.
    """
    from core.config.ui_language import resolve_ui_language

    return resolve_ui_language()


def ui_text(key: str, default: str | None = None, *, language: str | None = None, **params: object) -> str:
    lang = (language or active_ui_language()).lower().strip()
    if lang not in SUPPORTED_UI_LANGUAGES:
        lang = _DEFAULT_UI_LANGUAGE
    catalog = _load_ui_catalog(lang)
    fallback = _load_ui_catalog(_DEFAULT_UI_LANGUAGE)
    text = catalog.get(key) or fallback.get(key) or default or key
    if params:
        try:
            return text.format(**params)
        except (KeyError, ValueError):
            return text
    return text


def localize_term(term: str, *, language: str | None = None) -> str:
    raw = (term or "").strip()
    if not raw:
        return raw
    lang = (language or active_ui_language()).lower().strip()
    if lang == _DEFAULT_UI_LANGUAGE:
        return raw.replace("_", " ")
    normalized = raw.replace("_", " ").strip().lower()
    key = f"term.{normalized.replace(' ', '_')}"
    translated = ui_text(key, default="", language=lang)
    if translated and translated != key:
        return translated
    parts = normalized.split()
    if len(parts) > 1:
        mapped = [localize_term(part, language=lang) for part in parts]
        if any(m != p for m, p in zip(mapped, parts, strict=False)):
            return " ".join(mapped)
    return raw.replace("_", " ")


_STRICT_ACCESSORIES = frozenset(
    {
        "backpack",
        "handbag",
        "suitcase",
        "umbrella",
        "tie",
        "cell phone",
        "remote",
        "mouse",
        "toothbrush",
        "hair drier",
        "glasses",
        "sunglasses",
    }
)


def localize_prose(text: str, *, language: str | None = None) -> str:
    """Localize structured UI/report labels only.

    AI image captions are never passed through this helper — the vision/NLG
    caption stays in the model output language regardless of UI language.
    """
    if not text:
        return text
    lang = (language or active_ui_language()).lower().strip()
    if lang not in SUPPORTED_UI_LANGUAGES:
        lang = _DEFAULT_UI_LANGUAGE
    if lang == _DEFAULT_UI_LANGUAGE:
        return text

    # Only rewrite known report/status label phrases — not free narrative prose.
    updated = text
    for english, key in sorted(_PHRASE_KEYS, key=lambda item: len(item[0]), reverse=True):
        if english in {
            "working at a computer",
            "standing beside a horse",
            "leading a horse",
            "riding a horse",
            "riding a bicycle",
            "playing with a ball",
            "playing soccer",
            "playing football",
            "playing tennis",
            "playing basketball",
            "walking a dog",
            "using a phone",
            "sports ball",
            "tennis racket",
            "dining table",
            "traffic light",
            "stop sign",
            "cell phone",
            "shopping cart",
        }:
            # Activity/object phrases belong to native caption rendering, not prose mix.
            continue
        replacement = ui_text(key, default=english, language=lang)
        if replacement != english:
            updated = re.sub(re.escape(english), replacement, updated, flags=re.IGNORECASE)

    for english, key in (
        ("Objects", "label.objects"),
        ("Dominant", "label.dominant"),
        ("Environment", "label.environment"),
        ("Activities", "label.activities"),
        ("Relationships", "label.relationships"),
        ("Attributes", "label.attributes"),
        ("Setting", "label.setting"),
        ("Weather", "label.weather"),
        ("Complexity", "label.complexity"),
        ("Atmosphere", "label.atmosphere"),
        ("Resolution", "label.resolution"),
        ("Brightness", "label.brightness"),
        ("Contrast", "label.contrast"),
        ("Sharpness", "label.sharpness"),
        ("Operations", "label.operations"),
        ("Notes", "label.notes"),
        ("Passed", "msg.qa_passed"),
        ("Yes", "msg.yes"),
        ("No", "msg.no"),
        ("None", "msg.none"),
        ("Nodes", "label.nodes"),
        ("Relations", "label.relations"),
    ):
        replacement = ui_text(key, default=english, language=lang)
        if replacement != english:
            updated = re.sub(rf"\b{re.escape(english)}\b", replacement, updated)
    return updated


def strip_unverified_accessories(text: str, allowed_labels: set[str]) -> str:
    """Remove accessory mentions that lack strong detection evidence."""
    if not text:
        return text
    allowed = {label.lower() for label in allowed_labels}
    updated = text
    for accessory in sorted(_STRICT_ACCESSORIES, key=len, reverse=True):
        if accessory in allowed:
            continue
        updated = re.sub(
            rf"\b(?:wearing|carrying|holding|with)\s+(?:a|an|the)\s+{re.escape(accessory)}\b",
            "",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            rf"\b(?:a|an|the)\s+{re.escape(accessory)}\b",
            "",
            updated,
            flags=re.IGNORECASE,
        )
        updated = re.sub(rf"\b{re.escape(accessory)}\b", "", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\s{2,}", " ", updated)
    updated = re.sub(r"\s+([,.;:!?])", r"\1", updated)
    updated = re.sub(r"\s+and\s+and\b", " and ", updated, flags=re.IGNORECASE)
    # Drop orphan sentences created by accessory removal ("Is also visible.").
    kept: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", updated.strip()):
        sentence = part.strip()
        if not sentence:
            continue
        if re.match(r"^(?:Is|Are|Was|Were)\s+also\s+visible\.?$", sentence, re.IGNORECASE):
            continue
        if re.match(r"^(?:A|An|The)\s+is\s+also\s+visible\.?$", sentence, re.IGNORECASE):
            continue
        kept.append(sentence if sentence.endswith((".", "!", "?")) else sentence + ".")
    return " ".join(kept).strip()


def clear_ui_language_cache() -> None:
    _load_ui_catalog.cache_clear()


class CaptionRefiner:
    """Refines raw model captions for grammar and fluency without changing facts."""

    def __init__(self) -> None:
        self._validator = CaptionEvidenceValidator()

    def refine(
        self,
        primary: RawCaption,
        fallback: RawCaption | None,
        context: SceneContext,
    ) -> RefinedCaption:
        text = primary.text.strip()
        if not text and fallback:
            text = fallback.text.strip()

        text = self._validator.filter_unsupported_sentences(text, context)
        allowed = {node.label.lower() for node in context.graph.nodes}
        before = text
        text = strip_unverified_accessories(text, allowed)
        text = self.humanize(text)
        text = self._normalize_whitespace(text)
        text = self._fix_grammar(text)
        text = self._ensure_sentence_end(text)
        text = self._dedupe_phrases(text)
        # Never let refine destroy a rich caption into a short stub.
        if len(before.split()) >= 40 and len(text.split()) < int(len(before.split()) * 0.65):
            logger.warning(
                "Caption refine collapsed detail (%d→%d words); keeping richer original",
                len(before.split()),
                len(text.split()),
            )
            text = self._fix_grammar(self._normalize_whitespace(before))
            text = self._ensure_sentence_end(text)
        logger.info("Caption refine before=%s", before[:180])
        logger.info("Caption refine after=%s", text[:180])

        sources: list[str] = [primary.source]
        if fallback and fallback.text.strip():
            sources.append(fallback.source)

        logger.info("Caption refined from sources: %s", sources)
        return RefinedCaption(text=text, sources=tuple(dict.fromkeys(sources)))

    def polish(self, text: str) -> str:
        """Humanize and fix grammar without SceneContext vocabulary filtering."""
        updated = self.humanize(text)
        updated = self._normalize_whitespace(updated)
        updated = self._fix_grammar(updated)
        updated = self._ensure_sentence_end(updated)
        return self._dedupe_phrases(updated)

    def refine_evidence_caption(
        self,
        text: str,
        *,
        allowed_labels: set[str],
    ) -> str:
        """Drop unsupported accessories and keep one fluent English paragraph."""
        before = text
        updated = strip_unverified_accessories(text, allowed_labels)
        updated = self.polish(updated)
        if len(before.split()) >= 40 and len(updated.split()) < int(len(before.split()) * 0.65):
            logger.warning(
                "Evidence caption polish collapsed detail (%d→%d); keeping richer text",
                len(before.split()),
                len(updated.split()),
            )
            updated = before
        logger.info("Evidence caption before=%s", before[:180])
        logger.info("Evidence caption after=%s", updated[:180])
        return updated

    def humanize(self, text: str) -> str:
        """Remove robotic detector phrasing while preserving meaning."""
        if not text:
            return text
        from language.refinement.caption_sanity import fix_double_articles, strip_spatial_filler

        updated = text
        for pattern, replacement in _ROBOTIC_REPLACEMENTS:
            updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
        updated = strip_spatial_filler(updated)
        updated = fix_double_articles(updated)
        updated = re.sub(r"\s{2,}", " ", updated)
        updated = re.sub(r"\s+([,.;:!?])", r"\1", updated)
        return updated.strip()

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _fix_grammar(self, text: str) -> str:
        if not text:
            return text
        from language.refinement.caption_sanity import fix_double_articles

        text = text[0].upper() + text[1:]
        text = re.sub(r"\bi\b", "I", text)
        text = fix_double_articles(text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\.{2,}", ".", text)
        return text

    def _ensure_sentence_end(self, text: str) -> str:
        if text and text[-1] not in ".!?":
            return f"{text}."
        return text

    def _dedupe_phrases(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        kept: list[str] = []
        kept_tokens: list[set[str]] = []
        for sentence in sentences:
            key = re.sub(r"\W+", " ", sentence.lower()).strip()
            if not key:
                continue
            tokens = {t for t in key.split() if len(t) >= 3}
            if any(len(tokens & prior) / max(1, len(tokens | prior)) >= 0.72 for prior in kept_tokens):
                continue
            kept.append(sentence.strip())
            kept_tokens.append(tokens)
        return " ".join(kept)
