"""Translate the canonical English caption without re-running vision analysis."""

from __future__ import annotations

import re

from analysis.activity.ollama_client import OllamaClient
from core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_CAPTION_LANGUAGES: tuple[str, ...] = ("en", "fa", "de", "es", "zh")

_LANGUAGE_NAMES = {
    "en": "English",
    "fa": "Persian",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
}


class CaptionTranslator:
    """Text-only caption translation. Never touches images or VLM perception."""

    def __init__(self, client: OllamaClient | None = None) -> None:
        self._client = client or OllamaClient(model="gemma3:4b", timeout_seconds=90.0)
        self._translation_count = 0

    @property
    def translation_count(self) -> int:
        return self._translation_count

    def reset_translation_count(self) -> None:
        self._translation_count = 0

    def translate(self, canonical_en: str, target_lang: str) -> str:
        """Return caption in target language. English is identity. Preserves detail."""
        text = " ".join((canonical_en or "").split()).strip()
        lang = (target_lang or "en").lower().strip()
        if not text:
            return ""
        if lang not in SUPPORTED_CAPTION_LANGUAGES:
            lang = "en"
        if lang == "en":
            return text

        translated = self._translate_via_ollama(text, lang)
        if translated and self._looks_translated(translated, lang):
            self._translation_count += 1
            logger.info(
                "Caption translated en→%s | translation_count=%d chars=%d",
                lang,
                self._translation_count,
                len(translated),
            )
            return translated

        # Deterministic fallback: keep canonical English rather than inventing a thin rewrite.
        logger.warning("Caption translation to %s unavailable — keeping English canonical", lang)
        return text

    def _translate_via_ollama(self, text: str, lang: str) -> str:
        language_name = _LANGUAGE_NAMES.get(lang, lang)
        system = (
            "You are a precise translator for image captions. "
            "Translate the caption into the requested language. "
            "Preserve every factual detail, object, color, action, relationship, "
            "environment cue, and visible text. "
            "Do not shorten. Do not omit details. "
            "Do not add emotions, mood, intentions, or speculation. "
            "Return only the translated caption paragraph."
        )
        user = f"Target language: {language_name}\n\nCaption:\n{text}"
        try:
            response = self._client.generate_text(system=system, user=user, max_tokens=400)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama caption translation failed: %s", exc)
            return ""
        out = " ".join((response.text or "").split()).strip()
        out = out.strip('"').strip("'").strip()
        # Reject empty / echo that clearly failed.
        if not out or out.lower() == text.lower():
            return ""
        return out

    @staticmethod
    def _looks_translated(text: str, lang: str) -> bool:
        if lang == "fa":
            return bool(re.search(r"[\u0600-\u06FF]", text))
        if lang == "zh":
            return bool(re.search(r"[\u4e00-\u9fff]", text))
        if lang in {"de", "es"}:
            # Must remain Latin script and not empty.
            return bool(re.search(r"[A-Za-zÀ-ÿÄÖÜäöüß]{3,}", text))
        return bool(text.strip())
