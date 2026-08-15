"""Caption translator is text-only and language-scoped."""

from __future__ import annotations

from analysis.activity.ollama_client import OllamaResponse
from language.localization.caption_translator import CaptionTranslator, SUPPORTED_CAPTION_LANGUAGES


def test_supported_languages_are_exactly_five() -> None:
    assert SUPPORTED_CAPTION_LANGUAGES == ("en", "fa", "de", "es", "zh")


def test_english_is_identity() -> None:
    translator = CaptionTranslator(client=object())  # type: ignore[arg-type]
    text = "A person stands beside a desk in an office."
    assert translator.translate(text, "en") == text
    assert translator.translation_count == 0


def test_german_translation_path() -> None:
    class _StubClient:
        def generate_text(self, *, system: str, user: str, max_tokens: int = 400):
            return OllamaResponse(
                text="Eine Person steht neben einem Schreibtisch in einem Büro.",
                model="stub",
            )

    translator = CaptionTranslator(client=_StubClient())  # type: ignore[arg-type]
    out = translator.translate("A person stands beside a desk in an office.", "de")
    assert "Person" in out
    assert "Schreibtisch" in out or "Büro" in out
    assert translator.translation_count == 1
