"""UI language localizes analysis terms; accessories remain evidence-gated."""

from __future__ import annotations

import os

from language.refinement.caption_refiner import (
    clear_ui_language_cache,
    localize_term,
    strip_unverified_accessories,
    ui_text,
)
from language.semantic.natural_caption_service import NaturalCaptionService
from tests.unit.language.test_natural_caption_service import (
    _StubVision,
    _context,
    _image,
    _understanding,
)


def test_ui_terms_localize_for_interface_labels() -> None:
    clear_ui_language_cache()
    assert localize_term("horse", language="fa") == "اسب"
    assert localize_term("office", language="es") == "oficina"
    assert ui_text("msg.no_objects", language="es") == "No se detectaron objetos."


def test_pipeline_caption_stays_canonical_english_regardless_of_ui_language() -> None:
    """Analysis emits English; UI language must not mutate the canonical caption."""
    os.environ["SENTIVIS_UI_LANGUAGE"] = "fa"
    clear_ui_language_cache()
    try:
        service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
        paragraph = service.generate(_image(), _understanding(), context=_context())
        assert not any("\u0600" <= ch <= "\u06FF" for ch in paragraph)
        assert "appears to" not in paragraph.lower()
        assert "photographed scene" not in paragraph.lower()
        assert "chair" in paragraph.lower() or "sit" in paragraph.lower() or "person" in paragraph.lower()
    finally:
        os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
        clear_ui_language_cache()


def test_caption_translator_produces_persian_from_canonical_english() -> None:
    from analysis.activity.ollama_client import OllamaResponse
    from language.localization.caption_translator import CaptionTranslator

    class _StubClient:
        def generate_text(self, *, system: str, user: str, max_tokens: int = 400):
            return OllamaResponse(
                text="یک شخص با هودی سرمه‌ای روی صندلی در دفتر نشسته است.",
                model="stub",
            )

    translator = CaptionTranslator(client=_StubClient())  # type: ignore[arg-type]
    out = translator.translate(
        "A person in a navy hoodie sits on a chair in an office.",
        "fa",
    )
    assert "شخص" in out
    assert "صندلی" in out or "دفتر" in out
    assert translator.translation_count == 1


def test_english_ui_never_emits_persian_caption() -> None:
    os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
    clear_ui_language_cache()
    service = NaturalCaptionService(_StubVision())  # type: ignore[arg-type]
    paragraph = service.generate(_image(), _understanding(), context=_context())
    assert not any("\u0600" <= ch <= "\u06FF" for ch in paragraph)
    assert "hoodie" in paragraph.lower() or "navy" in paragraph.lower()
    assert "chair" in paragraph.lower() or "sit" in paragraph.lower()


def test_strip_unverified_backpack() -> None:
    text = "A person wearing a backpack stands near a horse with a handbag."
    cleaned = strip_unverified_accessories(text, allowed_labels={"person", "horse"})
    assert "backpack" not in cleaned.lower()
    assert "handbag" not in cleaned.lower()
    assert "horse" in cleaned.lower()


def test_verified_backpack_kept() -> None:
    text = "A person wearing a backpack stands outdoors."
    cleaned = strip_unverified_accessories(text, allowed_labels={"person", "backpack"})
    assert "backpack" in cleaned.lower()
