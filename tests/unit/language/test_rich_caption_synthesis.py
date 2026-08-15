"""Rich-scene fallback captions must synthesize verified evidence densely."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

from core.contracts.image import PreprocessedImage, ValidatedImage
from core.contracts.language import RawCaption
from language.refinement.caption_refiner import clear_ui_language_cache
from language.semantic.natural_caption_service import NaturalCaptionService
from tests.unit.language.test_caption_quality_overhaul import (
    _context,
    _image,
    _multi_person_understanding,
)
from tests.unit.language.test_indoor_person_retention import (
    _indoor_context,
    _indoor_understanding,
)


@pytest.fixture(autouse=True)
def _force_english_ui_language() -> None:
    os.environ["SENTIVIS_UI_LANGUAGE"] = "en"
    clear_ui_language_cache()
    yield
    clear_ui_language_cache()


class _QuietVision:
    def narrate(self, image: object, understanding: object) -> RawCaption:
        return RawCaption(text="", source="stub", confidence=0.0)


class _StubVision:
    def narrate(self, image: object, understanding: object) -> RawCaption:
        return RawCaption(text="indoor room", source="stub", confidence=0.4)


def test_multi_person_fallback_synthesizes_rich_grounded_paragraph() -> None:
    caption = NaturalCaptionService(_QuietVision()).generate(  # type: ignore[arg-type]
        _image(), _multi_person_understanding(), context=_context()
    )
    lower = caption.lower()
    assert "a person talking to a person" not in lower
    assert "looking at another person" not in lower
    assert "farther back in the frame" not in lower
    assert len(caption.split()) >= 15
    assert any(tok in lower for tok in ("room", "indoor", "chair", "table"))
    assert "another person" in lower or "both people" in lower
    assert lower.count("visible nearby") == 0
    assert lower.count("are visible nearby") == 0
    # No semantic restatement of the same people/object/depth facts.
    assert lower.count("farther back") <= 1
    assert lower.count("dining table") <= 1
    assert "both people share" not in lower
    assert lower.count("chair") <= 2


def test_indoor_kitchen_fallback_integrates_verified_fixtures() -> None:
    caption = NaturalCaptionService(_StubVision()).generate(  # type: ignore[arg-type]
        _image(), _indoor_understanding(), context=_indoor_context()
    )
    lower = caption.lower()
    assert any(tok in lower for tok in ("person", "prepar", "kitchen", "food"))
    assert "dining table" in lower
    assert "brown" in lower or "chair" in lower
    assert "refrigerator" in lower
    assert "tv" in lower
    assert "vase" in lower
    assert len(caption.split()) >= 28
    assert "fill out the surrounding space" not in lower
    assert lower.count("are visible nearby") == 0
    # Objects and setting introduced once — no inventory restatement.
    assert lower.count("dining table") <= 1
    assert lower.count("kitchen") <= 2
    assert "remains visible deeper" not in lower
    assert "anchors the central" not in lower
