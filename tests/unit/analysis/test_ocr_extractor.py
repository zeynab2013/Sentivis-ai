"""OCR extractor: empty when no text; extracts when text is present."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analysis.ocr.text_extractor import OcrExtractor


def _blank_image() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _text_image(*, size: int = 36, text: str = "SENTIVIS LAB") -> np.ndarray:
    img = Image.new("RGB", (480, 160), (20, 20, 24))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    draw.text((24, 48), text, fill=(245, 245, 245), font=font)
    return np.asarray(img, dtype=np.uint8)


def test_ocr_no_text_returns_empty_without_error() -> None:
    result = OcrExtractor().extract(_blank_image())
    assert result.texts == ()
    assert result.confidence == 0.0


def test_ocr_clear_text_extracts_meaningful_content() -> None:
    result = OcrExtractor().extract(_text_image(size=40, text="SENTIVIS LAB"))
    # Skip hard-fail only when literally no OCR backend can import.
    if result.source == "none":
        import importlib.util

        if importlib.util.find_spec("easyocr") is None:
            return
    joined = " ".join(result.texts).upper()
    assert result.texts, "OCR backend present but returned empty for clear text"
    assert any(token in joined for token in ("SENTIVIS", "LAB", "SENT", "VIS"))
    assert result.source in {"easyocr", "tesseract", "paddleocr"}


def test_ocr_small_text_does_not_crash() -> None:
    result = OcrExtractor().extract(_text_image(size=14, text="ROOM 204"))
    assert isinstance(result.texts, tuple)
    assert result.confidence >= 0.0
