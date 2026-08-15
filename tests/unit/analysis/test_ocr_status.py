"""OCR status semantics: success / empty / unavailable."""

from __future__ import annotations

import numpy as np

from analysis.ocr.text_extractor import OcrExtraction, OcrExtractor


def test_ocr_status_unavailable_when_all_backends_none(monkeypatch) -> None:
    extractor = OcrExtractor()

    def _none(_pixels):
        return (), 0.0, "none"

    monkeypatch.setattr(extractor, "_try_easyocr", _none)
    monkeypatch.setattr(extractor, "_try_tesseract", _none)
    monkeypatch.setattr(extractor, "_try_paddleocr", _none)
    result = extractor.extract(np.zeros((32, 32, 3), dtype=np.uint8))
    assert result.status == "unavailable"
    assert result.texts == ()
    assert result.source == "none"
    assert result.ok is False


def test_ocr_status_empty_when_engine_ran(monkeypatch) -> None:
    extractor = OcrExtractor()

    def _empty(_pixels):
        return (), 0.0, "easyocr"

    monkeypatch.setattr(extractor, "_try_easyocr", _empty)
    result = extractor.extract(np.zeros((32, 32, 3), dtype=np.uint8))
    assert result.status == "empty"
    assert result.source == "easyocr"
    assert result.ok is True


def test_ocr_status_success(monkeypatch) -> None:
    extractor = OcrExtractor()

    def _ok(_pixels):
        return ("HELLO",), 0.9, "easyocr"

    monkeypatch.setattr(extractor, "_try_easyocr", _ok)
    result = extractor.extract(np.zeros((32, 32, 3), dtype=np.uint8))
    assert result.status == "success"
    assert result.texts == ("HELLO",)
    assert result.ok is True


def test_ocr_extraction_default_status_field() -> None:
    payload = OcrExtraction(texts=(), confidence=0.0, source="none", processing_time_ms=1.0)
    assert payload.status == "unavailable"
