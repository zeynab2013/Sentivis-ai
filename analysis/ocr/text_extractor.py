"""OCR with graceful offline fallback (EasyOCR → Tesseract → PaddleOCR)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OcrExtraction:
    """OCR result with provenance and explicit status.

    ``status`` distinguishes:
    - success: text extracted
    - empty: an engine ran but found no readable text
    - unavailable: no OCR backend could be initialized
    - error: reserved for hard failures (currently mapped via unavailable)
    """

    texts: tuple[str, ...]
    confidence: float
    source: str
    processing_time_ms: float
    status: str = "unavailable"

    @property
    def ok(self) -> bool:
        return self.status in {"success", "empty"}


class OcrExtractor:
    """Extract visible text when an OCR backend is available."""

    _paddle_reader: object | None = None
    _easyocr_reader: object | None = None
    _paddle_failed: bool = False

    def extract(self, pixels: NDArray[np.uint8]) -> OcrExtraction:
        started = time.perf_counter()
        # EasyOCR is the reliable local path on this Windows/CPU stack.
        texts, confidence, source = self._try_easyocr(pixels)
        if not texts and source == "none":
            texts, confidence, source = self._try_tesseract(pixels)
        if not texts and source == "none":
            texts, confidence, source = self._try_paddleocr(pixels)

        # Orientation retry only when primary pass is empty/weak/fragmented.
        if self._needs_orientation_retry(texts, confidence):
            best_texts, best_conf, best_source = texts, confidence, source
            for k in (2, 1, 3):  # 180°, 90°, 270°
                rotated = np.rot90(pixels, k=k)
                cand_texts, cand_conf, cand_source = self._try_easyocr(rotated)
                if not cand_texts and cand_source == "none":
                    cand_texts, cand_conf, cand_source = self._try_tesseract(rotated)
                if not cand_texts and cand_source == "none":
                    continue
                if self._ocr_result_score(cand_texts, cand_conf) > self._ocr_result_score(
                    best_texts, best_conf
                ):
                    best_texts, best_conf, best_source = cand_texts, cand_conf, cand_source
            texts, confidence, source = best_texts, best_conf, best_source

        texts = self._dedupe_ocr_texts(texts)
        elapsed = (time.perf_counter() - started) * 1000.0
        if texts:
            status = "success"
            logger.info("OCR extracted %d fragments via %s", len(texts), source)
        elif source != "none":
            status = "empty"
            logger.info("OCR found no text via %s (empty result is valid)", source)
        else:
            status = "unavailable"
            logger.warning(
                "OCR unavailable: EasyOCR/Tesseract/PaddleOCR could not run "
                "(missing dependency, executable, or model)"
            )
        return OcrExtraction(
            texts=texts,
            confidence=confidence,
            source=source,
            processing_time_ms=elapsed,
            status=status,
        )

    @staticmethod
    def _ocr_result_score(texts: tuple[str, ...], confidence: float) -> float:
        if not texts:
            return 0.0
        longest = max(len(t.strip()) for t in texts)
        # Prefer coherent words over many short fragments.
        frag_penalty = 0.15 * sum(1 for t in texts if len(t.strip()) <= 2)
        return float(confidence) + 0.08 * longest - frag_penalty

    @staticmethod
    def _needs_orientation_retry(texts: tuple[str, ...], confidence: float) -> bool:
        if not texts:
            return True
        if confidence < 0.52:
            return True
        # Letter soup / tiny fragments often mean the page is rotated.
        if len(texts) >= 2 and all(len(t.strip()) <= 2 for t in texts):
            return True
        return False

    @staticmethod
    def _dedupe_ocr_texts(texts: tuple[str, ...]) -> tuple[str, ...]:
        """Deduplicate OCR tokens; never treat repeated tokens as extra objects."""
        if not texts:
            return ()
        out: list[str] = []
        seen: set[str] = set()
        for raw in texts:
            cleaned = " ".join(str(raw).split()).strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
        # If a full word exists, drop single-letter fragments that are just its letters.
        longer = [t for t in out if len(t.strip()) >= 3]
        if longer:
            letter_pool = {ch for word in longer for ch in word.upper() if ch.isalpha()}
            out = [
                t
                for t in out
                if len(t.strip()) >= 2
                or t.strip().upper() not in letter_pool
            ]
        # Cap fragments; prefer unique readable strings.
        return tuple(out[:12])

    def _try_easyocr(self, pixels: NDArray[np.uint8]) -> tuple[tuple[str, ...], float, str]:
        try:
            import easyocr

            if OcrExtractor._easyocr_reader is None:
                OcrExtractor._easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            reader = OcrExtractor._easyocr_reader
            assert reader is not None
            results = reader.readtext(pixels)
            texts: list[str] = []
            scores: list[float] = []
            for item in results:
                if len(item) < 3:
                    continue
                text = str(item[1]).strip()
                score = float(item[2])
                if text and score >= 0.45 and len(text) >= 2:
                    texts.append(text)
                    scores.append(score)
            if not texts:
                return (), 0.0, "easyocr"
            return tuple(texts[:12]), float(sum(scores) / len(scores)), "easyocr"
        except Exception as exc:  # noqa: BLE001
            logger.debug("EasyOCR unavailable: %s", exc)
            return (), 0.0, "none"

    def _try_tesseract(self, pixels: NDArray[np.uint8]) -> tuple[tuple[str, ...], float, str]:
        try:
            import shutil

            import pytesseract
            from PIL import Image

            if shutil.which("tesseract") is None:
                return (), 0.0, "none"
            image = Image.fromarray(pixels)
            text = pytesseract.image_to_string(image)
            fragments = tuple(
                part.strip()
                for part in text.replace("\n", " ").split(" ")
                if len(part.strip()) >= 3
            )
            joined = " ".join(fragments)
            if len(joined) < 3:
                return (), 0.0, "tesseract"
            return (joined[:120],), 0.5, "tesseract"
        except Exception:  # noqa: BLE001
            return (), 0.0, "none"

    def _try_paddleocr(self, pixels: NDArray[np.uint8]) -> tuple[tuple[str, ...], float, str]:
        """Fallback when EasyOCR/Tesseract are absent. May be unavailable on some CPU builds."""
        if OcrExtractor._paddle_failed:
            return (), 0.0, "none"
        try:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            if OcrExtractor._paddle_reader is None:
                from paddleocr import PaddleOCR  # type: ignore[import-not-found]

                # New paddleocr API rejects use_gpu; older builds need use_angle_cls.
                try:
                    OcrExtractor._paddle_reader = PaddleOCR(
                        lang="en",
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                    )
                except TypeError:
                    OcrExtractor._paddle_reader = PaddleOCR(lang="en")
            reader = OcrExtractor._paddle_reader
            assert reader is not None
            if hasattr(reader, "predict"):
                results = reader.predict(pixels)
                return self._parse_paddle_predict(results)
            results = reader.ocr(pixels, cls=True)
            return self._parse_paddle_legacy(results)
        except Exception as exc:  # noqa: BLE001
            OcrExtractor._paddle_failed = True
            logger.debug("PaddleOCR unavailable: %s", exc)
            return (), 0.0, "none"

    @staticmethod
    def _parse_paddle_legacy(results: object) -> tuple[tuple[str, ...], float, str]:
        texts: list[str] = []
        scores: list[float] = []
        rows = results[0] if results else None  # type: ignore[index]
        if not rows:
            return (), 0.0, "paddleocr"
        for row in rows:
            if not row or len(row) < 2:
                continue
            payload = row[1]
            if not payload or len(payload) < 2:
                continue
            text = str(payload[0]).strip()
            score = float(payload[1])
            if text and score >= 0.45 and len(text) >= 2:
                texts.append(text)
                scores.append(score)
        if not texts:
            return (), 0.0, "paddleocr"
        return tuple(texts[:12]), float(sum(scores) / len(scores)), "paddleocr"

    @staticmethod
    def _parse_paddle_predict(results: object) -> tuple[tuple[str, ...], float, str]:
        texts: list[str] = []
        scores: list[float] = []
        for item in results or ():  # type: ignore[union-attr]
            rec_texts = getattr(item, "get", None)
            if callable(rec_texts):
                # dict-like result
                words = item.get("rec_texts") or item.get("texts") or []
                confs = item.get("rec_scores") or item.get("scores") or []
            else:
                words = getattr(item, "rec_texts", None) or getattr(item, "texts", None) or []
                confs = getattr(item, "rec_scores", None) or getattr(item, "scores", None) or []
            for text, score in zip(list(words), list(confs) or [0.9] * len(list(words))):
                cleaned = str(text).strip()
                conf = float(score)
                if cleaned and conf >= 0.45 and len(cleaned) >= 2:
                    texts.append(cleaned)
                    scores.append(conf)
        if not texts:
            return (), 0.0, "paddleocr"
        return tuple(texts[:12]), float(sum(scores) / len(scores)), "paddleocr"
