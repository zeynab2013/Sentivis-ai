"""SAM2 segmentation refinement with YOLO/bbox fallback."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from analysis.common.mask_geometry import bbox_to_polygon, detection_with_fallback_mask, enrich_mask
from core.contracts.detection import Detection, DetectionResult, SegmentationMask
from core.contracts.image import PreprocessedImage
from core.logging import get_logger

logger = get_logger(__name__)


class Sam2SegmentationRefiner:
    """Refine YOLO detections with SAM2 masks when available."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._predictor: object | None = None
        self._available = self._try_load()

    @property
    def available(self) -> bool:
        return self._available

    def refine(self, detections: DetectionResult, image: PreprocessedImage) -> DetectionResult:
        """Attach precise segmentation masks to each detection."""
        if not detections.detections:
            return detections
        refined: list[Detection] = []
        for detection in detections.detections:
            mask = self._refine_single(detection, image, detections.image_width, detections.image_height)
            refined.append(
                Detection(
                    object_id=detection.object_id,
                    label=detection.label,
                    confidence=detection.confidence,
                    bounding_box=detection.bounding_box,
                    class_id=detection.class_id,
                    detected_at=detection.detected_at,
                    segmentation=mask,
                )
            )
        return replace(detections, detections=tuple(refined))

    def _refine_single(
        self,
        detection: Detection,
        image: PreprocessedImage,
        image_width: int,
        image_height: int,
    ) -> SegmentationMask:
        if detection.segmentation is not None and len(detection.segmentation.polygon) >= 3:
            return enrich_mask(detection.segmentation, image_width=image_width, image_height=image_height)
        if self._available and self._predictor is not None:
            sam_mask = self._sam2_mask(detection, image)
            if sam_mask is not None:
                return sam_mask
        fallback = detection_with_fallback_mask(detection, image_width, image_height)
        assert fallback.segmentation is not None
        return fallback.segmentation

    def _sam2_mask(
        self,
        detection: Detection,
        image: PreprocessedImage,
    ) -> SegmentationMask | None:
        try:
            box = detection.bounding_box
            prompt = [box.x_min, box.y_min, box.x_max, box.y_max]
            pixels = image.display_pixels
            if self._predictor is None:
                return None
            predict = getattr(self._predictor, "predict", None)
            if predict is None:
                return None
            masks = predict(pixels, box_prompt=prompt)
            if not masks:
                return None
            polygon = tuple((float(x), float(y)) for x, y in masks[0])
            image_area = float(max(1, image.display_pixels.shape[0] * image.display_pixels.shape[1]))
            area_ratio = detection.bounding_box.area / image_area
            return enrich_mask(
                SegmentationMask(polygon=polygon, area_ratio=area_ratio),
                image_width=image.display_pixels.shape[1],
                image_height=image.display_pixels.shape[0],
            )
        except Exception:
            logger.debug("SAM2 refinement failed for %s; using fallback", detection.object_id, exc_info=True)
            return None

    def _try_load(self) -> bool:
        # Prefer largest available checkpoint — accuracy over speed.
        candidates = (
            "sam2_hiera_large.pt",
            "sam2.1_hiera_large.pt",
            "sam2_hiera_base_plus.pt",
            "sam2.1_hiera_base_plus.pt",
            "sam2_hiera_small.pt",
            "sam2.1_hiera_small.pt",
            "sam2_hiera_tiny.pt",
        )
        expected = self._models_dir / "sam2" / "sam2_hiera_large.pt"
        checkpoint: Path | None = None
        for name in candidates:
            path = self._models_dir / "sam2" / name
            if path.is_file():
                checkpoint = path
                break
        if checkpoint is None:
            # Explicit diagnostic — do not pretend segmentation ran.
            logger.warning(
                "SAM2:\n"
                "Configured = yes\n"
                "Weights = missing\n"
                "Runtime = disabled\n"
                "Reason = weights unavailable (expected %s)",
                expected,
            )
            self._status = {
                "configured": True,
                "weights": "missing",
                "runtime": "disabled",
                "reason": "weights unavailable",
                "expected_path": str(expected),
            }
            return False
        try:
            import importlib

            build_sam2 = importlib.import_module("sam2.build_sam").build_sam2

            self._predictor = build_sam2(str(checkpoint))
            logger.info("SAM2 segmentation refiner loaded (%s).", checkpoint.name)
            self._status = {
                "configured": True,
                "weights": checkpoint.name,
                "runtime": "enabled",
                "reason": "loaded",
                "expected_path": str(expected),
            }
            return True
        except Exception as exc:
            logger.warning(
                "SAM2:\n"
                "Configured = yes\n"
                "Weights = present (%s)\n"
                "Runtime = disabled\n"
                "Reason = load failed (%s)",
                checkpoint.name,
                exc,
            )
            self._status = {
                "configured": True,
                "weights": checkpoint.name,
                "runtime": "disabled",
                "reason": f"load failed: {exc}",
                "expected_path": str(expected),
            }
            return False

    @property
    def status(self) -> dict[str, object]:
        """Explicit SAM2 availability diagnostic for UI/reports."""
        return getattr(
            self,
            "_status",
            {
                "configured": True,
                "weights": "unknown",
                "runtime": "disabled",
                "reason": "not initialized",
            },
        )

    @staticmethod
    def bbox_fallback_mask(detection: Detection, image_width: int, image_height: int) -> SegmentationMask:
        """Build rectangular mask from bounding box."""
        polygon = bbox_to_polygon(detection.bounding_box)
        image_area = float(max(1, image_width * image_height))
        return enrich_mask(
            SegmentationMask(polygon=polygon, area_ratio=detection.bounding_box.area / image_area),
            image_width=image_width,
            image_height=image_height,
        )
