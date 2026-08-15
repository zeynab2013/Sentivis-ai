"""YOLO detection engine."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from ultralytics import YOLO  # type: ignore[attr-defined]

from core.config.model_config import YoloModelConfig
from core.constants.model_kinds import ModelKind
from core.contracts.detection import BoundingBox, Detection, DetectionResult, SegmentationMask
from core.contracts.image import PreprocessedImage
from core.exceptions.vision import DetectionError
from core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Recall-first inference floor; class-specific gates applied after NMS.
_INFER_CONF_FLOOR = 0.22

# Primary subjects may pass with slightly lower confidence than clutter.
_PRIMARY_MIN_CONFIDENCE: dict[str, float] = {
    "person": 0.28,
    "dog": 0.30,
    "cat": 0.30,
    "horse": 0.30,
    "cow": 0.30,
    "sheep": 0.30,
    "bird": 0.32,
    "elephant": 0.30,
    "bear": 0.30,
    "zebra": 0.30,
    "giraffe": 0.30,
    "car": 0.32,
    "bus": 0.32,
    "truck": 0.32,
    "motorcycle": 0.32,
    "bicycle": 0.32,
    "airplane": 0.30,
    "train": 0.30,
    "boat": 0.32,
}

# Small accessories / props need stronger visual evidence than people or animals.
_ACCESSORY_MIN_CONFIDENCE: dict[str, float] = {
    "backpack": 0.78,
    "handbag": 0.75,
    "umbrella": 0.68,
    "tie": 0.75,
    "suitcase": 0.65,
    "cell phone": 0.72,
    "remote": 0.70,
    "mouse": 0.65,
    "book": 0.60,
    "bottle": 0.58,
    "cup": 0.58,
    "bowl": 0.55,
    "fork": 0.60,
    "knife": 0.60,
    "spoon": 0.60,
    "toothbrush": 0.70,
    "hair drier": 0.70,
    "scissors": 0.65,
    "clock": 0.60,
    "vase": 0.55,
    "parking meter": 0.60,
}
_SMALL_AREA_ACCESSORIES = frozenset(_ACCESSORY_MIN_CONFIDENCE)
_MIN_ACCESSORY_AREA_RATIO = 0.008
_CLUTTER_MIN_CONFIDENCE = 0.42

# Confusable COCO pairs — keep the stronger box when IoU is high.
_CONFUSABLE_PAIRS: tuple[frozenset[str], ...] = (
    frozenset({"chair", "bench"}),
    frozenset({"horse", "cow"}),
    frozenset({"dog", "bear"}),  # wolf not in COCO; bear is the common FP neighbor
    frozenset({"car", "truck"}),
    frozenset({"dining table", "bench"}),
)
_DUPLICATE_IOU = 0.55
_CONFUSABLE_IOU = 0.45


def _box_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Local IoU helper — vision must not import analysis/."""
    x_min = max(box_a.x_min, box_b.x_min)
    y_min = max(box_a.y_min, box_b.y_min)
    x_max = min(box_a.x_max, box_b.x_max)
    y_max = min(box_a.y_max, box_b.y_max)
    inter = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    if inter <= 0.0:
        return 0.0
    union = box_a.area + box_b.area - inter
    return inter / max(union, 1.0)


def _box_intersection_over_min(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Intersection over the smaller box — catches nested duplicate detections."""
    x_min = max(box_a.x_min, box_b.x_min)
    y_min = max(box_a.y_min, box_b.y_min)
    x_max = min(box_a.x_max, box_b.x_max)
    y_max = min(box_a.y_max, box_b.y_max)
    inter = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    if inter <= 0.0:
        return 0.0
    smaller = min(box_a.area, box_b.area)
    return inter / max(smaller, 1.0)


# Rigid roadside / fixture labels often produce nested duplicate boxes.
_NESTED_DEDUP_LABELS = frozenset(
    {
        "stop sign",
        "traffic light",
        "parking meter",
        "fire hydrant",
        "clock",
        "tv",
        "refrigerator",
        "microwave",
        "oven",
        "toilet",
        "bench",
    }
)
_DUPLICATE_IOA = 0.70
_NESTED_DEDUP_IOU = 0.28
_NESTED_DEDUP_IOA = 0.55


class YoloEngine:
    """Loads and runs YOLO inference with explicit lifecycle."""

    def __init__(self, config: YoloModelConfig, *, search_paths: tuple[Path, ...] = ()) -> None:
        self._config = config
        self._search_paths = search_paths
        self._model: YOLO | None = None
        self._device = config.preferred_device
        self._loaded = False

    @property
    def model_kind(self) -> ModelKind:
        return ModelKind.YOLO

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    def set_device(self, device: str) -> None:
        self._device = device

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from services.runtime.yolo_weights import resolve_yolo_weights_path

            resolved = resolve_yolo_weights_path(
                variant=self._config.variant,
                configured_path=self._config.weights_path,
                search_paths=self._search_paths,
            )
            model_name = str(resolved) if resolved is not None else f"{self._config.variant}.pt"
            self._model = YOLO(model_name)
            self._loaded = True
            logger.info("YOLO model loaded: %s on %s", model_name, self._device)
        except OSError as exc:
            raise DetectionError(
                "Object detection model could not be loaded.",
                f"YOLO load failed: {exc}",
                recoverable=True,
            ) from exc

    def infer(self, image: PreprocessedImage) -> DetectionResult:
        if not self._model:
            raise DetectionError(
                "Object detection is not ready.",
                "YOLO infer called before load",
                recoverable=False,
            )
        try:
            # Lower predict floor improves recall for people/animals; class gates filter later.
            infer_conf = min(float(self._config.confidence_threshold), _INFER_CONF_FLOOR)
            # CRITICAL: match Ultralytics imgsz to the prepared inference buffer.
            # Default imgsz=640 silently downscales a 1280 buffer and destroys small-object recall.
            infer_imgsz = max(int(image.inference_width), int(image.inference_height), 32)
            results = self._model.predict(
                source=image.inference_pixels,
                conf=infer_conf,
                iou=self._config.iou_threshold,
                imgsz=infer_imgsz,
                device=self._device,
                verbose=False,
            )
        except RuntimeError as exc:
            raise DetectionError(
                "Object detection failed during analysis.",
                f"YOLO inference error: {exc}",
                recoverable=True,
            ) from exc

        raw_detections: list[Detection] = []
        source = image.source
        scale_x = source.width / image.inference_width
        scale_y = source.height / image.inference_height
        image_area = max(1.0, float(source.width * source.height))
        detected_at = time.time()

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            masks = getattr(result, "masks", None)
            for box_index, box in enumerate(boxes):
                xyxy = box.xyxy.cpu().numpy().astype(float).flatten()
                x_min = float(xyxy[0] * scale_x)
                y_min = float(xyxy[1] * scale_y)
                x_max = float(xyxy[2] * scale_x)
                y_max = float(xyxy[3] * scale_y)
                bbox = BoundingBox(x_min, y_min, x_max, y_max)
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                label = result.names.get(class_id, str(class_id))
                segmentation = self._extract_segmentation(
                    masks,
                    box_index,
                    scale_x,
                    scale_y,
                    bbox.area / image_area,
                )
                raw_detections.append(
                    Detection(
                        object_id=f"obj-{uuid4().hex[:12]}",
                        label=label,
                        confidence=confidence,
                        bounding_box=bbox,
                        class_id=class_id,
                        detected_at=detected_at,
                        segmentation=segmentation,
                    )
                )

        for det in raw_detections:
            logger.info(
                "YOLO raw detection label=%s conf=%.3f area=%.4f",
                det.label,
                det.confidence,
                det.bounding_box.area / image_area,
            )

        detections, removed = self._filter_detections(raw_detections, image_area)
        for label, conf, reason in removed:
            logger.info(
                "YOLO filtered out label=%s conf=%.3f reason=%s",
                label,
                conf,
                reason,
            )
        logger.info(
            "YOLO detections raw=%d kept=%d removed=%d",
            len(raw_detections),
            len(detections),
            len(removed),
        )
        return DetectionResult(
            detections=tuple(detections),
            image_width=source.width,
            image_height=source.height,
            inference_timestamp=detected_at,
        )

    def _filter_detections(
        self,
        detections: list[Detection],
        image_area: float,
    ) -> tuple[list[Detection], list[tuple[str, float, str]]]:
        """Drop low-confidence / tiny / duplicate / confusable detections after YOLO NMS."""
        kept: list[Detection] = []
        removed: list[tuple[str, float, str]] = []

        # Spatial clusters for accessories: keep one strong box per label per region,
        # so two real backpacks on different people are not collapsed image-wide.
        accessory_winners: list[Detection] = []
        for det in detections:
            key = det.label.lower()
            if key not in _SMALL_AREA_ACCESSORIES:
                continue
            replaced = False
            for index, prior in enumerate(accessory_winners):
                if prior.label.lower() != key:
                    continue
                if _box_iou(det.bounding_box, prior.bounding_box) < 0.25:
                    # Far apart — likely distinct instances near different people.
                    continue
                if det.confidence > prior.confidence or (
                    abs(det.confidence - prior.confidence) < 0.05
                    and det.bounding_box.area > prior.bounding_box.area
                ):
                    accessory_winners[index] = det
                replaced = True
                break
            if not replaced:
                accessory_winners.append(det)
        accessory_winner_ids = {id(item) for item in accessory_winners}

        for det in detections:
            label = det.label.lower()
            area_ratio = det.bounding_box.area / image_area
            if label in _ACCESSORY_MIN_CONFIDENCE:
                min_conf = _ACCESSORY_MIN_CONFIDENCE[label]
            elif label in _PRIMARY_MIN_CONFIDENCE:
                min_conf = _PRIMARY_MIN_CONFIDENCE[label]
            else:
                min_conf = _CLUTTER_MIN_CONFIDENCE
            if det.confidence < min_conf:
                removed.append((det.label, det.confidence, f"conf<{min_conf:.2f}"))
                continue
            if label in _SMALL_AREA_ACCESSORIES and area_ratio < _MIN_ACCESSORY_AREA_RATIO:
                if det.confidence < max(min_conf, 0.85):
                    removed.append((det.label, det.confidence, f"tiny_area={area_ratio:.4f}"))
                    continue
            if label in _SMALL_AREA_ACCESSORIES and id(det) not in accessory_winner_ids:
                removed.append((det.label, det.confidence, "duplicate_accessory"))
                continue
            kept.append(det)

        kept, confusable_removed = self._suppress_confusable_duplicates(kept)
        removed.extend(confusable_removed)
        kept, duplicate_removed = self._suppress_same_label_duplicates(kept)
        removed.extend(duplicate_removed)
        return kept, removed

    def _suppress_confusable_duplicates(
        self,
        detections: list[Detection],
    ) -> tuple[list[Detection], list[tuple[str, float, str]]]:
        """Resolve chair/bench, horse/cow, car/truck overlaps by keeping the stronger box."""
        drop_ids: set[int] = set()
        removed: list[tuple[str, float, str]] = []
        for i, left in enumerate(detections):
            if id(left) in drop_ids:
                continue
            left_label = left.label.lower()
            for right in detections[i + 1 :]:
                if id(right) in drop_ids:
                    continue
                right_label = right.label.lower()
                pair = frozenset({left_label, right_label})
                if pair not in _CONFUSABLE_PAIRS:
                    continue
                if _box_iou(left.bounding_box, right.bounding_box) < _CONFUSABLE_IOU:
                    continue
                # Aspect heuristics when confidence is near-tied.
                winner, loser = left, right
                if abs(left.confidence - right.confidence) < 0.08:
                    winner, loser = self._disambiguate_pair(left, right)
                elif right.confidence > left.confidence:
                    winner, loser = right, left
                drop_ids.add(id(loser))
                removed.append((loser.label, loser.confidence, f"confusable_with_{winner.label}"))
        return [det for det in detections if id(det) not in drop_ids], removed

    @staticmethod
    def _disambiguate_pair(left: Detection, right: Detection) -> tuple[Detection, Detection]:
        """Prefer geometry-consistent label when confidence is tied."""
        labels = {left.label.lower(), right.label.lower()}
        by_label = {left.label.lower(): left, right.label.lower(): right}
        if labels == {"chair", "bench"}:
            # Benches are typically wider than tall; chairs are taller.
            for label, det in by_label.items():
                ratio = det.bounding_box.width / max(det.bounding_box.height, 1.0)
                if label == "bench" and ratio >= 1.15:
                    other = by_label["chair"]
                    return det, other
                if label == "chair" and ratio <= 0.95:
                    other = by_label["bench"]
                    return det, other
        if labels == {"car", "truck"}:
            for label, det in by_label.items():
                ratio = det.bounding_box.width / max(det.bounding_box.height, 1.0)
                area = det.bounding_box.area
                if label == "truck" and (ratio >= 1.6 or area >= by_label["car"].bounding_box.area * 1.15):
                    return det, by_label["car"]
                if label == "car" and ratio < 1.45 and area <= by_label["truck"].bounding_box.area:
                    return det, by_label["truck"]
        if labels == {"horse", "cow"}:
            # Horses are typically taller relative to width than cows.
            for label, det in by_label.items():
                ratio = det.bounding_box.height / max(det.bounding_box.width, 1.0)
                if label == "horse" and ratio >= 1.05:
                    return det, by_label["cow"]
                if label == "cow" and ratio < 0.95:
                    return det, by_label["horse"]
        # Default: higher confidence (or larger box on near-tie).
        if left.confidence > right.confidence:
            return left, right
        if right.confidence > left.confidence:
            return right, left
        if left.bounding_box.area >= right.bounding_box.area:
            return left, right
        return right, left

    def _suppress_same_label_duplicates(
        self,
        detections: list[Detection],
    ) -> tuple[list[Detection], list[tuple[str, float, str]]]:
        """Collapse near-identical same-label boxes into one semantic object.

        Uses IoU and intersection-over-min-area so nested boxes of one physical
        object (common for stop signs / traffic lights) do not inflate counts.
        """
        drop_ids: set[int] = set()
        removed: list[tuple[str, float, str]] = []
        for i, left in enumerate(detections):
            if id(left) in drop_ids:
                continue
            left_label = left.label.lower()
            for right in detections[i + 1 :]:
                if id(right) in drop_ids:
                    continue
                if left_label != right.label.lower():
                    continue
                iou = _box_iou(left.bounding_box, right.bounding_box)
                ioa = _box_intersection_over_min(left.bounding_box, right.bounding_box)
                nested = left_label in _NESTED_DEDUP_LABELS and (
                    iou >= _NESTED_DEDUP_IOU or ioa >= _NESTED_DEDUP_IOA
                )
                if iou < _DUPLICATE_IOU and ioa < _DUPLICATE_IOA and not nested:
                    continue
                if right.confidence > left.confidence or (
                    abs(right.confidence - left.confidence) < 0.04
                    and right.bounding_box.area > left.bounding_box.area
                ):
                    drop_ids.add(id(left))
                    removed.append((left.label, left.confidence, "duplicate_semantic"))
                    break
                drop_ids.add(id(right))
                removed.append((right.label, right.confidence, "duplicate_semantic"))
        return [det for det in detections if id(det) not in drop_ids], removed

    def _extract_segmentation(
        self,
        masks: object,
        box_index: int,
        scale_x: float,
        scale_y: float,
        area_ratio: float,
    ) -> SegmentationMask | None:
        if masks is None:
            return None
        try:
            xy = masks.xy  # type: ignore[attr-defined]
            if box_index >= len(xy):
                return None
            polygon_raw = xy[box_index]
            if polygon_raw is None or len(polygon_raw) < 3:
                return None
            polygon = tuple(
                (float(point[0] * scale_x), float(point[1] * scale_y))
                for point in polygon_raw
            )
            return SegmentationMask(polygon=polygon, area_ratio=area_ratio)
        except (AttributeError, IndexError, TypeError):
            return None

    def release(self) -> None:
        self._model = None
        self._loaded = False
        logger.info("YOLO model released")

    def clear_device_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
