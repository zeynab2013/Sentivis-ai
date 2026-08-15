"""Derive per-object attributes from detections, masks, and clothing cues."""

from analysis.common.geometry import position_zone
from core.config.analysis_config import AnalysisConfig
from core.contracts.analysis import Attribute, AttributeSet
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.contracts.image import PreprocessedImage
from core.logging import get_logger
from vision.crop_analysis.object_crop_analyzer import CropAnalysis, ObjectCropAnalyzer

logger = get_logger(__name__)

_PERSON_LABELS = {"person", "people", "man", "woman", "child"}
_MIN_ATTR_CONF = 0.55


class AttributeExtractor:
    """Extracts visual attributes for each detected object via mask-aware analysis."""

    def __init__(self, analysis_config: AnalysisConfig) -> None:
        self._config = analysis_config
        self._crop_analyzer = ObjectCropAnalyzer()

    def extract(self, detections: DetectionResult, image: PreprocessedImage) -> AttributeSet:
        image_area = max(1.0, detections.image_width * detections.image_height)
        attributes: list[Attribute] = []
        overlap_counts = self._overlap_counts(detections)
        # ROOT CAUSE FIX: color/clothing must sample the ORIGINAL image.
        # display_pixels are gray-world / CLAHE enhanced and systematically
        # collapse chromatic garments and animals toward gray.
        color_pixels = (
            image.original_display_pixels
            if image.original_display_pixels is not None
            else image.display_pixels
        )
        # When enhancement/SR changes resolution, detection boxes are in
        # ``detections.image_*`` space (enhanced). Remap into original pixel space
        # before cropping — otherwise every crop samples the wrong region.
        crop_detections = self._align_detections_to_pixels(detections, color_pixels)
        crops = self._crop_analyzer.analyze_all(color_pixels, crop_detections)

        for index, (detection, crop) in enumerate(zip(detections.detections, crops, strict=True)):
            attributes.extend(
                self._attributes_for_detection(
                    index,
                    detection,
                    crop,
                    image_area,
                    detections.image_width,
                    detections.image_height,
                    overlap_counts.get(index, 0),
                )
            )

        logger.debug("Extracted %d attributes from %d crops", len(attributes), len(crops))
        return AttributeSet(attributes=tuple(attributes))

    @staticmethod
    def _align_detections_to_pixels(
        detections: DetectionResult,
        pixels: object,
    ) -> tuple[Detection, ...]:
        """Scale boxes from detection image space into the color-sample pixel grid."""
        import numpy as np
        from dataclasses import replace

        if not isinstance(pixels, np.ndarray) or pixels.ndim < 2:
            return detections.detections
        pix_h, pix_w = int(pixels.shape[0]), int(pixels.shape[1])
        det_w = max(1, int(detections.image_width))
        det_h = max(1, int(detections.image_height))
        if pix_w == det_w and pix_h == det_h:
            return detections.detections
        scale_x = pix_w / float(det_w)
        scale_y = pix_h / float(det_h)
        aligned: list[Detection] = []
        for det in detections.detections:
            box = det.bounding_box
            scaled_box = BoundingBox(
                x_min=box.x_min * scale_x,
                y_min=box.y_min * scale_y,
                x_max=box.x_max * scale_x,
                y_max=box.y_max * scale_y,
            )
            scaled_mask = det.segmentation
            if scaled_mask is not None and scaled_mask.polygon:
                scaled_mask = replace(
                    scaled_mask,
                    polygon=tuple((x * scale_x, y * scale_y) for x, y in scaled_mask.polygon),
                )
            aligned.append(
                replace(det, bounding_box=scaled_box, segmentation=scaled_mask)
            )
        return tuple(aligned)

    def _attributes_for_detection(
        self,
        index: int,
        detection: Detection,
        crop: CropAnalysis,
        image_area: float,
        image_width: int,
        image_height: int,
        overlap_count: int,
    ) -> list[Attribute]:
        box = detection.bounding_box
        area_ratio = box.area / image_area
        cfg = self._config.attributes
        if crop.confidence < _MIN_ATTR_CONF and detection.label.lower() not in _PERSON_LABELS:
            return [
                Attribute(index, "confidence", f"{detection.confidence:.0%}"),
                Attribute(index, "visibility", self._visibility_label(detection.confidence)),
            ]

        attrs = [
            Attribute(index, "color", crop.dominant_color),
            Attribute(index, "dominant_color", crop.dominant_color),
            Attribute(index, "secondary_color", crop.secondary_color),
            Attribute(index, "relative_size", self._size_label(area_ratio)),
            Attribute(index, "estimated_distance", self._distance_label(area_ratio)),
            Attribute(index, "pose", self._pose_label(detection.label, box)),
            Attribute(index, "orientation", self._orientation_label(box)),
            Attribute(index, "facing_direction", self._facing_direction(box, image_width)),
            Attribute(index, "visibility", self._visibility_label(detection.confidence)),
            Attribute(index, "occlusion", self._occlusion_label(overlap_count)),
            Attribute(index, "confidence", f"{detection.confidence:.0%}"),
            Attribute(
                index,
                "segmentation",
                "mask" if detection.segmentation is not None else "bbox",
            ),
            Attribute(
                index,
                "position_zone",
                position_zone(box, image_width, image_height, cfg.zone_split_low, cfg.zone_split_high),
            ),
            Attribute(index, "material", crop.material),
            Attribute(index, "texture", crop.texture),
            Attribute(index, "brightness", crop.brightness),
            Attribute(index, "edge_density", crop.edge_density),
            Attribute(index, "crop_description", crop.description),
        ]
        clothing = crop.clothing
        if clothing is not None and clothing.confidence >= _MIN_ATTR_CONF:
            gated = {
                "hair_color": clothing.hair_color,
                "hair_length": clothing.hair_length,
                "hairstyle": clothing.hairstyle,
                "shirt_color": clothing.shirt_color,
                "pants_color": clothing.pants_color,
                "shoes_color": clothing.shoes_color,
                "clothing_color": clothing.clothing_color,
                "secondary_color": clothing.secondary_color,
                "clothing_type": clothing.clothing_type,
                "clothing_style": clothing.clothing_style,
                "clothing_texture": clothing.clothing_texture,
                "sleeve_length": clothing.sleeve_length,
                "jacket": clothing.jacket,
                "coat": clothing.coat,
                "dress": clothing.dress,
                "hoodie": clothing.hoodie,
                "blazer": clothing.blazer,
                "sweater": clothing.sweater,
                "skirt": clothing.skirt,
                "jeans": clothing.jeans,
                "shorts": clothing.shorts,
                "footwear_type": clothing.footwear_type,
                "backpack": clothing.backpack,
                "handbag": clothing.handbag,
                "glasses": clothing.glasses,
                "sunglasses": clothing.sunglasses,
                "hat": clothing.hat,
                "cap": clothing.cap,
                "watch": clothing.watch,
                "necklace": clothing.necklace,
                "earrings": clothing.earrings,
                "accessories": clothing.accessories,
            }
            for name, value in gated.items():
                if value in {"unknown", "unlikely", "possible", "casual"} and name not in {
                    "accessories",
                }:
                    continue
                if name == "clothing_type" and value in {"casual", "seated_outfit"}:
                    continue
                attrs.append(Attribute(index, name, value))
            if clothing.dominant_colors:
                attrs.append(Attribute(index, "clothing_palette", ", ".join(clothing.dominant_colors)))
            attrs.append(Attribute(index, "estimated_age", self._estimate_age(box)))
            attrs.append(Attribute(index, "estimated_gender", "unknown"))
        return attrs

    def _estimate_age(self, box: object) -> str:
        from core.contracts.detection import BoundingBox

        assert isinstance(box, BoundingBox)
        return "child" if box.height / max(box.width, 1.0) < 1.4 else "20-30"

    def _facing_direction(self, box: object, image_width: int) -> str:
        from core.contracts.detection import BoundingBox

        assert isinstance(box, BoundingBox)
        center_x = box.center_x / max(1.0, float(image_width))
        if center_x < 0.33:
            return "left"
        if center_x > 0.67:
            return "right"
        return "center"

    def _size_label(self, area_ratio: float) -> str:
        for threshold, label in self._config.size_labels():
            if area_ratio <= threshold:
                return label
        return "large"

    def _distance_label(self, area_ratio: float) -> str:
        if area_ratio >= self._config.attributes.distance_near_ratio:
            return "near"
        if area_ratio >= self._config.attributes.distance_medium_ratio:
            return "medium"
        return "far"

    def _pose_label(self, label: str, box: object) -> str:
        from core.contracts.detection import BoundingBox

        assert isinstance(box, BoundingBox)
        if label.lower() not in _PERSON_LABELS:
            return "not_applicable"
        ratio = box.height / max(box.width, 1.0)
        if ratio >= self._config.attributes.pose_standing_ratio:
            return "standing"
        if ratio <= self._config.attributes.pose_lying_ratio:
            return "lying"
        return "unknown"

    def _orientation_label(self, box: object) -> str:
        from core.contracts.detection import BoundingBox

        assert isinstance(box, BoundingBox)
        ratio = box.width / max(box.height, 1.0)
        if ratio >= 1.25:
            return "horizontal"
        if ratio <= 0.8:
            return "vertical"
        return "square"

    def _visibility_label(self, confidence: float) -> str:
        if confidence >= self._config.attributes.visibility_high_threshold:
            return "high"
        if confidence >= self._config.attributes.visibility_medium_threshold:
            return "medium"
        return "low"

    def _occlusion_label(self, overlap_count: int) -> str:
        if overlap_count >= 2:
            return "heavily_occluded"
        if overlap_count == 1:
            return "partially_occluded"
        return "clear"

    def _overlap_counts(self, detections: DetectionResult) -> dict[int, int]:
        counts: dict[int, int] = {index: 0 for index in range(len(detections.detections))}
        items = detections.detections
        threshold = self._config.relationships.overlap_distance
        for i, subject in enumerate(items):
            for j, obj in enumerate(items):
                if i == j:
                    continue
                dx = obj.bounding_box.center_x - subject.bounding_box.center_x
                dy = obj.bounding_box.center_y - subject.bounding_box.center_y
                if (dx**2 + dy**2) ** 0.5 < threshold:
                    counts[j] += 1
        return counts
