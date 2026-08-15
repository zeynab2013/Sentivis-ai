"""Rich stub models for deterministic acceptance testing."""

from __future__ import annotations

import time

from core.contracts.analysis import SceneContext
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.contracts.image import PreprocessedImage
from core.contracts.language import Prompt, RawCaption, VisualObservations


class AcceptanceStubDetector:
    """Detector returning two objects so relationships can be inferred."""

    def detect(self, image: PreprocessedImage) -> DetectionResult:
        now = time.time()
        return DetectionResult(
            detections=(
                Detection(
                    object_id="obj-person",
                    label="person",
                    confidence=0.92,
                    bounding_box=BoundingBox(20, 30, 120, 280),
                    class_id=0,
                    detected_at=now,
                ),
                Detection(
                    object_id="obj-chair",
                    label="chair",
                    confidence=0.88,
                    bounding_box=BoundingBox(140, 180, 260, 320),
                    class_id=56,
                    detected_at=now,
                ),
            ),
            image_width=image.source.width,
            image_height=image.source.height,
            inference_timestamp=now,
        )


class AcceptanceStubVisionLanguage:
    """BLIP stub with scene observations."""

    def understand(self, image: PreprocessedImage, context: SceneContext) -> VisualObservations:
        raw = RawCaption(
            text="A person sits near a chair in an indoor scene.",
            source="blip",
            confidence=0.91,
        )
        return VisualObservations(
            observations=("A person sits near a chair.", "Indoor lighting is visible."),
            object_attributes=("person: posture=seated", "chair: material=wood"),
            candidate_descriptions=("A person sits near a chair in an indoor scene.",),
            confidence=0.91,
            raw_caption=raw,
        )


class AcceptanceStubReasoning:
    """Gemma stub producing a final caption."""

    def reason(self, prompt: Prompt, context: SceneContext) -> RawCaption:
        return RawCaption(
            text="A person is seated beside a wooden chair in a calm indoor setting.",
            source="gemma",
            confidence=0.87,
        )
