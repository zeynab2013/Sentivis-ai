"""Deterministic pipeline model stubs for certification and tests."""

from __future__ import annotations

import time

from core.contracts.analysis import SceneContext
from core.contracts.detection import BoundingBox, Detection, DetectionResult
from core.contracts.image import PreprocessedImage
from core.contracts.language import Prompt, RawCaption, VisualObservations


class StubDetector:
    """Minimal object detector for verification workflows."""

    def detect(self, image: PreprocessedImage) -> DetectionResult:
        now = time.time()
        return DetectionResult(
            detections=(
                Detection(
                    object_id="obj-test-person",
                    label="person",
                    confidence=0.9,
                    bounding_box=BoundingBox(10, 10, 100, 200),
                    class_id=0,
                    detected_at=now,
                ),
            ),
            image_width=image.source.width,
            image_height=image.source.height,
            inference_timestamp=now,
        )


class StubVisionLanguage:
    """Minimal vision-language model stub."""

    def understand(self, image: PreprocessedImage, context: SceneContext) -> VisualObservations:
        raw = RawCaption(text="A person in a scene.", source="blip", confidence=0.9)
        return VisualObservations(
            observations=("A person in a scene.",),
            object_attributes=("person: color=red",),
            candidate_descriptions=("A person in a scene.",),
            confidence=0.9,
            raw_caption=raw,
        )


class StubReasoning:
    """Minimal reasoning model stub."""

    def reason(self, prompt: Prompt, context: SceneContext) -> RawCaption:
        return RawCaption(text="A meaningful scene with a person.", source="gemma", confidence=0.8)
