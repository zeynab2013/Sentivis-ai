"""Pose / action estimation with optional MediaPipe keypoints and relation fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.contracts.analysis import Relation
from core.contracts.detection import Detection, DetectionResult
from core.logging import get_logger

logger = get_logger(__name__)

_PERSON = {"person", "people", "man", "woman", "child"}
_SITTABLE = {"chair", "couch", "bench", "bed", "sofa"}
_HOLDABLE = {"cup", "bottle", "book", "cell phone", "laptop", "umbrella", "handbag", "backpack"}
_SPORT = {"sports ball", "tennis racket", "baseball bat", "skateboard", "surfboard"}
_READABLE = {"book", "laptop", "cell phone"}


@dataclass(frozen=True)
class PoseEstimate:
    """Pose/action estimate for one person detection."""

    object_index: int
    pose: str
    action: str
    confidence: float
    source: str
    processing_time_ms: float


class PoseEstimator:
    """Infer pose/actions; prefer MediaPipe when available, else geometry + relations."""

    def __init__(self) -> None:
        self._mp_pose = None
        self._mp_tried = False

    def estimate(
        self,
        detections: DetectionResult,
        relations: tuple[Relation, ...],
        pixels: NDArray[np.uint8] | None = None,
    ) -> tuple[PoseEstimate, ...]:
        started = time.perf_counter()
        results: list[PoseEstimate] = []
        for index, detection in enumerate(detections.detections):
            if detection.label.lower() not in _PERSON:
                continue
            pose, pose_conf, source = self._pose_from_evidence(detection, pixels)
            action, action_conf = self._action_from_relations(index, detections, relations, pose)
            confidence = min(pose_conf, action_conf)
            if confidence < 0.45:
                continue
            elapsed = (time.perf_counter() - started) * 1000.0
            results.append(
                PoseEstimate(
                    object_index=index,
                    pose=pose,
                    action=action,
                    confidence=confidence,
                    source=source,
                    processing_time_ms=elapsed,
                )
            )
        logger.debug("Pose estimates=%d", len(results))
        return tuple(results)

    def _pose_from_evidence(
        self,
        detection: Detection,
        pixels: NDArray[np.uint8] | None,
    ) -> tuple[str, float, str]:
        if pixels is not None:
            mp_pose = self._mediapipe_pose(pixels, detection)
            if mp_pose is not None:
                return mp_pose[0], mp_pose[1], "mediapipe"
        pose, conf = self._pose_from_box(detection)
        return pose, conf, "geometry"

    def _mediapipe_pose(
        self,
        pixels: NDArray[np.uint8],
        detection: Detection,
    ) -> tuple[str, float] | None:
        pose_module = self._ensure_mediapipe()
        if pose_module is None:
            return None
        box = detection.bounding_box
        height, width = pixels.shape[:2]
        x0 = int(max(0, box.x_min))
        y0 = int(max(0, box.y_min))
        x1 = int(min(width, box.x_max))
        y1 = int(min(height, box.y_max))
        if x1 - x0 < 24 or y1 - y0 < 40:
            return None
        crop = pixels[y0:y1, x0:x1]
        try:
            import cv2

            rgb = crop if crop.shape[2] == 3 else cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = pose_module.process(rgb)
            if result.pose_landmarks is None:
                return None
            landmarks = result.pose_landmarks.landmark
            # MediaPipe indices: 11/12 shoulders, 23/24 hips, 25/26 knees, 27/28 ankles
            shoulder_y = (landmarks[11].y + landmarks[12].y) / 2.0
            hip_y = (landmarks[23].y + landmarks[24].y) / 2.0
            knee_y = (landmarks[25].y + landmarks[26].y) / 2.0
            ankle_y = (landmarks[27].y + landmarks[28].y) / 2.0
            wrist_y = (landmarks[15].y + landmarks[16].y) / 2.0
            nose_y = landmarks[0].y

            torso = abs(hip_y - shoulder_y)
            leg_bend = abs(knee_y - hip_y)
            if torso < 0.12 and leg_bend < 0.18:
                return "sitting", min(0.88, detection.confidence)
            if ankle_y - hip_y > 0.35 and abs(shoulder_y - hip_y) > 0.15:
                # upright
                stride = abs(landmarks[27].x - landmarks[28].x)
                if stride > 0.18 and wrist_y < shoulder_y + 0.05:
                    return "walking", min(0.8, detection.confidence)
                if stride > 0.28:
                    return "running", min(0.75, detection.confidence)
                if nose_y < shoulder_y and abs(landmarks[15].y - landmarks[16].y) < 0.08:
                    return "standing", min(0.85, detection.confidence)
                return "standing", min(0.8, detection.confidence)
            if hip_y > 0.7:
                return "sitting", min(0.78, detection.confidence)
            return "standing", min(0.7, detection.confidence)
        except Exception as exc:  # noqa: BLE001
            logger.debug("MediaPipe pose failed: %s", exc)
            return None

    def _ensure_mediapipe(self) -> object | None:
        if self._mp_tried:
            return self._mp_pose
        self._mp_tried = True
        try:
            import mediapipe as mp

            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )
            logger.info("MediaPipe pose enabled")
        except Exception as exc:  # noqa: BLE001
            logger.debug("MediaPipe unavailable: %s", exc)
            self._mp_pose = None
        return self._mp_pose

    def _pose_from_box(self, detection: Detection) -> tuple[str, float]:
        """Geometry-only pose guess. Prefer unknown over confident hallucination.

        Aspect ratio = height / width:
        - tall boxes → standing
        - very wide/short boxes → lying
        - moderately short boxes → sitting
        - mid-band / degenerate geometry → unknown
        """
        box = detection.bounding_box
        width = float(box.width)
        height = float(box.height)
        if width <= 1.0 or height <= 1.0:
            return "unknown", 0.2
        ratio = height / width
        # Extreme horizontal: lying must be checked BEFORE sitting.
        if ratio <= 0.85:
            return "lying", min(0.7, detection.confidence)
        if ratio >= 1.75:
            return "standing", min(0.9, detection.confidence)
        if ratio <= 1.15:
            return "sitting", min(0.75, detection.confidence)
        # Ambiguous mid-band (roughly 1.15–1.75): do not invent a pose.
        return "unknown", 0.35

    def _action_from_relations(
        self,
        person_index: int,
        detections: DetectionResult,
        relations: tuple[Relation, ...],
        pose: str,
    ) -> tuple[str, float]:
        labels = {i: d.label.lower() for i, d in enumerate(detections.detections)}
        for relation in relations:
            if relation.subject_index != person_index or relation.confidence < 0.5:
                continue
            obj = labels.get(relation.object_index, "")
            rel = relation.relation_type
            if rel == "holding" and obj in _HOLDABLE:
                if obj in {"cup", "bottle", "wine glass"}:
                    return "holding", relation.confidence
                if obj == "book":
                    return "reading", relation.confidence
                if obj == "laptop":
                    return "using laptop", relation.confidence
                if obj == "cell phone":
                    return "holding", relation.confidence
                if obj == "umbrella":
                    return "holding", relation.confidence
                return "holding", relation.confidence
            if rel == "sitting_on" and obj in _SITTABLE:
                return "sitting", relation.confidence
            if rel == "playing_with" and obj in _SPORT:
                return "playing", relation.confidence
            if rel == "looking_at":
                return "looking", relation.confidence * 0.9
            if rel in {"walking_toward", "running_toward"}:
                return rel.replace("_", " "), relation.confidence
            if rel == "talking_to":
                return "talking", relation.confidence
        if pose in {"walking", "running", "standing", "sitting", "jumping"}:
            return pose, 0.58 if pose in {"walking", "running"} else 0.55
        return "unknown", 0.35
