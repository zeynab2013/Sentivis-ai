"""Pose geometry classification tests (lying / sitting / standing / unknown)."""

from __future__ import annotations

from analysis.pose.pose_estimator import PoseEstimator
from core.contracts.detection import BoundingBox, Detection


def _detection(width: float, height: float, *, confidence: float = 0.9) -> Detection:
    return Detection(
        object_id="person-0",
        label="person",
        confidence=confidence,
        bounding_box=BoundingBox(0.0, 0.0, width, height),
        class_id=0,
        detected_at=0.0,
    )


def test_pose_clearly_standing() -> None:
    est = PoseEstimator()
    pose, conf = est._pose_from_box(_detection(40, 100))  # noqa: SLF001
    assert pose == "standing"
    assert conf >= 0.7


def test_pose_clearly_lying() -> None:
    est = PoseEstimator()
    pose, conf = est._pose_from_box(_detection(120, 40))  # noqa: SLF001  ratio ~0.33
    assert pose == "lying"
    assert conf >= 0.5


def test_pose_clearly_sitting() -> None:
    est = PoseEstimator()
    pose, conf = est._pose_from_box(_detection(100, 100))  # noqa: SLF001  ratio 1.0
    assert pose == "sitting"
    assert conf >= 0.5


def test_pose_ambiguous_returns_unknown() -> None:
    est = PoseEstimator()
    pose, conf = est._pose_from_box(_detection(80, 120))  # noqa: SLF001  ratio 1.5
    assert pose == "unknown"
    assert conf < 0.5


def test_pose_degenerate_box_unknown() -> None:
    est = PoseEstimator()
    pose, conf = est._pose_from_box(_detection(0, 100))  # noqa: SLF001
    assert pose == "unknown"
    assert conf <= 0.3


def test_lying_reachable_not_swallowed_by_sitting() -> None:
    """Regression: ratio<=0.85 must classify lying, not sitting."""
    est = PoseEstimator()
    pose, _ = est._pose_from_box(_detection(200, 100))  # noqa: SLF001  ratio 0.5
    assert pose == "lying"
