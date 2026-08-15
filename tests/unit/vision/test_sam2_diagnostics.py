"""SAM2 must report explicit diagnostics when weights are missing."""

from __future__ import annotations

from pathlib import Path

from vision.segmentation.sam2_refiner import Sam2SegmentationRefiner


def test_sam2_missing_weights_diagnostic(tmp_path: Path) -> None:
    refiner = Sam2SegmentationRefiner(tmp_path)
    assert refiner.available is False
    status = refiner.status
    assert status["configured"] is True
    assert status["weights"] == "missing"
    assert status["runtime"] == "disabled"
    assert "unavailable" in str(status["reason"]).lower()
