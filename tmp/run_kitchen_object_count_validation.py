"""Kitchen object-count regression: refrigerator count must match verified entities."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from language.validation.caption_factuality import clamp_caption_object_counts, label_counts_from_verified

# Production kitchen with verified refrigerator (not sparse coco_kitchen).
KITCHEN = Path("tmp/uploads/5e4dc9b3-589c-43c6-84b0-5620936c9df4.png")


def main() -> None:
    path = KITCHEN if KITCHEN.exists() else Path("tmp/coco_kitchen.jpg")
    if not path.exists():
        raise SystemExit(f"missing {path}")
    startup = StartupOrchestrator().run()
    orch = startup.context.main_controller.pipeline._orchestrator
    opts = AnalysisOptions(
        enable_gemma=False,
        competition_mode=False,
        enable_enhancement=False,
        enable_super_resolution=False,
        enable_sam2=False,
    )
    result = orch.analyze(PipelineRequest(image_path=path, options=opts))
    cap = getattr(result.caption, "canonical_caption_en", None) or result.caption.text or ""
    ve = result.verified_evidence
    counts = label_counts_from_verified(ve) if ve else {}
    label_counter = Counter(
        (e.label or "").lower()
        for e in (ve.entities if ve else [])
        if getattr(e, "narrative_safe", True)
    )
    fridge_n = counts.get("refrigerator", label_counter.get("refrigerator", 0))
    chair_n = counts.get("chair", label_counter.get("chair", 0))
    lower = cap.lower()
    bad_fridge = bool(re.search(r"\b[2-9]\s+(?:\w+\s+){0,2}refrigerators?\b", lower))
    bad_fridge |= "several refrigerator" in lower or "multiple refrigerator" in lower
    if fridge_n <= 1:
        bad_fridge |= bool(
            re.search(
                r"\b(?:2|3|4|5|6|two|three|four|five|six)\s+(?:\w+\s+){0,2}refrigerators?\b",
                lower,
            )
        )
    out = {
        "image": str(path),
        "caption": cap,
        "verified_label_counts": dict(sorted(counts.items())),
        "refrigerator_verified": fridge_n,
        "chair_verified": chair_n,
        "bad_inflated_refrigerator": bad_fridge,
        "clamp_demo": clamp_caption_object_counts(
            "4 brown refrigerators and 5 brown chairs appear farther back.",
            verified=ve,
        ),
    }
    Path("tmp/KITCHEN_OBJECT_COUNT_VALIDATION.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))
    if bad_fridge:
        raise SystemExit("FAIL: inflated refrigerator count in caption")
    print("PASS kitchen refrigerator count")


if __name__ == "__main__":
    main()
