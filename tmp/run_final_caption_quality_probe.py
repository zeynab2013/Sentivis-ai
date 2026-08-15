"""Caption quality probe: critical + complex images after dynamic coverage pass."""

from __future__ import annotations

import json
from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest

CASES = [
    ("HORSE", Path("tmp/uploads/10815824_2997e03d76.jpg")),
    ("SOCCER", Path("tmp/uploads/47871819_db55ac4699.jpg")),
    ("MOTORCYCLE", Path("tmp/uploads/143552829_72b6ba49d4.jpg")),
    ("BICYCLE", Path("tmp/uploads/191003284_1025b0fb7d.jpg")),
    ("KITCHEN_COMPLEX", Path("tmp/coco_kitchen.jpg")),
    ("OUTDOOR_TRAIL", Path("tmp/uploads/95728660_d47de66544.jpg")),
    ("DENSE_INDOOR", Path("tmp/uploads/random_385406.jpg")),
    ("OCR_SOCCER", Path("tmp/uploads/47871819_db55ac4699.jpg")),
    ("COLOR_BIKE", Path("tmp/uploads/191003284_1025b0fb7d.jpg")),
    ("RELATION_HORSE", Path("tmp/uploads/10815824_2997e03d76.jpg")),
]


def main() -> None:
    startup = StartupOrchestrator().run()
    orch = startup.context.main_controller.pipeline._orchestrator
    opts = AnalysisOptions(
        enable_gemma=False,
        competition_mode=False,
        enable_enhancement=False,
        enable_super_resolution=False,
        enable_sam2=False,
    )
    rows = []
    for name, path in CASES:
        if not path.exists():
            rows.append({"name": name, "status": "MISSING", "path": str(path)})
            continue
        print(f"=== {name} ===", flush=True)
        result = orch.analyze(PipelineRequest(image_path=path, options=opts))
        caption = (
            getattr(result.caption, "canonical_caption_en", None)
            or result.caption.text
            or ""
        )
        qr = result.quality_report
        ve = result.verified_evidence
        entities = []
        if ve is not None:
            entities = sorted(
                {
                    (e.label or "").lower()
                    for e in ve.entities
                    if (e.label or "").strip()
                }
            )
        setting = ""
        if ve is not None and ve.scene is not None:
            setting = ve.scene.setting or ve.scene.scene_type or ""
        rows.append(
            {
                "name": name,
                "status": "OK",
                "path": str(path),
                "words": len(caption.split()),
                "sentences": max(1, caption.count(".") + caption.count("!") + caption.count("?")),
                "caption": caption,
                "setting": setting,
                "entities": entities[:20],
                "overall_quality": None if qr is None else qr.overall_quality,
                "object_coverage": None if qr is None else qr.object_coverage,
                "relationship_coverage": None if qr is None else qr.relationship_coverage,
                "hallucination_risk": None if qr is None else qr.hallucination_risk,
                "evidence_consistency": None if qr is None else qr.evidence_consistency,
            }
        )
        print(f"words={len(caption.split())} setting={setting}", flush=True)
        print(caption[:400], flush=True)

    out = Path("tmp/FINAL_CAPTION_QUALITY_PROBE.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
