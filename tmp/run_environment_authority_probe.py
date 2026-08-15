"""Environment-only before/after probe for surgical trail authority fix."""

from __future__ import annotations

from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest

CASES = [
    ("BICYCLE_TRAIL", Path("tmp/uploads/95728660_d47de66544.jpg")),
    ("HORSE", Path("tmp/uploads/10815824_2997e03d76.jpg")),
    ("SOCCER", Path("tmp/uploads/47871819_db55ac4699.jpg")),
    ("MOTORCYCLE", Path("tmp/uploads/143552829_72b6ba49d4.jpg")),
    ("BICYCLE_STREET", Path("tmp/uploads/191003284_1025b0fb7d.jpg")),
    ("KITCHEN", Path("tmp/coco_kitchen.jpg")),
    ("BUS", Path("tmp/competition_e2e_street.jpg")),
    ("ANIMAL", Path("tmp/uploads/random_850976.jpg")),
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
    print("=== ENVIRONMENT AUTHORITY PROBE ===\n")
    for name, path in CASES:
        if not path.exists():
            print(f"## {name} MISSING {path}\n")
            continue
        result = orch.analyze(PipelineRequest(image_path=path, options=opts))
        ve = result.verified_evidence
        scene = ve.scene
        caption = (
            getattr(result.caption, "canonical_caption_en", None)
            or result.caption.text
            or ""
        )
        acts = [a.activity for a in ve.activities if a.qa_safe][:3]
        print(f"## {name}")
        print(f"indoor_outdoor: {scene.indoor_outdoor}")
        print(f"scene_type: {scene.scene_type}")
        print(f"setting: {scene.setting}")
        print(f"activities: {acts}")
        print(f"caption: {caption[:220]}")
        print()


if __name__ == "__main__":
    main()
