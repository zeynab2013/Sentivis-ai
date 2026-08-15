"""Kitchen production-image validation: counts, color, naturalness."""

from __future__ import annotations

import gc
import re
from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from language.validation.caption_factuality import label_counts_from_verified


def main() -> None:
    path = Path("tmp/uploads/5e4dc9b3-589c-43c6-84b0-5620936c9df4.png")
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
    fridge_color = "missing"
    if ve is not None:
        for a in ve.attributes:
            ent = ve.entity_by_id(a.entity_id)
            if ent and ent.label.lower() == "refrigerator" and "color" in a.name.lower():
                if a.name in {"color", "dominant_color"}:
                    fridge_color = a.value
                    break
    lower = cap.lower()
    checks = {
        "fridge_count_1": counts.get("refrigerator") == 1,
        "chair_count_4": counts.get("chair") == 4,
        "person_count_2": counts.get("person") == 2,
        "kitchen_env": "kitchen" in lower or (ve and "kitchen" in (ve.scene.setting or "").lower()),
        "no_inflated_fridge": not bool(re.search(r"\b[2-9]\s+\w*\s*refrigerator", lower)),
        "fridge_not_brown_beige": fridge_color not in {"brown", "beige", "tan", "cream"},
        "fridge_color_ok": fridge_color in {"white", "gray", "unknown", "light gray", "black"},
        "no_person_and_person": "person and person" not in lower,
        "no_gender_noun": not bool(re.search(r"\b(?:man|woman|girl|boy)\b", lower)),
        "natural_not_census_only": not lower.strip().startswith("detected objects"),
    }
    print("CAPTION:", cap)
    print("COUNTS:", counts)
    print("FRIDGE_COLOR:", fridge_color)
    print("CHECKS:", checks)
    print("PASS" if all(checks.values()) else "FAIL")
    out = Path("tmp/FINAL_KITCHEN_NATURALNESS_VALIDATION.md")
    lines = [
        "# Kitchen naturalness + count + color validation",
        "",
        f"- CAPTION: {cap}",
        f"- COUNTS: {counts}",
        f"- FRIDGE_COLOR: {fridge_color}",
        "- CHECKS:",
    ]
    for k, v in checks.items():
        lines.append(f"  - {k}: {'PASS' if v else 'FAIL'}")
    lines.append(f"- RESULT: {'PASS' if all(checks.values()) else 'FAIL'}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out)
    del result
    gc.collect()


if __name__ == "__main__":
    main()
