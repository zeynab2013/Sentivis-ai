"""Development-only vision debug dumps (masks, crops, colors, clothing, importance).

Usage:
  python scripts/dump_vision_debug.py path/to/image.jpg

Writes under <image_dir>/.sentivis_cache/vision_debug/<stem>/
Does not change the Streamlit UI.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SENTIVIS_TEST_MODE", "1")

from PIL import Image

from analysis.scene_reasoner.scene_reasoner import SceneReasoner
from app.container import DependencyContainer
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config
from core.contracts.pipeline import AnalysisOptions, PipelineRequest


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/dump_vision_debug.py <image>")
        raise SystemExit(2)
    image_path = Path(sys.argv[1]).resolve()
    if not image_path.is_file():
        print(f"Missing image: {image_path}")
        raise SystemExit(1)

    ctx = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    orch = ctx.main_controller.pipeline._orchestrator  # noqa: SLF001
    result = orch.analyze(
        PipelineRequest(
            image_path=image_path,
            options=AnalysisOptions(
                competition_mode=True,
                enable_enhancement=True,
                enable_sam2=True,
                enable_gemma=False,
            ),
        )
    )

    out_dir = image_path.parent / ".sentivis_cache" / "vision_debug" / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Re-run crop path on ORIGINAL pixels for color truthing.
    preprocessed = None
    # Pull detections from scene graph labels + boxes is incomplete; re-analyze via container pieces.
    # Use attributes + graph from result for the dump, and enhanced preview if present.
    enhanced = result.enhanced_preview_path
    if enhanced and Path(enhanced).is_file():
        Image.open(enhanced).save(out_dir / "enhanced.png")

    caption = result.caption.narrative_full or result.caption.text
    (out_dir / "caption.txt").write_text(caption, encoding="utf-8")

    # Attribute / object dump
    rows: list[dict[str, object]] = []
    attrs_by_index: dict[int, dict[str, str]] = {}
    for attr in result.scene_context.attributes.attributes:
        attrs_by_index.setdefault(attr.object_index, {})[attr.name] = attr.value

    for node in result.scene_context.graph.nodes:
        values = attrs_by_index.get(node.index, {})
        rows.append(
            {
                "index": node.index,
                "label": node.label,
                "zone": node.position_zone,
                "area_ratio": node.bounding_box_area_ratio,
                "dominant_color": values.get("dominant_color") or values.get("color"),
                "shirt_color": values.get("shirt_color"),
                "pants_color": values.get("pants_color"),
                "clothing_type": values.get("clothing_type"),
                "jacket": values.get("jacket"),
                "hoodie": values.get("hoodie"),
                "coat": values.get("coat"),
                "segmentation": values.get("segmentation"),
            }
        )
    (out_dir / "objects.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Importance ranking via SceneReasoner on the same context (facts not on PipelineResult).
    understanding = SceneReasoner().reason(result.scene_context)
    (out_dir / "evidence_brief.txt").write_text(understanding.evidence_brief, encoding="utf-8")
    facts = [
        {
            "subject": f.subject,
            "predicate": f.predicate,
            "value": f.value,
            "confidence": f.confidence,
            "source": f.source,
        }
        for f in understanding.facts
        if f.confidence >= 0.55
    ]
    (out_dir / "high_confidence_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    (out_dir / "ranked_subjects.json").write_text(
        json.dumps(list(understanding.ranked_subjects), indent=2), encoding="utf-8"
    )

    # Fact coverage vs caption
    lower = caption.lower()
    missing = [
        f
        for f in facts
        if f["predicate"] not in {"is", "visibility", "occlusion", "confidence", "segmentation"}
        and str(f["value"]).lower() not in lower
        and str(f["value"]).lower().replace("_", " ") not in lower
    ]
    (out_dir / "missing_from_caption.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")

    print(f"Wrote debug dump → {out_dir}")
    print(f"Caption: {caption}")
    print(f"High-conf facts: {len(facts)} | Missing from caption: {len(missing)}")


if __name__ == "__main__":
    main()
