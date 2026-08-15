"""Caption quality overhaul — real pipeline stage dump."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

IMAGES = [
    ROOT / "tmp" / "uploads" / "random_268119.jpg",
    ROOT / "tmp" / "uploads" / "random_385406.jpg",
    ROOT / "tmp" / "uploads" / "random_240850.jpg",
    ROOT / "tmp" / "uploads" / "random_891568.jpg",
]
OUT = ROOT / "tmp" / "caption_quality_overhaul_dump.json"


def main() -> int:
    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest

    startup = StartupOrchestrator().run()
    pipe = startup.context.main_controller.pipeline._orchestrator  # noqa: SLF001

    rows: list[dict] = []
    for image in IMAGES:
        if not image.is_file():
            rows.append({"image": str(image), "error": "missing"})
            continue
        if hasattr(pipe._vision_language, "reset_execution_count"):
            pipe._vision_language.reset_execution_count()

        # Hook natural caption internals via temporary monkeypatch logging.
        natural = pipe._natural_caption
        stages: dict[str, str] = {}
        original_generate = natural.generate

        def _generate(image_obj, understanding, context=None, _orig=original_generate, _stages=stages):
            scene = natural._build_semantic_scene(understanding)
            story = natural._story_facts(understanding, scene=scene)
            brief = natural._build_understanding_brief(story, scene, understanding)
            raw = natural._compose_scene_narrative(story, scene=scene, brief=brief)
            _stages["template_raw"] = raw
            _stages["defining_interaction"] = scene.defining_interaction
            _stages["people"] = ",".join(story.people)
            try:
                narrated = natural._vision.narrate(image_obj, understanding)
                _stages["vlm_narrate"] = (narrated.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                _stages["vlm_narrate"] = f"<error: {exc}>"
            final = _orig(image_obj, understanding, context=context)
            _stages["natural_final"] = final
            return final

        natural.generate = _generate  # type: ignore[method-assign]
        try:
            result = pipe.analyze(
                PipelineRequest(
                    image_path=image,
                    options=AnalysisOptions(
                        enable_gemma=True,
                        enable_enhancement=True,
                        enable_sam2=False,
                    ),
                )
            )
        finally:
            natural.generate = original_generate  # type: ignore[method-assign]

        locked = result.caption
        people = sum(
            1
            for n in result.scene_context.graph.nodes
            if n.label.lower() in {"person", "man", "woman", "child"}
        )
        rows.append(
            {
                "image": image.name,
                "people_detected": people,
                "objects": [n.label for n in result.scene_context.graph.nodes[:12]],
                "stages": dict(stages),
                "locked_caption": locked.text,
                "canonical_en": locked.canonical_caption_en,
                "sources": list(locked.sources),
                "robotic_failure": (
                    "a person talking to a person" in (locked.text or "").lower()
                    or "second person stands farther back in the frame"
                    in (locked.text or "").lower()
                ),
                "word_count": len((locked.text or "").split()),
            }
        )
        print("=" * 72)
        print(image.name, "people=", people, "words=", rows[-1]["word_count"])
        print("TEMPLATE:", stages.get("template_raw", "")[:300])
        print("VLM:", (stages.get("vlm_narrate") or "")[:300])
        print("NATURAL:", stages.get("natural_final", "")[:300])
        print("LOCKED:", (locked.text or "")[:400])
        print("ROBOTIC:", rows[-1]["robotic_failure"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
