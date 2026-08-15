"""Live re-verify indoor person retention + suggested question path."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

from app.startup.orchestrator import StartupOrchestrator  # noqa: E402
from core.contracts.pipeline import AnalysisOptions, PipelineRequest  # noqa: E402
from language.assistant import (  # noqa: E402
    VisionAssistant,
    VisionAssistantSession,
    build_evidence_packet,
    generate_suggested_questions,
)


def main() -> int:
    img = ROOT / "tmp" / "uploads" / "random_268119.jpg"
    if not img.exists():
        print(json.dumps({"error": f"missing image: {img}"}))
        return 2

    startup = StartupOrchestrator().run()
    pipe = startup.context.main_controller.pipeline._orchestrator
    if hasattr(pipe._vision_language, "reset_execution_count"):
        pipe._vision_language.reset_execution_count()

    result = pipe.analyze(
        PipelineRequest(
            image_path=img,
            options=AnalysisOptions(
                enable_gemma=True,
                enable_enhancement=True,
                enable_sam2=True,
            ),
        )
    )
    cap = result.caption.canonical_caption_en or result.caption.text
    packet = build_evidence_packet(
        result.scene_context,
        canonical_caption_en=cap,
        evidence_brief=result.evidence_brief,
        ocr_snippets=result.ocr_snippets,
        verified_evidence=result.verified_evidence,
    )
    qs = generate_suggested_questions(packet, language="en", limit=1)
    sess = VisionAssistantSession(image_key="indoor", evidence=packet)
    assistant = VisionAssistant(client=None)
    payload = {
        "caption": cap,
        "suggestions": qs,
        "doing": assistant.answer(sess, "What is the person doing?", language="en"),
        "near": assistant.answer(sess, "What objects are near the person?", language="en"),
        "shoes": assistant.answer(sess, "What color are the shoes?", language="en"),
        "vlm": result.initial_vlm_calls or result.metrics.vlm_executions,
        "assistant_vlm": sess.assistant_vlm_calls,
        "primary_entities": list(result.scene_context.dominant_objects),
        "object_count": result.scene_context.object_count,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
