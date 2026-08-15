"""Probe farm-image QA answers without full pipeline re-run when possible."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

IMAGE = ROOT / "tmp" / "uploads" / "10815824_2997e03d76.jpg"
OUT = ROOT / "tmp" / "farm_qa_hardening_probe.json"

QUESTIONS = [
    "Is there fire or smoke visible?",
    "What color clothing is the person wearing?",
    "How many people are visible?",
    "How many horses are visible?",
    "Is the person holding the horse?",
    "What color is the other horse?",
]


def main() -> int:
    from app.startup.orchestrator import StartupOrchestrator
    from core.contracts.pipeline import AnalysisOptions, PipelineRequest
    from language.assistant import build_evidence_packet
    from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession

    if not IMAGE.is_file():
        raise SystemExit(f"missing {IMAGE}")

    startup = StartupOrchestrator().run()
    pipe = startup.context.main_controller.pipeline._orchestrator  # noqa: SLF001
    result = pipe.analyze(
        PipelineRequest(
            image_path=IMAGE,
            options=AnalysisOptions(enable_gemma=True, enable_enhancement=True, enable_sam2=False),
        )
    )
    caption = result.caption.canonical_caption_en or result.caption.text
    packet = build_evidence_packet(
        result.scene_context,
        canonical_caption_en=caption,
        evidence_brief=result.evidence_brief,
        ocr_snippets=result.ocr_snippets,
        verified_evidence=result.verified_evidence,
    )
    assistant = VisionAssistant()
    answers: dict[str, str] = {}
    for q in QUESTIONS:
        session = VisionAssistantSession(image_key=IMAGE.name, evidence=packet)
        answers[q] = assistant.answer(session, q, language="en")

    payload = {
        "image": IMAGE.name,
        "caption": caption,
        "enhancement_status": getattr(
            getattr(result, "image_quality", None), "enhancement_status", None
        )
        or getattr(result, "enhancement_status", None),
        "answers": answers,
        "leak_checks": {
            q: {
                "has_breakdown": "breakdown" in a.lower(),
                "has_person_1": "person_1" in a.lower(),
                "has_entities_header": "entities" in a.lower() and ":" in a.lower(),
            }
            for q, a in answers.items()
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
