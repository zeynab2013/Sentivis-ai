"""Quick suggested-question + leak check on farm caption packet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from language.assistant import build_evidence_packet, generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession

IMAGE = ROOT / "tmp" / "uploads" / "10815824_2997e03d76.jpg"
OUT = ROOT / "tmp" / "farm_qa_suggestions_audit.json"


def main() -> int:
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
    suggestions = generate_suggested_questions(packet, language="en", limit=5)
    extra_qs = [
        "What color is the horse?",
        "What is the person doing?",
        "Is there a fire visible?",
    ]
    assistant = VisionAssistant()
    extras = {}
    for q in extra_qs:
        session = VisionAssistantSession(image_key=IMAGE.name, evidence=packet)
        extras[q] = assistant.answer(session, q, language="en")
    payload = {
        "caption": caption,
        "suggestions": suggestions,
        "extra_answers": extras,
        "ui_nav_has_dashboard_import": "render_dashboard"
        in (ROOT / "streamlit_app" / "main.py").read_text(encoding="utf-8"),
        "ui_nav_routes_dashboard": 'nav == t("streamlit.nav.dashboard")'
        in (ROOT / "streamlit_app" / "main.py").read_text(encoding="utf-8"),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
