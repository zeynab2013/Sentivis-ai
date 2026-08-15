"""Quick workstation quality check after final hardening."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["SENTIVIS_UI_LANGUAGE"] = "en"

from core.contracts.language import RawCaption
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.suggested_questions import generate_suggested_questions
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.refinement.caption_refiner import clear_ui_language_cache
from language.semantic.natural_caption_service import NaturalCaptionService

# Import helpers without package path issues.
sys.path.insert(0, str(ROOT / "tests" / "unit" / "language"))
import test_final_hardening_regressions as t  # noqa: E402


class _V:
    def narrate(self, image, understanding):
        return RawCaption(text="a person at a desk", source="stub", confidence=0.4)


def main() -> None:
    clear_ui_language_cache()
    cap = NaturalCaptionService(_V()).generate(  # type: ignore[arg-type]
        t._image(), t._workstation_understanding(), context=t._workstation_context()
    )
    print("CAPTION:", cap)
    packet = build_evidence_packet(
        t._workstation_context(),
        canonical_caption_en=cap,
        evidence_brief="person shirt charcoal; keyboard; navy chair; tv",
    )
    print("SUGGESTION:", generate_suggested_questions(packet, language="en", limit=1))
    session = VisionAssistantSession(image_key="ws", evidence=packet)
    answer = VisionAssistant(client=None).answer(  # type: ignore[arg-type]
        session, "What color is the man's t-shirt?", language="en"
    )
    print("ANSWER:", answer)
    print("ASSISTANT_VLM:", session.assistant_vlm_calls)
    lower = cap.lower()
    assert "main work underway" not in lower
    assert "setting remains" not in lower
    assert lower.count("keyboard") <= 1
    assert "charcoal tv" not in lower
    assert "charcoal" in answer.lower()
    assert session.assistant_vlm_calls == 0
    print("OK")


if __name__ == "__main__":
    main()
