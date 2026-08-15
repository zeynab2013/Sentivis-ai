"""Targeted forensic: horse colors + motorcycle QA activity packet."""
from __future__ import annotations

from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.assistant.entity_indexing import ordered_people


def run_case(name: str, path: Path, questions: list[str]) -> None:
    print("=" * 72)
    print("CASE", name, path)
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
    verified = result.verified_evidence
    caption = (
        getattr(result.caption, "canonical_caption_en", None)
        or getattr(result.caption, "text", None)
        or ""
    )
    print("CAPTION:", caption[:400])
    if verified is None:
        print("NO verified_evidence")
        return
    print("people_count:", verified.people_count)
    print("ACTIVITIES:")
    for a in verified.activities:
        print(
            f"  [{a.evidence_level.value}] {a.activity!r} narr={a.narrative_safe} "
            f"qa={a.qa_safe} conf={a.confidence:.2f} ents={a.entity_ids} src={a.source}"
        )
    print("PERSON COLOR ATTRS:")
    for attr in verified.attributes:
        if not attr.entity_id.startswith("person"):
            continue
        if "color" not in attr.name:
            continue
        print(
            f"  {attr.entity_id}.{attr.name}={attr.value} "
            f"status={attr.status.value} src={attr.source} conf={attr.confidence:.2f} "
            f"narr={attr.narrative_safe} qa={attr.qa_safe}"
        )
    print("HORSE COLOR ATTRS:")
    for attr in verified.attributes:
        if "horse" not in attr.entity_id:
            continue
        if "color" not in attr.name:
            continue
        print(
            f"  {attr.entity_id}.{attr.name}={attr.value} "
            f"status={attr.status.value} src={attr.source} conf={attr.confidence:.2f}"
        )
    packet = build_evidence_packet(
        verified_evidence=verified,
        canonical_caption_en=caption,
    )
    print("PACKET activities:")
    for item in packet.items:
        if item.kind != "activity":
            continue
        print(
            f"  value={item.value!r} level={item.evidence_level!r} "
            f"status={item.claim_status!r} conf={item.confidence:.2f} "
            f"eid={item.entity_id!r} reliable={item.reliable}"
        )
    print("PACKET clothing attrs:")
    for item in packet.items:
        if item.kind != "attribute":
            continue
        if "color" not in item.predicate:
            continue
        if not (item.entity_id.startswith("person") or "person" in (item.subject or "").lower()):
            continue
        print(
            f"  {item.entity_id}.{item.predicate}={item.value} "
            f"status={item.claim_status} conf={item.confidence:.2f} reliable={item.reliable}"
        )
    va = VisionAssistant()
    session = VisionAssistantSession(
        image_key=name,
        evidence=packet,
    )
    print("ordered_people:", len(ordered_people(packet)))
    for q in questions:
        ans = va.answer(session, q)
        print(f"Q: {q}")
        print(f"A: {ans}")


def main() -> None:
    run_case(
        "horse",
        Path("tmp/uploads/10815824_2997e03d76.jpg"),
        [
            "What color clothing is the person wearing?",
            "What color clothing is the first person wearing?",
            "What color clothing is the second person wearing?",
            "What color is the horse?",
            "What is the person doing?",
        ],
    )
    run_case(
        "motorcycle",
        Path("tmp/uploads/143552829_72b6ba49d4.jpg"),
        [
            "What are they doing?",
            "What is the person doing?",
            "What is the person doing in this scene?",
            "What activity is the person performing?",
            "What color clothing is the person wearing?",
        ],
    )


if __name__ == "__main__":
    main()
