"""Forensic color/activity audit — horse, motorcycle, soccer. No production edits."""
from __future__ import annotations

from pathlib import Path

from app.startup.orchestrator import StartupOrchestrator
from core.contracts.pipeline import AnalysisOptions, PipelineRequest
from language.assistant.evidence_packet import build_evidence_packet
from language.assistant.entity_indexing import find_person_attribute, ordered_people, resolve_person_reference
from language.assistant.vision_assistant import VisionAssistant, VisionAssistantSession
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever


CASES = [
    ("horse", Path("tmp/uploads/10815824_2997e03d76.jpg")),
    ("motorcycle", Path("tmp/uploads/143552829_72b6ba49d4.jpg")),
    ("soccer", Path("tmp/uploads/47871819_db55ac4699.jpg")),
]


def dump_case(name: str, path: Path, orch, va: VisionAssistant) -> None:
    print("=" * 72)
    print("CASE", name, path)
    opts = AnalysisOptions(
        enable_gemma=False,
        competition_mode=False,
        enable_enhancement=False,
        enable_super_resolution=False,
        enable_sam2=False,
    )
    result = orch.analyze(PipelineRequest(image_path=path, options=opts))
    ve = result.verified_evidence
    ctx = result.scene_context
    cap = getattr(result.caption, "canonical_caption_en", None) or result.caption.text
    print("CAPTION:", cap)

    print("\n--- RAW SceneContext attributes (pre-verified) ---")
    if ctx is not None:
        for a in ctx.attributes.attributes:
            if a.name in {"confidence", "visibility"}:
                continue
            if "color" not in a.name and a.name not in {"clothing_type", "clothing_style"}:
                continue
            node = next((n for n in ctx.graph.nodes if n.index == a.object_index), None)
            lab = node.label if node else "?"
            print(f"  obj#{a.object_index}({lab}).{a.name}={a.value}")

    if ve is None:
        print("NO verified_evidence")
        return

    print("\n--- VERIFIED person/object colors ---")
    for attr in ve.attributes:
        if "color" not in attr.name:
            continue
        if not (
            attr.entity_id.startswith("person")
            or "horse" in attr.entity_id
            or "motorcycle" in attr.entity_id
            or "ball" in attr.entity_id
        ):
            continue
        print(
            f"  {attr.entity_id}.{attr.name}={attr.value} "
            f"status={attr.status.value} src={attr.source} conf={attr.confidence:.2f} "
            f"narr={attr.narrative_safe} qa={attr.qa_safe}"
        )

    print("\n--- VERIFIED activities ---")
    for a in ve.activities:
        print(
            f"  [{a.evidence_level.value}] {a.activity!r} narr={a.narrative_safe} "
            f"qa={a.qa_safe} ents={a.entity_ids} src={a.source} support={a.supporting_relations}"
        )

    print("\n--- Rejected color-related ---")
    for r in ve.rejected:
        if "color" in r.predicate or "clothing" in r.predicate or "color" in r.reason:
            print(f"  {r.subject}.{r.predicate}={r.value} reason={r.reason} src={r.source}")

    packet = build_evidence_packet(verified_evidence=ve, canonical_caption_en=cap)
    people = ordered_people(packet)
    print("\n--- ordered_people ---")
    for p in people:
        print(f"  #{p.ordinal} {p.entity_id} area={p.area_ratio:.4f} zone={p.position_zone} conf={p.confidence:.2f}")

    retriever = VisualEvidenceRetriever()
    q_color = "What color clothing is the person wearing?"
    resolved = resolve_person_reference(q_color, packet)
    print("\n--- clothing QA resolution ---")
    print("resolve_person_reference ->", resolved)
    if resolved is not None:
        for preds in (("shirt_color", "clothing_color"), ("clothing_color", "shirt_color"), ("color",), ("dominant_color",)):
            item = find_person_attribute(packet, resolved, predicates=preds, require_reliable=True)
            print(f"  find_person_attribute{preds} ->", None if item is None else f"{item.predicate}={item.value} status={item.claim_status} conf={item.confidence}")
        pair = retriever._safe_person_clothing_color(packet, resolved, question=q_color)
        print("  _safe_person_clothing_color ->", pair)

    session = VisionAssistantSession(image_key=name, evidence=packet)
    questions = [
        "What color clothing is the person wearing?",
        "What are they doing?",
        "What is the person doing?",
        "How many people are visible?",
        "What readable text appears in the scene?",
        "What color is the sports ball?",
        "What color is the horse?",
    ]
    print("\n--- QA ---")
    for q in questions:
        ans = va.answer(session, q)
        print(f"Q: {q}")
        print(f"A: {ans}")


def main() -> None:
    startup = StartupOrchestrator().run()
    orch = startup.context.main_controller.pipeline._orchestrator
    va = VisionAssistant()
    for name, path in CASES:
        dump_case(name, path, orch, va)


if __name__ == "__main__":
    main()
