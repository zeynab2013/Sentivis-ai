"""Vision Assistant — evidence-grounded Q&A with Gemma 3 4B (no VLM re-run)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from analysis.activity.ollama_client import OllamaClient
from core.logging import get_logger
from language.assistant.evidence_packet import AssistantEvidencePacket
from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever

logger = get_logger(__name__)

_LANGUAGE_NAMES = {
    "en": "English",
    "fa": "Persian",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
}

_UNKNOWN_EN = (
    "I can't reliably determine that from the image."
)

_UNAVAILABLE_PHRASES = (
    "the caption does not mention",
    "the caption doesn't mention",
    "caption does not mention",
    "caption doesn't mention",
    "not mentioned in the caption",
    "according to the caption",
    "the caption only says",
    "based on the caption",
    "not in the caption",
    "caption does not include",
    "was not detected",
    "were not detected",
    "this was not detected",
    "that was not detected",
    "the image does not contain this information",
    "no information about",
    "i don't have information",
    "i do not have information",
)

_EVIDENCE_DUMP_MARKERS = (
    "here's a breakdown",
    "here is a breakdown",
    "okay, here's a breakdown",
    "entities (key",
    "key objects",
    "overall scene understanding",
    "evidence packet",
    "verifiedsceneevidence",
    "scenecontext",
    "person_1",
    "person_2",
    "horse_1",
    "horse_2",
    "object_1",
    "claim status",
    "narrative_safe",
    "verification_tier",
    "relationships:",
    "attributes:",
    "entities:",
)

@dataclass
class AssistantTurn:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class VisionAssistantSession:
    """Image-scoped multi-turn memory. Cleared when the image changes."""

    image_key: str
    evidence: AssistantEvidencePacket
    turns: list[AssistantTurn] = field(default_factory=list)
    assistant_vlm_calls: int = 0  # must stay 0 — no vision re-perception
    assistant_llm_calls: int = 0


class VisionAssistant:
    """Answer questions from the frozen evidence packet using Gemma 3 4B text."""

    def __init__(self, client: OllamaClient | None = None) -> None:
        self._client = client or OllamaClient(model="gemma3:4b", timeout_seconds=90.0, keep_alive="30m")
        self._retriever = VisualEvidenceRetriever()

    def answer(
        self,
        session: VisionAssistantSession,
        question: str,
        *,
        language: str = "en",
    ) -> str:
        q = " ".join((question or "").split()).strip()
        if not q:
            return ""
        # Resolve follow-up references against prior turns before retrieval.
        q = self._resolve_followup(q, session.turns)
        session.turns.append(AssistantTurn(role="user", text=q))
        retrieval = self._retriever.retrieve(session.evidence, q)
        history = self._history_block(session.turns[:-1])
        lang = (language or "en").lower()
        language_name = _LANGUAGE_NAMES.get(lang, "English")

        # Prefer deterministic evidence answers for clear attribute/object questions.
        # This prevents caption-chatbot refusals when evidence supports the fact.
        if retrieval.direct_answer_en:
            text = retrieval.direct_answer_en
            if lang != "en":
                text = self._translate_answer(text, language_name) or text
            session.assistant_llm_calls += 1 if lang != "en" and text != retrieval.direct_answer_en else 0
            # Direct path never uses VLM.
            text = self._strip_appended_scene_caption(text, session.evidence)
            logger.info(
                "VisionAssistant direct-evidence answer | q=%r chars=%d items=%d",
                q[:120],
                len(text),
                len(retrieval.selected),
            )
            session.turns.append(AssistantTurn(role="assistant", text=text))
            logger.info(
                "VisionAssistant answered | llm_calls=%d vlm_calls=%d q_chars=%d answer_chars=%d mode=direct",
                session.assistant_llm_calls,
                session.assistant_vlm_calls,
                len(q),
                len(text),
            )
            return text

        # Insufficient evidence: do not invent via LLM when retrieval found nothing reliable.
        if not retrieval.has_reliable_match and not retrieval.selected:
            text = self._unknown(lang)
            session.turns.append(AssistantTurn(role="assistant", text=text))
            logger.info(
                "VisionAssistant insufficient-evidence | q=%r",
                q[:120],
            )
            return text

        system = (
            "You are Sentivis Vision Assistant — an IMAGE-GROUNDED Q&A system.\n"
            "The VISUAL EVIDENCE FACTS block is the ONLY source of truth.\n"
            "Rules:\n"
            "1. Answer the USER QUESTION directly in 1–3 natural sentences.\n"
            "2. Use only RELIABLE evidence facts. Never invent garment types, "
            "emotions, ages, names, or interactions.\n"
            "3. If a GROUNDED ANSWER DRAFT is provided, follow it closely.\n"
            "4. If evidence is insufficient, say you can't reliably determine that "
            "from the image — do not guess.\n"
            "5. NEVER dump Entities / Attributes / Relationships inventories.\n"
            "6. NEVER mention person_1, horse_1, evidence packets, SceneContext, "
            "confidence tables, or internal IDs.\n"
            "7. NEVER say 'here's a breakdown' or 'overall scene understanding'.\n"
            "8. NEVER say 'the caption does not mention' a fact present in evidence.\n"
            "9. Do not invent clothing types (t-shirt/jacket/pants) unless evidence "
            "explicitly lists clothing_type.\n"
            "10. Prefer a short factual answer over analysis essays.\n"
            "11. NEVER append, restate, or paraphrase the scene caption after the answer.\n"
            f"12. Respond in {language_name}."
        )
        user = (
            f"QUESTION:\n{q}\n\n"
            f"CONVERSATION HISTORY:\n{history or '(none)'}\n\n"
            f"{retrieval.prompt_block}\n"
        )
        try:
            logger.info(
                "VisionAssistant retrieve | q=%r evidence_chars=%d items=%d direct=%s",
                q[:120],
                len(retrieval.prompt_block),
                len(retrieval.selected),
                bool(retrieval.direct_answer_en),
            )
            response = self._client.generate_text(
                system=system,
                user=user,
                max_tokens=120,
                purpose="assistant",
            )
            session.assistant_llm_calls += 1
            text = " ".join((response.text or "").split()).strip()
            text = self._scrub_caption_dependence(text, retrieval.has_reliable_match)
            text = self._scrub_evidence_dump(text)
            text = self._strip_appended_scene_caption(text, session.evidence)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vision Assistant LLM failed: %s", exc)
            text = self._unknown(lang)
        if not text:
            text = self._unknown(lang)
        session.turns.append(AssistantTurn(role="assistant", text=text))
        logger.info(
            "VisionAssistant answered | llm_calls=%d vlm_calls=%d q_chars=%d answer_chars=%d mode=llm",
            session.assistant_llm_calls,
            session.assistant_vlm_calls,
            len(q),
            len(text),
        )
        return text

    def _translate_answer(self, text: str, language_name: str) -> str:
        try:
            response = self._client.generate_text(
                system=(
                    f"Translate the answer into {language_name}. "
                    "Preserve all visual facts. Return only the translation."
                ),
                user=text,
                max_tokens=180,
                purpose="translation",
            )
            return " ".join((response.text or "").split()).strip()
        except Exception:  # noqa: BLE001
            return text

    def _resolve_followup(self, question: str, prior_turns: list[AssistantTurn]) -> str:
        """Expand pronouns / 'which one' using the last assistant inventory when possible."""
        q = (question or "").strip()
        lower = q.lower()
        if not prior_turns:
            return q
        # Never rewrite activity questions — "What are they doing?" must keep
        # its activity intent even after an object-inventory turn.
        if any(tok in lower for tok in ("doing", "activity", "action", "happening")):
            return q

        # Only rewrite clear follow-up forms.
        followup_markers = (
            "which one",
            "which of them",
            "which of those",
            "the first",
            "the second",
            "the third",
            "closest to",
            "nearest to",
            " that ",
            " it ",
            " them ",
            "those",
        )
        starts_followup = lower.startswith(("which", "what about", "and the", "how about"))
        if not (starts_followup or any(m in f" {lower} " for m in followup_markers)):
            # Short pronouns-only questions
            if lower not in {"it?", "that?", "them?", "this?", "which one?", "which?"}:
                return q

        # Find the last assistant answer that listed objects.
        last_answer = ""
        last_user = ""
        for turn in reversed(prior_turns):
            if turn.role == "assistant" and not last_answer:
                last_answer = turn.text
            elif turn.role == "user" and not last_user:
                last_user = turn.text
            if last_answer and last_user:
                break
        if not last_answer:
            return q

        # Extract simple noun phrases from prior answer (laptop, cup, notebook…).
        nouns = re.findall(
            r"\b(laptop|keyboard|mouse|monitor|cup|mug|bottle|book|notebook|phone|"
            r"chair|table|desk|bicycle|car|bag|backpack|person|refrigerator|tv|vase|"
            r"dining table|skis|snowboard|motorcycle|horse)\b",
            last_answer.lower(),
        )
        # Deduplicate preserving order.
        seen: list[str] = []
        for n in nouns:
            if n not in seen:
                seen.append(n)
        if not seen:
            return q

        if "which one" in lower or lower in {"which?", "which one?"}:
            return f"Which of these is being referred to: {', '.join(seen)}?"
        if "closest" in lower or "nearest" in lower:
            if "person" in " ".join(seen) or "person" in last_user.lower():
                others = [n for n in seen if n != "person"]
                if others:
                    return f"Which of {', '.join(others)} is closest to the person?"
            return f"Which of {', '.join(seen)} is closest?"
        if lower.startswith("what about") or lower.startswith("how about"):
            return q  # leave specific "what about X" alone
        # Never rewrite they/them to a non-person object for activity-like questions
        # (handled above). For other follow-ups, only rewrite when the antecedent
        # is a single non-person object.
        if re.search(r"\b(it|that)\b", lower) and len(seen) == 1 and seen[0] != "person":
            return re.sub(r"\b(it|that)\b", seen[0], q, flags=re.I)
        return q

    def _scrub_caption_dependence(self, text: str, has_reliable_match: bool) -> str:
        """Replace caption-centric / false-unavailable refusals."""
        lower = (text or "").lower()
        if any(p in lower for p in _UNAVAILABLE_PHRASES):
            if has_reliable_match:
                return (
                    "Based on the visual evidence available for this image, "
                    "I can describe what is supported there — please rephrase if needed."
                )
            return self._unknown("en")
        return text

    def _scrub_evidence_dump(self, text: str) -> str:
        """Reject LLM answers that leak internal evidence packet structure."""
        lower = (text or "").lower()
        if not lower:
            return text
        if any(marker in lower for marker in _EVIDENCE_DUMP_MARKERS):
            logger.warning("VisionAssistant scrubbed evidence-dump answer")
            return self._unknown("en")
        # Structural dump: many labeled sections in one reply.
        section_hits = sum(
            1
            for section in ("entities", "attributes", "relationships", "activities", "environment")
            if re.search(rf"\b{section}\b\s*[:\-]", lower)
        )
        if section_hits >= 2:
            logger.warning("VisionAssistant scrubbed multi-section inventory answer")
            return self._unknown("en")
        return text

    def _strip_appended_scene_caption(self, text: str, evidence) -> str:
        """QA answers must not append the scene caption or a second narrative."""
        from language.assistant.visual_evidence_retriever import VisualEvidenceRetriever

        packet = evidence
        if packet is None:
            return text
        return VisualEvidenceRetriever()._strip_appended_caption(text or "", packet)

    def _history_block(self, turns: list[AssistantTurn]) -> str:
        if not turns:
            return ""
        recent = turns[-6:]
        lines = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.text}")
        return "\n".join(lines)

    def _unknown(self, lang: str) -> str:
        if lang == "en":
            return _UNKNOWN_EN
        fallback = {
            "de": "Das lässt sich aus dem Bild nicht zuverlässig bestimmen.",
            "es": "No puedo determinarlo de forma fiable a partir de la imagen.",
            "fa": "از روی تصویر نمی‌توان این مورد را به‌طور قابل‌اعتماد تعیین کرد.",
            "zh": "根据图像无法可靠地判断这一点。",
        }
        return fallback.get(lang, _UNKNOWN_EN)
