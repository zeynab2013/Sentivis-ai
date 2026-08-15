"""High-level semantic synthesis via Ollama (caption only — no activity invention)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from analysis.activity.ollama_client import OllamaClient, ollama_call_stats, reset_ollama_call_stats
from analysis.semantic.semantic_evidence_prompt import SYSTEM_PROMPT, build_semantic_reasoning_prompt
from analysis.semantic.semantic_response_parser import ParsedSemanticReasoning, parse_semantic_response
from core.config.analysis_config import AnalysisConfig, SemanticReasoningConfig
from core.contracts.analysis import SceneContext
from core.contracts.language import RawCaption, VisualObservations
from core.logging import get_logger
from language.validation.caption_validator import CaptionEvidenceValidator

logger = get_logger(__name__)


@dataclass(frozen=True)
class SemanticReasoningResult:
    """High-level semantic synthesis from verified evidence."""

    caption: RawCaption | None
    scene_explanation: str
    rejected_conclusions: tuple[str, ...]
    contradictions_resolved: tuple[str, ...]
    source: str


class SemanticReasoningService:
    """Uses Ollama ONLY for scene explanation and caption synthesis."""

    def __init__(self, analysis_config: AnalysisConfig) -> None:
        self._config = analysis_config
        self._semantic_cfg: SemanticReasoningConfig = analysis_config.semantic_reasoning
        self._validator = CaptionEvidenceValidator()
        # Reuse one client so keep_alive can retain the model between calls.
        self._client = OllamaClient(
            base_url=self._semantic_cfg.base_url,
            model=self._semantic_cfg.model,
            timeout_seconds=self._semantic_cfg.timeout_seconds,
            keep_alive="30m",
        )
        self._semantic_call_count = 0

    @property
    def enabled(self) -> bool:
        return self._semantic_cfg.enabled

    @property
    def prefer_over_gemma(self) -> bool:
        return self._semantic_cfg.prefer_over_gemma

    @property
    def semantic_call_count(self) -> int:
        return self._semantic_call_count

    def reset_call_count(self) -> None:
        self._semantic_call_count = 0

    def synthesize(
        self,
        context: SceneContext,
        observations: VisualObservations | None,
        *,
        mode: str | None = None,
    ) -> SemanticReasoningResult:
        """Produce evidence-backed caption; activities in context are preserved unchanged."""
        resolved = (mode or os.environ.get("SENTIVIS_SEMANTIC_MODE") or self._semantic_cfg.mode).lower()
        if not self._semantic_cfg.enabled or resolved in {"off", "heuristic", "none"}:
            return SemanticReasoningResult(
                caption=None,
                scene_explanation="",
                rejected_conclusions=(),
                contradictions_resolved=(),
                source="disabled",
            )
        try:
            return self._synthesize_with_ollama(context, observations)
        except Exception as exc:
            logger.warning("Ollama semantic synthesis failed: %s", exc)
            if self._semantic_cfg.fallback_to_context_caption:
                return SemanticReasoningResult(
                    caption=None,
                    scene_explanation="",
                    rejected_conclusions=(),
                    contradictions_resolved=(),
                    source="fallback",
                )
            raise

    def _synthesize_with_ollama(
        self,
        context: SceneContext,
        observations: VisualObservations | None,
    ) -> SemanticReasoningResult:
        user_prompt = build_semantic_reasoning_prompt(context, observations)
        # Primary model first; only cascade on hard failure (not on success path).
        candidates = tuple(dict.fromkeys((self._semantic_cfg.model, *self._semantic_cfg.models)))
        last_error: Exception | None = None
        for model in candidates:
            try:
                if self._client.model != model:
                    self._client = OllamaClient(
                        base_url=self._semantic_cfg.base_url,
                        model=model,
                        timeout_seconds=self._semantic_cfg.timeout_seconds,
                        keep_alive="30m",
                    )
                response = self._client.generate_json(
                    SYSTEM_PROMPT,
                    user_prompt,
                    max_tokens=480,
                    purpose="semantic",
                )
                self._semantic_call_count += 1
                stats = ollama_call_stats()
                logger.info(
                    "Ollama semantic synthesis model=%s "
                    "prompt_chars=%d prompt_tokens=%d output_tokens=%d load_ms=%.0f eval_ms=%.0f "
                    "semantic_calls=%d",
                    model,
                    len(SYSTEM_PROMPT) + len(user_prompt),
                    stats.last_prompt_tokens,
                    stats.last_output_tokens,
                    stats.last_load_duration_ms,
                    stats.last_eval_duration_ms,
                    self._semantic_call_count,
                )
                parsed: ParsedSemanticReasoning = parse_semantic_response(response.text)
                caption = self._validated_caption(parsed.caption, context)
                caption_len = len(caption.text if caption else "")
                logger.info("Ollama semantic synthesis model=%s caption_len=%d", model, caption_len)
                return SemanticReasoningResult(
                    caption=caption,
                    scene_explanation=parsed.scene_explanation,
                    rejected_conclusions=parsed.rejected_conclusions,
                    contradictions_resolved=parsed.contradictions_resolved,
                    source=f"ollama:{response.model}",
                )
            except Exception as exc:
                last_error = exc
                logger.warning("Ollama semantic model %s failed: %s", model, exc)
                # Do not cascade through the entire model list on every failure —
                # only the primary gemma3:4b is required for normal analysis.
                if model == self._semantic_cfg.model:
                    break
        if last_error:
            raise last_error
        raise RuntimeError("No Ollama models configured for semantic reasoning")

    def _validated_caption(self, text: str, context: SceneContext) -> RawCaption | None:
        if not text.strip():
            return None
        filtered = self._validator.filter_unsupported_sentences(text, context)
        if not filtered.strip():
            return None
        return RawCaption(text=filtered, source="ollama_semantic", confidence=0.78)


__all__ = [
    "SemanticReasoningResult",
    "SemanticReasoningService",
    "reset_ollama_call_stats",
    "ollama_call_stats",
]
