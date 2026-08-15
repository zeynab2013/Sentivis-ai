"""HTTP client for Ollama structured generation."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OllamaResponse:
    """Parsed Ollama generate/chat response."""

    text: str
    model: str
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0


@dataclass
class OllamaCallStats:
    """Process-wide Ollama call counters for diagnostics."""

    semantic_calls: int = 0
    translation_calls: int = 0
    other_calls: int = 0
    last_load_duration_ms: float = 0.0
    last_eval_duration_ms: float = 0.0
    last_prompt_tokens: int = 0
    last_output_tokens: int = 0
    last_wall_ms: float = 0.0


_STATS = OllamaCallStats()


def ollama_call_stats() -> OllamaCallStats:
    return _STATS


def reset_ollama_call_stats() -> None:
    global _STATS
    _STATS = OllamaCallStats()


class OllamaClient:
    """Minimal Ollama REST client (no raw pixels — text/JSON only)."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "gemma3:4b",
        timeout_seconds: float = 120.0,
        keep_alive: str | int = "30m",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._keep_alive = keep_alive

    @property
    def model(self) -> str:
        return self._model

    def generate_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 220,
        purpose: str = "semantic",
    ) -> OllamaResponse:
        """Request JSON-formatted completion from Ollama."""
        payload = {
            "model": self._model,
            "prompt": f"{system}\n\n{user}",
            "stream": False,
            "format": "json",
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens,
                # Prefer shorter deterministic completions for structured JSON.
                "num_ctx": 4096,
            },
        }
        return self._post("/api/generate", payload, purpose=purpose)

    def generate_text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 280,
        purpose: str = "translation",
    ) -> OllamaResponse:
        """Text-only generation (no image) — used for caption translation."""
        payload = {
            "model": self._model,
            "prompt": f"{system}\n\n{user}",
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"temperature": 0.1, "num_predict": max_tokens, "num_ctx": 4096},
        }
        return self._post("/api/generate", payload, purpose=purpose)

    def generate_with_image(
        self,
        prompt: str,
        image_b64: str,
        *,
        max_tokens: int = 220,
        temperature: float = 0.1,
        purpose: str = "vision",
    ) -> OllamaResponse:
        """Multimodal generate for Gemma Vision / other Ollama vision tags."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        return self._post("/api/generate", payload, purpose=purpose)

    def _post(self, path: str, payload: dict[str, object], *, purpose: str) -> OllamaResponse:
        url = f"{self._base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Ollama request failed: {exc}") from exc
        wall_ms = (time.perf_counter() - started) * 1000.0

        text = str(data.get("response", "")).strip()
        model = str(data.get("model", self._model))
        total_ns = int(data.get("total_duration") or 0)
        load_ns = int(data.get("load_duration") or 0)
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        eval_tokens = int(data.get("eval_count") or 0)
        eval_ns = int(data.get("eval_duration") or 0)

        _STATS.last_wall_ms = wall_ms
        _STATS.last_load_duration_ms = load_ns / 1_000_000.0
        _STATS.last_eval_duration_ms = eval_ns / 1_000_000.0
        _STATS.last_prompt_tokens = prompt_tokens
        _STATS.last_output_tokens = eval_tokens
        if purpose == "semantic":
            _STATS.semantic_calls += 1
        elif purpose == "translation":
            _STATS.translation_calls += 1
        else:
            _STATS.other_calls += 1

        logger.info(
            "Ollama %s model=%s wall_ms=%.0f load_ms=%.0f eval_ms=%.0f "
            "prompt_tokens=%d output_tokens=%d keep_alive=%s",
            purpose,
            model,
            wall_ms,
            _STATS.last_load_duration_ms,
            _STATS.last_eval_duration_ms,
            prompt_tokens,
            eval_tokens,
            self._keep_alive,
        )
        return OllamaResponse(
            text=text,
            model=model,
            total_duration_ns=total_ns,
            load_duration_ns=load_ns,
            prompt_eval_count=prompt_tokens,
            eval_count=eval_tokens,
            eval_duration_ns=eval_ns,
        )
