"""Ollama client must send keep_alive and track call purposes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from analysis.activity.ollama_client import OllamaClient, ollama_call_stats, reset_ollama_call_stats


def test_generate_json_includes_keep_alive_and_counts_semantic() -> None:
    reset_ollama_call_stats()
    client = OllamaClient(model="gemma3:4b", keep_alive="30m")
    payload = {
        "response": '{"caption":"ok"}',
        "model": "gemma3:4b",
        "total_duration": 1_000_000_000,
        "load_duration": 100_000_000,
        "prompt_eval_count": 120,
        "eval_count": 40,
        "eval_duration": 800_000_000,
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    with patch("urllib.request.urlopen", return_value=mock_resp) as urlopen:
        result = client.generate_json("sys", "user", max_tokens=220, purpose="semantic")
    assert result.text
    body = json.loads(urlopen.call_args[0][0].data.decode("utf-8"))
    assert body["keep_alive"] == "30m"
    assert body["options"]["num_predict"] == 220
    stats = ollama_call_stats()
    assert stats.semantic_calls == 1
    assert stats.translation_calls == 0
    assert stats.last_prompt_tokens == 120
    assert stats.last_output_tokens == 40


def test_generate_text_counts_translation() -> None:
    reset_ollama_call_stats()
    client = OllamaClient(keep_alive="30m")
    payload = {"response": "hola", "model": "gemma3:4b"}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    with patch("urllib.request.urlopen", return_value=mock_resp):
        client.generate_text(system="s", user="u", purpose="translation")
    stats = ollama_call_stats()
    assert stats.translation_calls == 1
    assert stats.semantic_calls == 0
