"""Pure TTS helpers — cache keys, filenames, language normalization (no Streamlit)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Must stay aligned with UI languages and edge-tts voice map.
SUPPORTED_TTS_LANGUAGES: frozenset[str] = frozenset({"en", "fa", "es", "de", "zh"})

_VOICE_CONFIG_ID = "edge-v1"


@dataclass(frozen=True)
class AudioFormatInfo:
    extension: str
    mime: str


def normalize_tts_language(language: str | None) -> str:
    """Map UI language code to a supported TTS language (default en)."""
    code = (language or "en").lower().strip()
    if code.startswith("zh"):
        code = "zh"
    if code not in SUPPORTED_TTS_LANGUAGES:
        return "en"
    return code


def audio_cache_key(
    text: str,
    language: str,
    *,
    voice_config: str = _VOICE_CONFIG_ID,
) -> str:
    """Stable cache key: language + voice config + caption body."""
    lang = normalize_tts_language(language)
    body = (text or "").strip()
    raw = f"{lang}|{voice_config}|{body}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def detect_audio_format(audio: bytes, hint: str | None = None) -> AudioFormatInfo:
    """Detect MP3 vs WAV from magic bytes (and optional MIME/format hint)."""
    if audio[:4] == b"RIFF" or (hint and "wav" in hint.lower()):
        return AudioFormatInfo(extension="wav", mime="audio/wav")
    return AudioFormatInfo(extension="mp3", mime="audio/mpeg")


def download_filename(language: str, cache_key: str, extension: str) -> str:
    """Safe download name: caption_<lang>_<short-id>.<ext> (no caption text, no paths)."""
    lang = normalize_tts_language(language)
    short = re.sub(r"[^a-zA-Z0-9]", "", (cache_key or ""))[:8] or "audio"
    ext = re.sub(r"[^a-z0-9]", "", (extension or "mp3").lower()) or "mp3"
    # Block path separators / traversal regardless of inputs.
    name = f"caption_{lang}_{short}.{ext}"
    name = name.replace("\\", "_").replace("/", "_").replace("..", "_")
    if len(name) > 80:
        name = name[:76] + f".{ext}"
    return name


def is_valid_caption_for_tts(text: str | None) -> bool:
    return bool((text or "").strip())
