"""CPU-first multilingual text-to-speech for displayed UI text.

Reads already-translated display text. Does NOT translate.
Prefers edge-tts (network, CPU) then pyttsx3 / Windows SAPI fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

from core.logging import get_logger
from language.tts.audio_utils import (
    SUPPORTED_TTS_LANGUAGES,
    audio_cache_key,
    detect_audio_format,
    normalize_tts_language,
)

logger = get_logger(__name__)

# Voices match SENTIVIS UI languages (edge-tts Neural).
_EDGE_VOICES: dict[str, str] = {
    "en": "en-US-JennyNeural",
    "fa": "fa-IR-DilaraNeural",
    "es": "es-ES-ElviraNeural",
    "de": "de-DE-KatjaNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}

_CACHE_DIR = Path(tempfile.gettempdir()) / "sentivis_tts_cache"
_VOICE_CONFIG_ID = "edge-v1"


@dataclass(frozen=True)
class SpeechArtifact:
    """Generated speech for one caption+language pair."""

    audio: bytes
    language: str
    voice: str
    cache_key: str
    extension: str
    mime: str


class SpeechService:
    """Synthesize speech for currently displayed text on CPU."""

    def __init__(self) -> None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._backend = self._detect_backend()
        logger.info("TTS backend=%s", self._backend)

    @staticmethod
    def _detect_backend() -> str:
        try:
            import edge_tts  # noqa: F401

            return "edge_tts"
        except Exception:  # noqa: BLE001
            pass
        try:
            import pyttsx3  # noqa: F401

            return "pyttsx3"
        except Exception:  # noqa: BLE001
            pass
        return "none"

    @property
    def available(self) -> bool:
        return self._backend != "none"

    @property
    def supported_languages(self) -> frozenset[str]:
        return SUPPORTED_TTS_LANGUAGES

    def voice_for(self, language: str) -> str:
        lang = normalize_tts_language(language)
        return _EDGE_VOICES[lang]

    def synthesize(self, text: str, language: str = "en") -> bytes | None:
        """Return MP3/WAV audio bytes for ``text`` in ``language``."""
        artifact = self.synthesize_artifact(text, language)
        return artifact.audio if artifact else None

    def synthesize_artifact(self, text: str, language: str = "en") -> SpeechArtifact | None:
        """Synthesize and return metadata-aware artifact (uses disk cache)."""
        body = (text or "").strip()
        if not body:
            logger.info("TTS skipped: empty caption")
            return None
        lang = normalize_tts_language(language)
        if (language or "").lower().strip() not in SUPPORTED_TTS_LANGUAGES and (
            language or ""
        ).lower().strip() not in {"", "en", "zh-cn", "zh-tw", "zh-hans", "zh-hant"}:
            logger.warning(
                "TTS language %r unsupported; using %s voice for synthesis",
                language,
                lang,
            )
        voice = self.voice_for(lang)
        key = audio_cache_key(body, lang, voice_config=f"{_VOICE_CONFIG_ID}:{voice}")
        cache_path = _CACHE_DIR / f"{key}.bin"
        if cache_path.is_file() and cache_path.stat().st_size > 32:
            data = cache_path.read_bytes()
            fmt = detect_audio_format(data)
            return SpeechArtifact(
                audio=data,
                language=lang,
                voice=voice,
                cache_key=key,
                extension=fmt.extension,
                mime=fmt.mime,
            )

        audio: bytes | None = None
        if self._backend == "edge_tts":
            audio = self._edge_tts(body, lang)
        if audio is None and self._backend in {"edge_tts", "pyttsx3"}:
            audio = self._pyttsx3(body, lang)
        if audio is None:
            audio = self._powershell_sapi(body, lang)

        if not audio:
            logger.warning("TTS synthesis failed for lang=%s backend=%s", lang, self._backend)
            return None

        try:
            cache_path.write_bytes(audio)
        except OSError as exc:
            logger.warning("TTS disk cache write failed: %s", exc)

        fmt = detect_audio_format(audio)
        return SpeechArtifact(
            audio=audio,
            language=lang,
            voice=voice,
            cache_key=key,
            extension=fmt.extension,
            mime=fmt.mime,
        )

    def _edge_tts(self, text: str, language: str) -> bytes | None:
        try:
            import edge_tts

            voice = _EDGE_VOICES.get(language, _EDGE_VOICES["en"])

            async def _run() -> bytes:
                communicate = edge_tts.Communicate(text, voice)
                chunks: list[bytes] = []
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk["data"])
                return b"".join(chunks)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        return pool.submit(lambda: asyncio.run(_run())).result(timeout=60)
                return loop.run_until_complete(_run())
            except RuntimeError:
                return asyncio.run(_run())
        except Exception as exc:  # noqa: BLE001
            logger.warning("edge-tts failed (%s): %s", language, exc)
            return None

    def _pyttsx3(self, text: str, language: str) -> bytes | None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            prefix = {"en": "en", "fa": "fa", "es": "es", "de": "de", "zh": "zh"}.get(
                language, "en"
            )
            for voice in engine.getProperty("voices") or []:
                vid = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
                if prefix == "zh" and ("zh" in vid or "chinese" in vid or "huihui" in vid):
                    engine.setProperty("voice", voice.id)
                    break
                if prefix != "en" and prefix in vid:
                    engine.setProperty("voice", voice.id)
                    break
                if prefix == "en" and ("en-" in vid or "english" in vid):
                    engine.setProperty("voice", voice.id)
                    break
            out = _CACHE_DIR / f"_tmp_{hashlib.sha1(text.encode()).hexdigest()[:12]}.wav"
            engine.save_to_file(text, str(out))
            engine.runAndWait()
            if out.is_file():
                data = out.read_bytes()
                try:
                    out.unlink()
                except OSError:
                    pass
                return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("pyttsx3 failed: %s", exc)
        return None

    def _powershell_sapi(self, text: str, language: str) -> bytes | None:
        """Last-resort Windows SAPI via PowerShell (English only)."""
        if language != "en":
            return None
        try:
            import subprocess

            out = _CACHE_DIR / f"_sapi_{hashlib.sha1(text.encode()).hexdigest()[:12]}.wav"
            safe = text.replace("'", "''")[:800]
            script = (
                f"Add-Type -AssemblyName System.Speech; "
                f"$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.SetOutputToWaveFile('{out}'); "
                f"$s.Speak('{safe}'); "
                f"$s.Dispose()"
            )
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                timeout=45,
                check=False,
            )
            if completed.returncode == 0 and out.is_file():
                data = out.read_bytes()
                try:
                    out.unlink()
                except OSError:
                    pass
                return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("SAPI fallback failed: %s", exc)
        return None


_SERVICE: SpeechService | None = None


def get_speech_service() -> SpeechService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = SpeechService()
    return _SERVICE


def synthesize_display_text(text: str, language: str = "en") -> bytes | None:
    """Public helper used by Streamlit TTS buttons."""
    return get_speech_service().synthesize(text, language)


def synthesize_display_artifact(text: str, language: str = "en") -> SpeechArtifact | None:
    """Public helper returning audio + format metadata."""
    return get_speech_service().synthesize_artifact(text, language)
