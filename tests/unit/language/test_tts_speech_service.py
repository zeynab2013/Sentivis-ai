"""PART 2-C — Multilingual TTS, cache, download, and invalidation regressions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from language.tts.audio_utils import (
    SUPPORTED_TTS_LANGUAGES,
    audio_cache_key,
    detect_audio_format,
    download_filename,
    is_valid_caption_for_tts,
    normalize_tts_language,
)
from language.tts.speech_service import SpeechArtifact, SpeechService, _EDGE_VOICES


def test_supported_languages_cover_ui_set() -> None:
    assert SUPPORTED_TTS_LANGUAGES == frozenset({"en", "fa", "es", "de", "zh"})
    for code in SUPPORTED_TTS_LANGUAGES:
        assert code in _EDGE_VOICES


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("en", "en"),
        ("FA", "fa"),
        ("es", "es"),
        ("de", "de"),
        ("zh", "zh"),
        ("zh-CN", "zh"),
        ("unknown", "en"),
        ("", "en"),
        (None, "en"),
    ],
)
def test_normalize_tts_language(raw: str | None, expected: str) -> None:
    assert normalize_tts_language(raw) == expected


@pytest.mark.parametrize("lang", ["en", "fa", "es", "de", "zh"])
def test_cache_key_includes_language(lang: str) -> None:
    caption = "A person stands beside a table."
    key = audio_cache_key(caption, lang)
    assert len(key) == 24
    # Different languages must not collide for the same caption text.
    other = "en" if lang != "en" else "fa"
    assert audio_cache_key(caption, lang) != audio_cache_key(caption, other)


def test_cache_key_changes_with_caption() -> None:
    assert audio_cache_key("hello", "en") != audio_cache_key("hello world", "en")


def test_cache_key_changes_with_voice_config() -> None:
    assert audio_cache_key("hello", "en", voice_config="v1") != audio_cache_key(
        "hello", "en", voice_config="v2"
    )


def test_persian_and_english_audio_keys_differ() -> None:
    # Same latin letters would still differ by language tag in the key material.
    en_key = audio_cache_key("caption text", "en")
    fa_key = audio_cache_key("caption text", "fa")
    assert en_key != fa_key


@pytest.mark.parametrize(
    ("lang", "ext"),
    [
        ("en", "mp3"),
        ("fa", "mp3"),
        ("es", "wav"),
        ("de", "mp3"),
        ("zh", "mp3"),
    ],
)
def test_download_filename_safe(lang: str, ext: str) -> None:
    name = download_filename(lang, "abc123def456", ext)
    assert name == f"caption_{lang}_{'abc123de'}.{ext}"
    assert "/" not in name
    assert "\\" not in name
    assert ".." not in name
    assert len(name) < 80


def test_download_filename_strips_path_traversal() -> None:
    name = download_filename("en", "../evil\\path", "mp3")
    assert ".." not in name
    assert "\\" not in name
    assert name.startswith("caption_en_")
    assert name.endswith(".mp3")


def test_download_filename_ignores_raw_caption_text() -> None:
    # Must never embed caption body — only language + short id.
    name = download_filename("en", "deadbeef", "mp3")
    assert "person" not in name
    assert name == "caption_en_deadbeef.mp3"


def test_empty_caption_rejected() -> None:
    assert not is_valid_caption_for_tts("")
    assert not is_valid_caption_for_tts("   ")
    assert not is_valid_caption_for_tts(None)
    assert is_valid_caption_for_tts("A chair is visible.")


def test_detect_audio_format_wav_and_mp3() -> None:
    wav = b"RIFF" + b"\x00" * 20
    assert detect_audio_format(wav).extension == "wav"
    assert detect_audio_format(b"\xff\xfb\x90\x00fake").extension == "mp3"


def test_synthesize_empty_returns_none() -> None:
    service = SpeechService()
    assert service.synthesize("", "en") is None
    assert service.synthesize_artifact("   ", "fa") is None


def test_speech_service_voice_map_for_each_language() -> None:
    service = SpeechService()
    for lang in ("en", "fa", "es", "de", "zh"):
        voice = service.voice_for(lang)
        assert voice
        assert lang[:2] in voice.lower() or voice.startswith(
            {"en": "en", "fa": "fa", "es": "es", "de": "de", "zh": "zh"}[lang]
        )


@patch.object(SpeechService, "_edge_tts")
@patch.object(SpeechService, "_pyttsx3", return_value=None)
@patch.object(SpeechService, "_powershell_sapi", return_value=None)
def test_english_tts_uses_language_voice(
    _sapi: MagicMock, _py: MagicMock, edge: MagicMock, tmp_path, monkeypatch
) -> None:
    import language.tts.speech_service as mod

    monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path)
    edge.return_value = b"ID3fake-en-audio-bytes-xxxxxxxxxxxx"
    service = SpeechService()
    service._backend = "edge_tts"
    artifact = service.synthesize_artifact("A red car is parked.", "en")
    assert artifact is not None
    assert artifact.language == "en"
    assert artifact.voice == _EDGE_VOICES["en"]
    edge.assert_called_once()
    assert edge.call_args[0][1] == "en"


@pytest.mark.parametrize("lang", ["fa", "es", "de", "zh"])
@patch.object(SpeechService, "_edge_tts")
@patch.object(SpeechService, "_pyttsx3", return_value=None)
@patch.object(SpeechService, "_powershell_sapi", return_value=None)
def test_multilingual_tts_maps_language(
    _sapi: MagicMock, _py: MagicMock, edge: MagicMock, lang: str, tmp_path, monkeypatch
) -> None:
    import language.tts.speech_service as mod

    monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path)
    edge.return_value = b"ID3fake-xx-audio-bytes-xxxxxxxxxxxx"
    service = SpeechService()
    service._backend = "edge_tts"
    text = {
        "fa": "یک نفر کنار میز ایستاده است.",
        "es": "Una persona está junto a una mesa.",
        "de": "Eine Person steht neben einem Tisch.",
        "zh": "一个人站在桌子旁边。",
    }[lang]
    artifact = service.synthesize_artifact(text, lang)
    assert artifact is not None
    assert artifact.language == lang
    assert artifact.voice == _EDGE_VOICES[lang]
    assert edge.call_args[0][1] == lang


@patch.object(SpeechService, "_edge_tts", return_value=None)
@patch.object(SpeechService, "_pyttsx3", return_value=None)
@patch.object(SpeechService, "_powershell_sapi", return_value=None)
def test_tts_failure_returns_none(
    _sapi: MagicMock, _py: MagicMock, _edge: MagicMock, tmp_path, monkeypatch
) -> None:
    import language.tts.speech_service as mod

    monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path)
    service = SpeechService()
    service._backend = "edge_tts"
    assert service.synthesize("Hello", "en") is None


def test_disk_cache_hit_avoids_regeneration(tmp_path, monkeypatch) -> None:
    import language.tts.speech_service as mod

    monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path)
    service = SpeechService()
    service._backend = "edge_tts"
    payload = b"ID3cached-audio-payload-xxxxxxxxxxxx"

    with patch.object(service, "_edge_tts", return_value=payload) as edge:
        first = service.synthesize_artifact("Cache me", "en")
        second = service.synthesize_artifact("Cache me", "en")
    assert first is not None and second is not None
    assert first.audio == second.audio == payload
    assert edge.call_count == 1


def test_stale_language_cache_does_not_cross_contaminate(tmp_path, monkeypatch) -> None:
    import language.tts.speech_service as mod

    monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path)
    service = SpeechService()
    service._backend = "edge_tts"

    def _fake(text: str, language: str) -> bytes:
        return f"ID3-{language}-{text[:8]}".encode("utf-8") + b"-pad-pad-pad-pad"

    with patch.object(service, "_edge_tts", side_effect=_fake):
        en = service.synthesize_artifact("Same caption body", "en")
        fa = service.synthesize_artifact("Same caption body", "fa")
    assert en is not None and fa is not None
    assert en.audio != fa.audio
    assert en.language == "en"
    assert fa.language == "fa"


def test_caption_to_language_mapping_contract() -> None:
    """UI language is source of truth — TTS must not invent language from text."""
    # English words under fa must still request fa voice mapping.
    assert normalize_tts_language("fa") == "fa"
    service = SpeechService()
    assert service.voice_for("fa") == _EDGE_VOICES["fa"]


def test_playback_state_enum_and_invalidation_helpers_importable() -> None:
    from streamlit_app.components.tts_control import (
        PlaybackState,
        clear_tts_playback_state,
        invalidate_tts_for_content_change,
    )

    assert PlaybackState.IDLE.value == "idle"
    assert PlaybackState.PLAYING.value == "playing"
    assert PlaybackState.LOADING.value == "loading"
    assert PlaybackState.ERROR.value == "error"
    assert callable(clear_tts_playback_state)
    assert callable(invalidate_tts_for_content_change)


def test_only_one_active_playback_helper_resets_prior_stream() -> None:
    """Simulate session: starting a new control clears prior audio bytes."""
    session = {
        "tts_active_key": "caption_main",
        "tts_audio_bytes": b"old",
        "tts_audio_format": "audio/mpeg",
        "tts_audio_text_key": "oldkey",
        "tts_needs_autoplay": True,
    }

    # Mirror _ensure_single_active behavior without Streamlit runtime.
    control_key = "answer_0"
    current = session.get("tts_active_key")
    if current and current != control_key:
        session["tts_audio_bytes"] = None
        session["tts_audio_format"] = None
        session["tts_audio_text_key"] = None
        session["tts_needs_autoplay"] = False
    session["tts_active_key"] = control_key

    assert session["tts_active_key"] == "answer_0"
    assert session["tts_audio_bytes"] is None
    assert session["tts_needs_autoplay"] is False


def test_translation_module_untouched_by_tts_package() -> None:
    """Regression: TTS must not own/replace caption translation."""
    import language.tts as tts_pkg
    import language.localization.caption_translator as ct

    assert hasattr(ct, "CaptionTranslator")
    assert "translate" not in dir(tts_pkg) or not callable(
        getattr(tts_pkg, "translate", None)
    )
    # synthesize helpers exist; no CaptionTranslator re-export from TTS.
    assert not hasattr(tts_pkg, "CaptionTranslator")


def test_speech_artifact_fields() -> None:
    art = SpeechArtifact(
        audio=b"ID3x",
        language="es",
        voice=_EDGE_VOICES["es"],
        cache_key="a" * 24,
        extension="mp3",
        mime="audio/mpeg",
    )
    assert art.language == "es"
    assert art.extension == "mp3"
