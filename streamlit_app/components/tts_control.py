"""Competition-ready caption TTS controls — speaks already-displayed text.

States: IDLE | LOADING | PLAYING | STOPPED | ERROR

Does NOT translate. Language comes from the UI (``ui_language``).
"""

from __future__ import annotations

import base64
import html
from enum import Enum

import streamlit as st
import streamlit.components.v1 as components

from language.tts.audio_utils import (
    audio_cache_key,
    detect_audio_format,
    download_filename,
    is_valid_caption_for_tts,
    normalize_tts_language,
)
from streamlit_app.i18n import t


class PlaybackState(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    PLAYING = "playing"
    STOPPED = "stopped"
    ERROR = "error"


def clear_tts_playback_state() -> None:
    """Invalidate active playback (language change, new image, new caption)."""
    st.session_state.tts_active_key = None
    st.session_state.tts_audio_bytes = None
    st.session_state.tts_audio_format = None
    st.session_state.tts_audio_text_key = None
    st.session_state.tts_audio_language = None
    st.session_state.tts_needs_autoplay = False
    st.session_state.tts_playback_state = PlaybackState.IDLE.value
    st.session_state.tts_error_message = None
    # Drop per-control loading flags.
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("tts_loading_"):
            st.session_state[key] = False


def invalidate_tts_for_content_change() -> None:
    """Call when image, caption, or UI language changes."""
    clear_tts_playback_state()


def render_tts_button(text: str, *, language: str, key: str) -> None:
    """Legacy entry — prefer ``render_caption_tts`` for the caption card."""
    render_inline_tts(text, language=language, key=key)


def render_caption_tts(text: str, *, language: str, key: str = "caption_main") -> None:
    """Listen / Stop + Download controls for the caption header row."""
    render_inline_tts(
        text,
        language=language,
        key=key,
        icon_only=False,
        allow_download=True,
    )


def _session_cache_key(language: str, text: str) -> str:
    return audio_cache_key(text, language)


def _stop_playback() -> None:
    st.session_state.tts_active_key = None
    st.session_state.tts_audio_bytes = None
    st.session_state.tts_audio_format = None
    st.session_state.tts_audio_text_key = None
    st.session_state.tts_audio_language = None
    st.session_state.tts_needs_autoplay = False
    st.session_state.tts_playback_state = PlaybackState.STOPPED.value
    st.session_state.tts_error_message = None


def _ensure_single_active(control_key: str) -> None:
    """Only one TTS control may be active."""
    current = st.session_state.get("tts_active_key")
    if current and current != control_key:
        # Switching streams — clear previous without marking ERROR.
        st.session_state.tts_audio_bytes = None
        st.session_state.tts_audio_format = None
        st.session_state.tts_audio_text_key = None
        st.session_state.tts_needs_autoplay = False


def render_inline_tts(
    text: str,
    *,
    language: str,
    key: str,
    icon_only: bool = False,
    allow_download: bool = False,
) -> None:
    """Listen/Stop toggle with optional download. Hides native audio chrome."""
    body = (text or "").strip()
    if not is_valid_caption_for_tts(body):
        return

    lang = normalize_tts_language(language)
    text_key = _session_cache_key(lang, body)
    cache_bucket = st.session_state.setdefault("tts_audio_cache", {})
    cached = cache_bucket.get(text_key)

    # Invalidate if this control was active for a different caption/language.
    active_match = (
        st.session_state.get("tts_active_key") == key
        and st.session_state.get("tts_audio_text_key") == text_key
        and st.session_state.get("tts_audio_language") in {None, lang}
        and bool(st.session_state.get("tts_audio_bytes"))
    )
    if st.session_state.get("tts_active_key") == key and not active_match:
        clear_tts_playback_state()

    loading_key = f"tts_loading_{key}"
    is_loading = bool(st.session_state.get(loading_key))
    is_playing = active_match
    state = PlaybackState.PLAYING if is_playing else PlaybackState.IDLE
    if is_loading:
        state = PlaybackState.LOADING
    if st.session_state.get("tts_playback_state") == PlaybackState.ERROR.value and not is_playing:
        state = PlaybackState.ERROR

    cols = st.columns([1, 1] if allow_download else [1], gap="medium")

    with cols[0]:
        if state == PlaybackState.LOADING:
            label = f"⏳ {t('streamlit.tts.loading')}"
            disabled = True
            help_text = t("streamlit.tts.help")
        elif is_playing:
            label = f"⏹ {t('streamlit.tts.stop')}" if not icon_only else "⏹"
            disabled = False
            help_text = t("streamlit.tts.stop")
        else:
            play_label = t("streamlit.tts.play")
            label = f"🔊 {play_label}" if not icon_only else "🔊"
            disabled = False
            help_text = t("streamlit.tts.help")

        clicked = st.button(
            label,
            key=f"tts_toggle_{key}",
            help=help_text,
            type="secondary",
            disabled=disabled,
            use_container_width=True,
        )

    if clicked and is_playing:
        _stop_playback()
        st.session_state[loading_key] = False
        st.rerun()

    if clicked and not is_playing and not is_loading:
        _ensure_single_active(key)
        st.session_state.tts_active_key = key
        st.session_state.tts_audio_bytes = None
        st.session_state.tts_audio_format = None
        st.session_state.tts_audio_text_key = text_key
        st.session_state.tts_audio_language = lang
        st.session_state.tts_needs_autoplay = False
        st.session_state[loading_key] = True
        st.session_state.tts_playback_state = PlaybackState.LOADING.value
        st.session_state.tts_error_message = None

        audio = None
        fmt = None
        if cached and cached.get("bytes") and cached.get("language") == lang:
            audio = cached["bytes"]
            fmt = cached.get("format")
        else:
            with st.spinner(t("streamlit.tts.loading")):
                try:
                    from language.tts import synthesize_display_artifact

                    artifact = synthesize_display_artifact(body, lang)
                    if artifact is not None:
                        audio = artifact.audio
                        fmt = artifact.mime
                except Exception as exc:  # noqa: BLE001
                    from core.logging import get_logger

                    get_logger(__name__).warning("TTS UI synthesis failed: %s", exc)
                    audio = None

        st.session_state[loading_key] = False
        if not audio:
            clear_tts_playback_state()
            st.session_state.tts_playback_state = PlaybackState.ERROR.value
            st.session_state.tts_error_message = t("streamlit.tts.error")
            st.warning(t("streamlit.tts.error"))
            return

        if not fmt:
            fmt = detect_audio_format(audio).mime
        cache_bucket[text_key] = {
            "bytes": audio,
            "format": fmt,
            "language": lang,
        }
        st.session_state.tts_audio_cache = cache_bucket
        st.session_state.tts_audio_bytes = audio
        st.session_state.tts_audio_format = fmt
        st.session_state.tts_audio_text_key = text_key
        st.session_state.tts_audio_language = lang
        st.session_state.tts_active_key = key
        st.session_state.tts_needs_autoplay = True
        st.session_state.tts_playback_state = PlaybackState.PLAYING.value
        st.rerun()

    # Resolve downloadable audio for this exact caption+language only.
    ready = None
    if cached and cached.get("bytes") and cached.get("language") == lang:
        ready = cached
    elif (
        active_match
        and st.session_state.get("tts_audio_bytes")
        and st.session_state.get("tts_audio_language") == lang
    ):
        ready = {
            "bytes": st.session_state.tts_audio_bytes,
            "format": st.session_state.get("tts_audio_format") or "audio/mpeg",
            "language": lang,
        }
        cache_bucket[text_key] = {**ready, "language": lang}
        st.session_state.tts_audio_cache = cache_bucket

    if allow_download and len(cols) > 1:
        with cols[1]:
            dl_label = f"⬇ {t('streamlit.tts.download')}"
            if ready and ready.get("bytes"):
                audio_bytes = ready["bytes"]
                info = detect_audio_format(audio_bytes, ready.get("format"))
                filename = download_filename(lang, text_key, info.extension)
                st.download_button(
                    label=dl_label,
                    data=audio_bytes,
                    file_name=filename,
                    mime=info.mime,
                    key=f"tts_download_{key}_{text_key[:8]}",
                    help=t("streamlit.tts.download_help"),
                    use_container_width=True,
                )
            else:
                # Prepare download without requiring Listen first.
                prep = st.button(
                    dl_label,
                    key=f"tts_prep_download_{key}",
                    help=t("streamlit.tts.download_help"),
                    type="secondary",
                    use_container_width=True,
                )
                if prep:
                    st.session_state[loading_key] = True
                    with st.spinner(t("streamlit.tts.loading")):
                        try:
                            from language.tts import synthesize_display_artifact

                            artifact = synthesize_display_artifact(body, lang)
                        except Exception as exc:  # noqa: BLE001
                            from core.logging import get_logger

                            get_logger(__name__).warning("TTS download prep failed: %s", exc)
                            artifact = None
                    st.session_state[loading_key] = False
                    if artifact is None:
                        st.session_state.tts_playback_state = PlaybackState.ERROR.value
                        st.warning(t("streamlit.tts.error"))
                    else:
                        cache_bucket[text_key] = {
                            "bytes": artifact.audio,
                            "format": artifact.mime,
                            "language": lang,
                        }
                        st.session_state.tts_audio_cache = cache_bucket
                        st.rerun()

    if st.session_state.get("tts_playback_state") == PlaybackState.ERROR.value:
        err = st.session_state.get("tts_error_message") or t("streamlit.tts.error")
        if not is_playing:
            st.caption(err)

    # Inject autoplay only once per Listen — avoid restart loops on Streamlit reruns.
    if (
        st.session_state.get("tts_active_key") == key
        and st.session_state.get("tts_audio_bytes")
        and st.session_state.get("tts_audio_text_key") == text_key
        and st.session_state.get("tts_audio_language") == lang
        and st.session_state.get("tts_needs_autoplay")
    ):
        audio = st.session_state.tts_audio_bytes
        fmt = st.session_state.get("tts_audio_format") or "audio/mpeg"
        info = detect_audio_format(audio, fmt)
        b64 = base64.b64encode(audio).decode("ascii")
        safe_key = html.escape(key)
        components.html(
            f"""
            <audio id="sentivis-tts-{safe_key}" autoplay style="display:none">
              <source src="data:{info.mime};base64,{b64}" type="{info.mime}">
            </audio>
            <script>
              const el = document.getElementById("sentivis-tts-{safe_key}");
              if (el) {{ el.play().catch(() => {{}}); }}
            </script>
            """,
            height=0,
        )
        st.session_state.tts_needs_autoplay = False
