"""Streamlit Vision Assistant panel — ask anything about the current image."""

from __future__ import annotations

import html

import streamlit as st

from core.contracts.pipeline import PipelineResult
from language.assistant import (
    VisionAssistant,
    VisionAssistantSession,
    build_evidence_packet,
    generate_suggested_questions,
)
from streamlit_app.i18n import t


def _image_key(result: PipelineResult) -> str:
    path = str(result.request.image_path)
    caption = result.caption.canonical_caption_en or result.caption.text
    return f"{path}|{hash(caption) & 0xFFFFFFFF}"


def _ensure_session(result: PipelineResult) -> VisionAssistantSession:
    key = _image_key(result)
    existing = st.session_state.get("vision_assistant_session")
    if isinstance(existing, VisionAssistantSession) and existing.image_key == key:
        return existing
    packet = build_evidence_packet(
        result.scene_context,
        canonical_caption_en=result.caption.canonical_caption_en or result.caption.text,
        evidence_brief=result.evidence_brief,
        ocr_snippets=result.ocr_snippets,
        verified_evidence=result.verified_evidence,
    )
    session = VisionAssistantSession(image_key=key, evidence=packet)
    st.session_state.vision_assistant_session = session
    st.session_state.vision_assistant_suggestions = None
    st.session_state.vision_assistant_suggestions_lang = None
    return session


def _answered_user_questions(session: VisionAssistantSession) -> list[str]:
    return [turn.text for turn in session.turns if turn.role == "user"]


def _suggestions(result: PipelineResult, language: str, session: VisionAssistantSession) -> list[str]:
    """Build or refresh image-specific suggested questions (up to 5)."""
    answered = _answered_user_questions(session)
    cache_key = (
        language,
        tuple(answered[-6:]),
    )
    cached = st.session_state.get("vision_assistant_suggestions")
    cached_meta = st.session_state.get("vision_assistant_suggestions_meta")
    if cached and cached_meta == cache_key:
        return list(cached)
    questions = generate_suggested_questions(
        session.evidence,
        language=language,
        limit=5,
        answered_questions=answered,
    )
    st.session_state.vision_assistant_suggestions = questions
    st.session_state.vision_assistant_suggestions_meta = cache_key
    st.session_state.vision_assistant_suggestions_lang = language
    return questions


def render_vision_assistant(result: PipelineResult | None, *, language: str = "en") -> None:
    if result is None:
        return

    session = _ensure_session(result)
    st.markdown(
        f'<div class="glass-card vision-assistant-panel" style="padding-bottom:0.35rem;">'
        f'<p class="metric-pill" style="margin-bottom:0.35rem;">{t("streamlit.assistant.title")}</p>'
        f'<p style="color:var(--text-secondary);margin:0 0 0.75rem;font-size:0.94rem;line-height:1.5;">'
        f'{t("streamlit.assistant.subtitle")}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    pending = st.session_state.pop("vision_assistant_pending", None)
    with st.form("vision_assistant_form", clear_on_submit=True):
        in_col, btn_col = st.columns([5, 1])
        with in_col:
            typed = st.text_input(
                t("streamlit.assistant.input_label"),
                placeholder=t("streamlit.assistant.input_placeholder"),
                key="vision_assistant_input",
                label_visibility="collapsed",
            )
        with btn_col:
            st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
            asked = st.form_submit_button(t("streamlit.assistant.ask"), type="primary", use_container_width=True)

    suggestions = _suggestions(result, language, session)
    if suggestions:
        st.caption(t("streamlit.assistant.suggested"))
        st.markdown('<div class="va-suggestions">', unsafe_allow_html=True)
        for index, question in enumerate(suggestions):
            if st.button(
                question,
                key=f"va_suggest_{index}_{language}",
                use_container_width=True,
            ):
                st.session_state.vision_assistant_pending = question
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    question = (pending or typed or "").strip()
    if asked or pending:
        if question:
            assistant = st.session_state.get("vision_assistant_engine")
            if not isinstance(assistant, VisionAssistant):
                assistant = VisionAssistant()
                st.session_state.vision_assistant_engine = assistant
            with st.spinner(t("streamlit.assistant.thinking")):
                assistant.answer(session, question, language=language)
            st.session_state.vision_assistant_session = session
            # Refresh suggestions after each turn so they explore unanswered angles.
            st.session_state.vision_assistant_suggestions = None
            st.session_state.vision_assistant_suggestions_meta = None

    if session.turns:
        st.markdown(f"**{t('streamlit.assistant.conversation')}**")
        for index, turn in enumerate(session.turns[-12:]):
            safe_text = html.escape(turn.text)
            if turn.role == "user":
                st.markdown(
                    f'<div class="va-turn va-user">'
                    f'<span class="va-role">{t("streamlit.assistant.you")}</span><br/>'
                    f'<span class="va-text">{safe_text}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="va-turn va-assistant">'
                    f'<span class="va-role">{t("streamlit.assistant.sentivis")}</span><br/>'
                    f'<span class="va-text">{safe_text}</span></div>',
                    unsafe_allow_html=True,
                )
                from streamlit_app.components.tts_control import render_inline_tts

                render_inline_tts(turn.text, language=language, key=f"answer_{index}", icon_only=True)
        show_diag = True
        try:
            from streamlit_app import preferences as prefs

            show_diag = not bool(prefs.load_competition_mode())
        except Exception:  # noqa: BLE001
            show_diag = True
        if show_diag:
            with st.expander(t("streamlit.assistant.diagnostics_title"), expanded=False):
                st.caption(
                    t("streamlit.assistant.diagnostics").format(
                        llm=session.assistant_llm_calls,
                        vlm=session.assistant_vlm_calls,
                        initial=result.initial_vlm_calls or result.metrics.vlm_executions,
                    )
                )


def clear_vision_assistant_state() -> None:
    for key in (
        "vision_assistant_session",
        "vision_assistant_suggestions",
        "vision_assistant_suggestions_lang",
        "vision_assistant_suggestions_meta",
        "vision_assistant_pending",
        "vision_assistant_input",
    ):
        if key in st.session_state:
            del st.session_state[key]
