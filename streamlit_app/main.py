"""Sentivis AI — Premium Streamlit Application."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from streamlit_app.runtime import configure_runtime

configure_runtime()

import streamlit as st

from core.contracts.pipeline import StageProgress
from core.utils.paths import uploads_dir
from language.localization.caption_translator import CaptionTranslator
from streamlit_app import preferences as prefs
from streamlit_app.backend import StreamlitBackend
from streamlit_app.branding import favicon_path, logo_path
from streamlit_app.components.exports import render_exports
from streamlit_app.components.results import render_results
from streamlit_app.components.viewer import load_display_image, render_minimap, render_overlays
from streamlit_app.diagnostics import build_readiness_report
from streamlit_app.i18n import LANGUAGE_LABELS, SUPPORTED_LANGUAGES, set_language, sync_language_from_prefs, t
from ui.media import fetch_random_public_image, is_online
from streamlit_app.components.vision_assistant import clear_vision_assistant_state
from streamlit_app.startup import initialize_backend
from streamlit_app.theme import inject_theme

try:
    from streamlit_image_comparison import image_comparison
except ImportError:
    image_comparison = None

try:
    from streamlit_option_menu import option_menu
except ImportError:
    option_menu = None


def _init_session() -> None:
    if "ui_language" not in st.session_state:
        # Canonical language source of truth across Streamlit reruns.
        st.session_state.ui_language = prefs.load_language()
    if "readiness" not in st.session_state:
        st.session_state.readiness = build_readiness_report()
    if "backend" not in st.session_state:
        readiness = st.session_state.readiness
        if readiness.ready:
            with st.spinner(t("streamlit.startup.loading")):
                st.session_state.backend = initialize_backend()
        else:
            st.session_state.backend = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "image_path" not in st.session_state:
        st.session_state.image_path = None
    if "original_image" not in st.session_state:
        st.session_state.original_image = None
    if "enhanced_image" not in st.session_state:
        st.session_state.enhanced_image = None
    if "display_image" not in st.session_state:
        st.session_state.display_image = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "nav" not in st.session_state:
        st.session_state.nav = "Analyze"
    if "auto_analyze" not in st.session_state:
        st.session_state.auto_analyze = False
    if "viewer_mode" not in st.session_state:
        st.session_state.viewer_mode = "fit"
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = t("streamlit.status.idle")
    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False
    if "caption_translator" not in st.session_state:
        st.session_state.caption_translator = CaptionTranslator()
    if "last_translation_count" not in st.session_state:
        st.session_state.last_translation_count = 0
    # Keep catalog + prefs aligned with session_state (never recreate randomly).
    set_language(st.session_state.ui_language)


def _current_language() -> str:
    lang = st.session_state.get("ui_language") or prefs.load_language() or "en"
    if lang not in SUPPORTED_LANGUAGES:
        return "en"
    return str(lang)


def _switch_language(target: str) -> None:
    """Persist UI language and translate caption only — never rerun vision."""
    code = (target or "en").lower().strip()
    if code not in SUPPORTED_LANGUAGES:
        code = "en"
    if code == _current_language() and prefs.load_language() == code:
        return
    st.session_state.ui_language = code
    set_language(code)
    _apply_caption_language(code)
    # Stale TTS must not keep playing after language (and caption text) change.
    from streamlit_app.components.tts_control import invalidate_tts_for_content_change

    invalidate_tts_for_content_change()
    # Regenerate suggested questions in the new language (text only).
    st.session_state.vision_assistant_suggestions = None
    st.session_state.vision_assistant_suggestions_lang = None
    st.rerun()


def _canonical_caption(result: object) -> str:
    caption = getattr(result, "caption", None)
    if caption is None:
        return ""
    return (
        getattr(caption, "canonical_caption_en", "")
        or getattr(caption, "narrative_full", "")
        or getattr(caption, "text", "")
        or ""
    ).strip()


def _apply_caption_language(lang: str) -> None:
    """Swap displayed caption language without re-running vision analysis."""
    result = st.session_state.get("result")
    if result is None:
        return
    canonical = _canonical_caption(result)
    if not canonical:
        return
    translations = dict(getattr(result, "caption_translations", ()) or ())
    translations.setdefault("en", canonical)
    target = (lang or "en").lower().strip()
    if target not in SUPPORTED_LANGUAGES:
        target = "en"

    if target in translations and translations[target].strip():
        display = translations[target].strip()
    elif target == "en":
        display = canonical
        translations["en"] = canonical
    else:
        translator: CaptionTranslator = st.session_state.caption_translator
        before = translator.translation_count
        with st.spinner(t("streamlit.status.translating")):
            display = translator.translate(canonical, target)
        st.session_state.last_translation_count = max(0, translator.translation_count - before)
        translations[target] = display

    from language.semantic.narrative_generator import (
        executive_summary_from_paragraph,
        short_caption_from_paragraph,
    )

    short = short_caption_from_paragraph(display)
    executive = executive_summary_from_paragraph(display)
    new_caption = replace(
        result.caption,
        text=display,
        narrative_full=display,
        narrative_short=short,
        executive_summary=executive,
        canonical_caption_en=canonical,
    )
    st.session_state.result = replace(
        result,
        caption=new_caption,
        caption_translations=tuple(sorted(translations.items())),
    )


def _configure_page() -> None:
    icon = str(favicon_path()) if favicon_path().is_file() else "🔮"
    st.set_page_config(
        page_title=t("app.name"),
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme(high_contrast=prefs.load_high_contrast())


def _render_readiness_panel() -> None:
    """Competition UI: hide readiness/diagnostics unless startup failed."""
    readiness = st.session_state.readiness
    if readiness.offline_mode:
        st.sidebar.warning(t("streamlit.offline.banner"))
    # Ready systems stay quiet — judges should not see health/diagnostics noise.
    if readiness.ready:
        return
    st.sidebar.error(readiness.title)
    st.sidebar.caption(readiness.summary)
    with st.sidebar.expander("Startup diagnostics", expanded=True):
        st.markdown(f"**Assets:** {'OK' if readiness.assets_ok else 'Missing'}")
        st.markdown(f"**Configuration:** {'OK' if readiness.config_ok else 'Missing'}")
        st.markdown(f"**Permissions:** {'OK' if readiness.permissions_ok else 'Blocked'}")
        st.markdown(f"**DEVICE:** {'CUDA' if readiness.gpu_available else 'CPU'}")
        st.markdown(f"**CUDA available:** {'Yes' if readiness.gpu_available else 'No'}")
        st.markdown(f"**CPU:** {readiness.cpu_name}")
        st.markdown(f"**RAM available:** {readiness.ram_available_gb:.1f} GB")
        st.markdown(f"**Disk free:** {readiness.disk_free_gb:.1f} GB")
        for item in readiness.dependencies:
            status = "✓" if item.available else ("!" if not item.required else "✗")
            st.markdown(f"{status} **{item.name}** — {item.detail}")
        for note in readiness.notes:
            st.caption(note)


def _render_sidebar(backend: StreamlitBackend | None) -> str:
    logo = logo_path()
    if logo.is_file():
        st.sidebar.image(str(logo), width=120)
    st.sidebar.markdown(f'<p class="brand-mark gradient-header">{t("app.name")}</p>', unsafe_allow_html=True)
    st.sidebar.caption(t("app.slogan"))
    _render_readiness_panel()

    # One competition workspace only — keep Settings/Dashboard code, hide from nav.
    nav_labels = [t("streamlit.nav.analyze")]
    icons = ["image"]
    if option_menu is not None:
        nav = option_menu(
            menu_title=None,
            options=nav_labels,
            icons=icons,
            default_index=0,
            styles={
                "container": {"padding": "4px 0", "background-color": "transparent"},
                "icon": {"color": "#FF3FA4", "font-size": "17px"},
                "nav-link": {
                    "font-size": "14px",
                    "color": "#FFFFFF",
                    "border-radius": "12px",
                    "padding": "10px 14px",
                    "margin": "2px 0",
                },
                "nav-link-selected": {
                    "background-color": "rgba(255, 63, 164, 0.22)",
                    "border": "1px solid rgba(255, 63, 164, 0.45)",
                },
            },
        )
    else:
        nav = st.sidebar.radio("Navigation", nav_labels, label_visibility="collapsed")
    nav_label: str = str(nav)

    if not prefs.load_competition_mode():
        st.sidebar.markdown("---")
        competition = st.sidebar.toggle(
            t("settings.competition_enable"), value=prefs.load_competition_mode()
        )
        if competition != prefs.load_competition_mode():
            prefs.save_competition_mode(competition)

    # Health / model diagnostics stay out of the primary competition sidebar.

    if st.session_state.history:
        st.sidebar.markdown(f"### {t('sidebar.recent')}")
        for entry in st.session_state.history[:6]:
            st.sidebar.caption(f"• {entry}")

    return nav_label


def _render_main_language_selector() -> None:
    """Compact language control on the main competition page (no Settings page)."""
    current = _current_language()
    labels = [LANGUAGE_LABELS[code] for code in SUPPORTED_LANGUAGES]
    code_by_label = {LANGUAGE_LABELS[code]: code for code in SUPPORTED_LANGUAGES}
    st.markdown(
        f'<div class="language-bar"><span class="language-bar-label">🌐 {t("settings.language")}</span></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([1.35, 2.4, 2.25])
    with cols[0]:
        st.caption(t("streamlit.language.caption_only_hint"))
    with cols[1]:
        choice = st.selectbox(
            t("settings.language"),
            options=labels,
            index=labels.index(LANGUAGE_LABELS.get(current, "English")),
            key="main_language_select",
            label_visibility="collapsed",
        )
    selected = code_by_label.get(choice, "en")
    if selected != current:
        _switch_language(selected)


def _render_settings() -> None:
    st.markdown(f"## {t('settings.title')}")
    tabs = st.tabs(
        [
            t("settings.general"),
            t("settings.appearance"),
            t("settings.performance"),
            t("settings.competition"),
            t("settings.accessibility"),
        ]
    )
    with tabs[0]:
        st.markdown(f'<div class="glass-card"><p>{t("settings.general.desc")}</p></div>', unsafe_allow_html=True)
    with tabs[1]:
        st.markdown(f"**{t('settings.language')}**")
        settings_lang = st.selectbox(
            t("settings.language"),
            options=list(SUPPORTED_LANGUAGES),
            format_func=lambda c: LANGUAGE_LABELS.get(c, c),
            index=max(0, list(SUPPORTED_LANGUAGES).index(_current_language())),
            key="settings_lang_select",
        )
        if settings_lang != _current_language():
            _switch_language(settings_lang)
    with tabs[2]:
        enh = st.checkbox(t("settings.enhancement.enable"), value=prefs.load_enable_enhancement())
        sr = st.checkbox(t("settings.enhancement.super_resolution"), value=prefs.load_enable_super_resolution())
        sam2 = st.checkbox(t("settings.enhancement.sam2"), value=prefs.load_enable_sam2())
        cmp = st.checkbox(t("settings.enhancement.comparison"), value=prefs.load_comparison_mode())
        if st.button(t("button.restore_defaults")):
            prefs.save_enable_enhancement(True)
            prefs.save_enable_super_resolution(False)
            prefs.save_enable_sam2(True)
            prefs.save_comparison_mode(False)
            st.rerun()
        else:
            prefs.save_enable_enhancement(enh)
            prefs.save_enable_super_resolution(sr)
            prefs.save_enable_sam2(sam2)
            prefs.save_comparison_mode(cmp)
    with tabs[3]:
        st.toggle(t("settings.competition_enable"), value=prefs.load_competition_mode(), key="comp_tab")
    with tabs[4]:
        hc = st.checkbox(t("settings.accessibility.high_contrast"), value=prefs.load_high_contrast())
        prefs.save_high_contrast(hc)


def _set_image(path: Path, *, auto_analyze: bool = False) -> None:
    st.session_state.image_path = path
    st.session_state.result = None
    st.session_state.original_image = str(path)
    st.session_state.enhanced_image = None
    st.session_state.display_image = str(path)
    st.session_state.auto_analyze = auto_analyze
    st.session_state.processing_status = t("streamlit.status.ready_image")
    st.session_state.viewer_mode = "fit"
    clear_vision_assistant_state()
    from streamlit_app.components.tts_control import invalidate_tts_for_content_change

    invalidate_tts_for_content_change()


def _clear_session() -> None:
    st.session_state.image_path = None
    st.session_state.result = None
    st.session_state.original_image = None
    st.session_state.enhanced_image = None
    st.session_state.display_image = None
    st.session_state.auto_analyze = False
    st.session_state.processing_status = t("streamlit.status.idle")
    clear_vision_assistant_state()
    from streamlit_app.components.tts_control import invalidate_tts_for_content_change

    invalidate_tts_for_content_change()


def _render_hero() -> None:
    st.markdown(
        f'<div class="hero-header">'
        f'<p class="hero-title gradient-header">{t("app.name")}</p>'
        f'<p class="hero-subtitle">{t("app.slogan")}</p>'
        f'<p style="color:#D7CFE8;font-size:0.95rem;margin:0.35rem 0 0;max-width:42rem;">'
        f'{t("app.workflow_hint")}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_toolbar() -> None:
    online = is_online() and not getattr(st.session_state.readiness, "offline_mode", False)
    st.markdown('<div class="toolbar-shell">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption(t("streamlit.toolbar.upload"))
        upload = st.file_uploader(
            t("streamlit.toolbar.upload"),
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            label_visibility="collapsed",
            key="toolbar_uploader",
        )
        if upload is not None:
            upload_dir = uploads_dir()
            tmp = upload_dir / upload.name
            tmp.write_bytes(upload.getvalue())
            current = Path(st.session_state.image_path) if st.session_state.image_path else None
            if current != tmp:
                _set_image(tmp, auto_analyze=False)
    with c2:
        st.caption("\u00a0")
        if st.button(
            t("streamlit.toolbar.random"),
            use_container_width=True,
            disabled=not online,
            help=None if online else t("streamlit.toolbar.random_offline"),
        ):
            try:
                with st.spinner(t("streamlit.toolbar.random_loading")):
                    path = fetch_random_public_image()
                _set_image(path, auto_analyze=True)
                st.rerun()
            except Exception as exc:
                st.error(t("streamlit.toolbar.random_failed"))
                with st.expander(t("streamlit.results.technical"), expanded=False):
                    st.caption(str(exc))
    with c3:
        st.caption("\u00a0")
        if st.button(
            t("streamlit.toolbar.clear"),
            use_container_width=True,
            help=t("tooltip.clear"),
        ):
            _clear_session()
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _viewer_meta_html(image_path: Path, result) -> str:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        width, height = 0, 0
    quality_level = "—"
    enhancement = "—"
    status = st.session_state.processing_status
    if result is not None and result.image_quality is not None:
        iq = result.image_quality
        level = (iq.quality_level or "MEDIUM").upper()
        quality_level = {
            "HIGH": t("streamlit.viewer.quality_high"),
            "MEDIUM": t("streamlit.viewer.quality_medium"),
            "LOW": t("streamlit.viewer.quality_low"),
        }.get(level, level)
        status_key = getattr(iq, "enhancement_status", "") or ""
        if iq.enhancement_applied or status_key in {"ENHANCEMENT_APPLIED", "ENHANCEMENT_VERIFIED"}:
            enhancement = t("streamlit.viewer.enhancement_applied")
            final_image = t("streamlit.viewer.final_enhanced")
        elif status_key == "ENHANCEMENT_FAILED":
            enhancement = t("streamlit.viewer.enhancement_failed")
            final_image = t("streamlit.viewer.final_original")
        elif status_key == "ENHANCEMENT_ATTEMPTED_UNVERIFIED":
            enhancement = t("streamlit.viewer.enhancement_unverified")
            final_image = t("streamlit.viewer.final_original")
        elif status_key == "ENHANCEMENT_REJECTED" or getattr(iq, "enhancement_rejected", False):
            enhancement = t("streamlit.viewer.enhancement_rejected")
            final_image = t("streamlit.viewer.final_original")
        elif "disabled" in (getattr(iq, "verification_reason", "") or "").lower():
            enhancement = t("streamlit.viewer.enhancement_failed")
            final_image = t("streamlit.viewer.final_original")
        elif level == "HIGH" or status_key == "ENHANCEMENT_NOT_REQUIRED":
            enhancement = t("streamlit.viewer.enhancement_not_required")
            final_image = t("streamlit.viewer.final_original")
        elif getattr(iq, "enhancement_attempted", False):
            enhancement = t("streamlit.viewer.enhancement_failed")
            final_image = t("streamlit.viewer.final_original")
        else:
            enhancement = t("streamlit.viewer.enhancement_not_required")
            final_image = t("streamlit.viewer.final_original")
        status = t("status.complete")
    elif result is not None:
        status = t("status.complete")
        final_image = "—"
    else:
        final_image = "—"
    reason = ""
    # Keep raw verification reasons off the competition primary chips.
    show_reason = not prefs.load_competition_mode()
    if show_reason and result is not None and result.image_quality is not None:
        reason_text = (
            getattr(result.image_quality, "verification_reason", "")
            or getattr(result.image_quality, "rejection_reason", "")
            or ""
        )
        if reason_text:
            reason = (
                f'<span class="viewer-chip"><strong>{t("streamlit.viewer.enhancement_reason")}</strong> '
                f"{reason_text}</span>"
            )
    return (
        f'<div class="viewer-meta">'
        f'<span class="viewer-chip"><strong>{t("streamlit.viewer.resolution")}</strong> {width}×{height}</span>'
        f'<span class="viewer-chip"><strong>{t("streamlit.viewer.quality_level")}</strong> {quality_level}</span>'
        f'<span class="viewer-chip"><strong>{t("streamlit.viewer.enhancement")}</strong> {enhancement}</span>'
        f'<span class="viewer-chip"><strong>{t("streamlit.viewer.final_image")}</strong> {final_image}</span>'
        f'{reason}'
        f'<span class="viewer-chip"><strong>{t("streamlit.viewer.status")}</strong> {status}</span>'
        f"</div>"
    )


def _execute_analysis(backend: StreamlitBackend, image_path: Path) -> None:
    analyzing = t("status.analyzing", device="GPU/CPU")
    st.session_state.analysis_running = True
    st.session_state.processing_status = analyzing
    st.markdown(
        f'<div class="status-banner"><div class="loading-orbit"></div>'
        f'<p style="margin:0.4rem 0 0;">{analyzing}</p></div>',
        unsafe_allow_html=True,
    )
    progress = st.progress(0, text=analyzing)
    status = st.empty()

    def on_progress(event: StageProgress) -> None:
        pct = max(0, min(100, int(event.percent)))
        progress.progress(pct, text=f"{event.stage.display_name} · {event.message}")
        status.caption(event.device or "")
        st.session_state.processing_status = event.stage.display_name

    try:
        result = backend.analyze(
            Path(image_path),
            competition_mode=prefs.load_competition_mode(),
            enable_enhancement=prefs.load_enable_enhancement(),
            enable_super_resolution=prefs.load_enable_super_resolution(),
            enable_sam2=prefs.load_enable_sam2(),
            on_progress=on_progress,
        )
        st.session_state.result = result
        st.session_state.original_image = str(image_path)
        clear_vision_assistant_state()
        from streamlit_app.components.tts_control import invalidate_tts_for_content_change

        invalidate_tts_for_content_change()
        enhanced = getattr(result, "enhanced_preview_path", None)
        applied = bool(
            result.image_quality is not None and result.image_quality.enhancement_applied and enhanced
        )
        st.session_state.enhanced_image = str(enhanced) if applied else None
        st.session_state.display_image = (
            str(enhanced) if applied and Path(str(enhanced)).is_file() else str(image_path)
        )
        st.session_state.auto_analyze = False
        st.session_state.processing_status = t("status.complete")
        st.session_state.last_translation_count = 0
        # Align displayed caption with selected UI language (translation only — no VLM).
        _apply_caption_language(_current_language())
        display_result = st.session_state.result or result
        preview = (
            display_result.caption.narrative_full
            or display_result.caption.narrative_short
            or display_result.caption.text
        )[:60]
        st.session_state.history.insert(0, f"{Path(image_path).name} — {preview}")
        st.success(t("status.complete"))
        st.rerun()
    except Exception as exc:
        st.session_state.auto_analyze = False
        st.session_state.processing_status = t("status.failed")
        st.error(t("status.failed"))
        with st.expander(t("streamlit.results.technical"), expanded=False):
            st.caption(str(exc))
    finally:
        st.session_state.analysis_running = False


def _run_analysis(backend: StreamlitBackend) -> None:
    _render_hero()
    _render_main_language_selector()
    _render_toolbar()

    image_path = st.session_state.image_path
    analysis_running = bool(st.session_state.get("analysis_running", False))
    # Primary action sits above the fold — judges must see it immediately.
    st.markdown('<div class="analyze-ready action-bar">', unsafe_allow_html=True)
    btn_col1, btn_col2, _ = st.columns([1.35, 1.0, 2.65])
    with btn_col1:
        analyze = st.button(
            t("button.analyze"),
            type="primary",
            disabled=image_path is None or analysis_running,
            use_container_width=True,
        )
    with btn_col2:
        # Cancel is only meaningful while analysis is actively running.
        cancel = False
        if analysis_running:
            cancel = st.button(t("button.cancel"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if cancel:
        backend.cancel()
        st.session_state.auto_analyze = False
        st.session_state.analysis_running = False
        st.session_state.processing_status = t("status.cancelled")
        st.warning(t("status.cancelled"))

    should_run = (analyze or st.session_state.auto_analyze) and image_path and not analysis_running
    if should_run:
        _execute_analysis(backend, Path(image_path))
        return

    # Wider results column so captions wrap less; image stays primary center.
    col_left, col_center, col_right = st.columns([0.85, 2.0, 2.05], gap="large")

    with col_left:
        has_result = st.session_state.result is not None
        # Competition primary path: image/caption/QA. Overlay tools stay available
        # but demoted under Advanced Visualization.
        adv_viz_label = t("streamlit.viewer.advanced_viz")
        if adv_viz_label == "streamlit.viewer.advanced_viz" or not adv_viz_label.strip():
            adv_viz_label = "Advanced Visualization"
        with st.expander(adv_viz_label, expanded=False):
            show_boxes = st.toggle(t("overlay.detections"), value=False, key="adv_show_boxes")
            show_rels = st.toggle(t("overlay.relationships"), value=False, key="adv_show_rels")
            show_labels = st.toggle(t("overlay.labels"), value=False, key="adv_show_labels")
            comparison = st.toggle(
                t("overlay.comparison"),
                value=prefs.load_comparison_mode(),
                disabled=not has_result,
                key="adv_comparison",
            )
            opacity = st.slider(t("overlay.opacity"), 0, 100, 50, key="adv_opacity")
            mode = st.radio(
                t("streamlit.viewer.mode"),
                options=["fit", "actual", "zoom"],
                format_func=lambda m: {
                    "fit": t("streamlit.viewer.fit"),
                    "actual": t("streamlit.viewer.actual"),
                    "zoom": t("streamlit.viewer.zoom_mode"),
                }[m],
                horizontal=True,
                key="viewer_mode",
            )
            zoom = 100
            if mode == "zoom":
                zoom = st.slider(t("streamlit.viewer.zoom"), 50, 300, 100, 10, key="adv_zoom")
            st.caption(t("streamlit.viewer.pan_hint"))

    with col_center:
        if image_path and Path(image_path).is_file():
            original_path = Path(st.session_state.get("original_image") or image_path)
            enhanced_state = st.session_state.get("enhanced_image")
            display_state = st.session_state.get("display_image") or image_path
            result = st.session_state.result
            enhanced_path = None
            if result is not None and result.enhanced_preview_path is not None:
                enhanced_path = Path(result.enhanced_preview_path)
            elif enhanced_state:
                enhanced_path = Path(str(enhanced_state))
            has_enhanced = (
                result is not None
                and result.image_quality is not None
                and result.image_quality.enhancement_applied
                and enhanced_path is not None
                and enhanced_path.is_file()
            )
            # Source of truth: accepted enhancement becomes the primary display image.
            primary_path = (
                enhanced_path
                if has_enhanced
                else Path(str(display_state) if display_state else original_path)
            )
            if not primary_path.is_file():
                primary_path = original_path
            base_original = load_display_image(original_path)
            base_display = load_display_image(primary_path)
            st.markdown(_viewer_meta_html(original_path, result), unsafe_allow_html=True)
            if comparison and has_enhanced and image_comparison is not None:
                image_comparison(
                    img1=str(original_path),
                    img2=str(enhanced_path),
                    label1=t("streamlit.viewer.original"),
                    label2=t("streamlit.viewer.enhanced"),
                    width=700,
                )
            elif has_enhanced:
                # Enhanced is primary; original is secondary for honesty/compare.
                st.caption(t("streamlit.viewer.final_enhanced"))
                display = render_overlays(
                    base_display,
                    result,
                    show_boxes=show_boxes,
                    show_relationships=show_rels,
                    show_labels=show_labels,
                    overlay_opacity=opacity / 100.0,
                )
                st.image(display, use_container_width=True)
                with st.expander(t("streamlit.viewer.original"), expanded=False):
                    st.image(base_original, use_container_width=True)
            else:
                display = render_overlays(
                    base_display,
                    result,
                    show_boxes=show_boxes,
                    show_relationships=show_rels,
                    show_labels=show_labels,
                    overlay_opacity=opacity / 100.0,
                )
                w, h = display.size
                if mode == "actual":
                    scale = 1.0
                elif mode == "zoom":
                    scale = zoom / 100.0
                else:
                    scale = min(1.0, 900 / max(w, 1))
                display = display.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample=3)
                label = (
                    t("streamlit.viewer.final_original")
                    if result is not None
                    else t("streamlit.viewer.original")
                )
                st.caption(label)
                st.image(display, use_container_width=(mode == "fit"))
            mini = render_minimap(base_display)
            st.caption(t("streamlit.viewer.minimap"))
            st.image(mini, width=140)
        else:
            st.markdown(
                f'<div class="glass-card" style="min-height:280px;display:flex;align-items:center;justify-content:center;">'
                f'<p style="color:#D7CFE8;text-align:center;">{t("image.placeholder")}<br/>'
                f'<span style="color:#FF3FA4;font-weight:600;">{t("streamlit.toolbar.hint")}</span></p></div>',
                unsafe_allow_html=True,
            )

    with col_right:
        # Do not render a standalone empty "Analysis Results" header card.
        # Vision Caption / empty-state content comes from render_results itself.
        render_results(st.session_state.result)
        if st.session_state.result:
            render_exports(
                st.session_state.result,
                backend.export_manager,
                backend.app_config.paths.exports_dir,
            )


def main() -> None:
    # Language must be published before any t() / theme injection.
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = prefs.load_language()
    set_language(st.session_state.ui_language)
    _configure_page()
    _init_session()
    backend: StreamlitBackend | None = st.session_state.backend
    nav = _render_sidebar(backend)

    if backend is None:
        st.error(st.session_state.readiness.summary)
        st.info(t("streamlit.startup.resolve_hint"))
        return

    # Competition UI: single Analyze workspace only (no Settings/Dashboard routes).
    _ = nav
    _run_analysis(backend)


if __name__ == "__main__":
    main()
