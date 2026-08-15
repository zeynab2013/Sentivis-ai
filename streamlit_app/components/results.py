"""Results panel — caption-first competition presentation."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.contracts.pipeline import PipelineResult
from streamlit_app.i18n import t
from streamlit_app.components.vision_assistant import render_vision_assistant
from ui.formatters.result_formatters import (
    format_activities,
    format_caption_confidence,
    format_color_palette,
    format_detected_objects,
    format_environment,
    format_execution_metrics,
    format_image_quality,
    format_object_details,
    format_quality_report,
    format_relationships,
    format_scene_graph,
)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _competition_mode() -> bool:
    try:
        from streamlit_app import preferences as prefs

        return bool(prefs.load_competition_mode())
    except Exception:  # noqa: BLE001
        return True


def _confidence_bar(label: str, value: float | None) -> str:
    if value is None:
        return (
            f'<div style="margin:0.55rem 0;">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;gap:0.75rem;">'
            f'<span style="color:#D7CFE8;word-break:normal;overflow-wrap:normal;">{_escape(label)}</span>'
            f'<strong style="color:#B9B0CC;white-space:nowrap;">Unavailable</strong></div>'
            f'<div class="confidence-track"><div class="confidence-fill" style="width:0%;opacity:0.25;"></div></div>'
            f"</div>"
        )
    pct = max(0.0, min(1.0, float(value))) * 100.0
    return (
        f'<div style="margin:0.55rem 0;">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;gap:0.75rem;">'
        f'<span style="color:#D7CFE8;word-break:normal;overflow-wrap:normal;">{_escape(label)}</span>'
        f'<strong style="color:#FF3FA4;white-space:nowrap;">{pct:.0f}%</strong></div>'
        f'<div class="confidence-track"><div class="confidence-fill" style="width:{pct:.1f}%;"></div></div>'
        f"</div>"
    )


def _render_quality_indicators(result: PipelineResult) -> None:
    report = result.quality_report
    hall_safety = (
        None if report.hallucination_risk is None else max(0.0, 1.0 - report.hallucination_risk)
    )
    bars = "".join(
        [
            _confidence_bar(t("streamlit.metric.caption_quality"), report.overall_quality),
            _confidence_bar(t("streamlit.quality.evidence"), report.evidence_consistency),
            _confidence_bar(t("streamlit.quality.objects"), report.object_coverage),
            _confidence_bar(t("streamlit.quality.relationships"), report.relationship_coverage),
            _confidence_bar(t("streamlit.quality.activities"), report.activity_coverage),
            _confidence_bar(t("streamlit.quality.hallucination_safety"), hall_safety),
        ]
    )
    st.markdown(
        f'<div class="glass-card"><p class="metric-pill">{t("streamlit.section.caption_confidence")}</p>'
        f'<p style="color:#B9B0CC;font-size:0.78rem;margin:0 0 0.4rem;">{t("streamlit.quality.internal_note")}</p>'
        f"{bars}</div>",
        unsafe_allow_html=True,
    )


def _enhancement_user_label(iq) -> tuple[str, str]:
    """Return (enhancement_label, final_image_label) in user-facing language."""
    level = (iq.quality_level or "MEDIUM").upper()
    status_key = getattr(iq, "enhancement_status", "") or ""
    if iq.enhancement_applied or status_key in {"ENHANCEMENT_APPLIED", "ENHANCEMENT_VERIFIED"}:
        return t("streamlit.viewer.enhancement_applied"), t("streamlit.viewer.final_enhanced")
    if status_key == "ENHANCEMENT_FAILED":
        return t("streamlit.viewer.enhancement_failed"), t("streamlit.viewer.final_original")
    if status_key == "ENHANCEMENT_ATTEMPTED_UNVERIFIED":
        return t("streamlit.viewer.enhancement_unverified"), t("streamlit.viewer.final_original")
    if status_key == "ENHANCEMENT_REJECTED" or getattr(iq, "enhancement_rejected", False):
        return t("streamlit.viewer.enhancement_rejected"), t("streamlit.viewer.final_original")
    if getattr(iq, "enhancement_attempted", False) and not iq.enhancement_applied:
        return t("streamlit.viewer.enhancement_failed"), t("streamlit.viewer.final_original")
    reason = (getattr(iq, "verification_reason", "") or "").lower()
    if "disabled" in reason:
        return t("streamlit.viewer.enhancement_failed"), t("streamlit.viewer.final_original")
    if level == "HIGH" or status_key == "ENHANCEMENT_NOT_REQUIRED":
        return t("streamlit.viewer.enhancement_not_required"), t("streamlit.viewer.final_original")
    return t("streamlit.viewer.enhancement_not_required"), t("streamlit.viewer.final_original")


def render_results(result: PipelineResult | None) -> None:
    if result is None:
        st.markdown(
            f'<div class="glass-card">'
            f'<p class="metric-pill">{t("results.empty.title")}</p>'
            f'<p style="color:var(--text-secondary);margin:0.35rem 0;">{t("results.empty.body")}</p>'
            f'<p style="color:var(--primary);font-weight:600;margin:0;">{t("results.empty.action")}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    competition = _competition_mode()
    paragraph = (
        result.caption.narrative_full
        or result.caption.executive_summary
        or result.caption.text
    ).strip()
    safe = _escape(paragraph)
    language = st.session_state.get("ui_language") or "en"

    st.markdown(
        f'<div class="caption-panel" style="margin-bottom:0;border-bottom-left-radius:0;'
        f'border-bottom-right-radius:0;padding-bottom:0.45rem;">'
        f'<p class="caption-kicker" style="margin:0;">{t("streamlit.results.caption_kicker")}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
    from streamlit_app.components.tts_control import render_caption_tts

    # Full-width voice row under the kicker — avoids cramped header overlap.
    st.markdown('<div class="caption-tts-bar">', unsafe_allow_html=True)
    render_caption_tts(paragraph, language=language, key="caption_main")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="caption-panel" style="margin-top:0;border-top-left-radius:0;'
        f'border-top-right-radius:0;padding-top:0.45rem;">'
        f'<p class="caption-body">{safe}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    render_vision_assistant(result, language=language)

    # Compact enhancement status for judges (no metric dumps on primary path).
    if result.image_quality is not None:
        iq = result.image_quality
        level = (iq.quality_level or "MEDIUM").upper()
        level_label = {
            "HIGH": t("streamlit.viewer.quality_high"),
            "MEDIUM": t("streamlit.viewer.quality_medium"),
            "LOW": t("streamlit.viewer.quality_low"),
        }.get(level, level)
        enh_label, final_label = _enhancement_user_label(iq)
        st.markdown(
            f'<div class="glass-card"><p class="metric-pill">{t("section.image_quality")}</p>'
            f'<div class="viewer-meta" style="margin:0.55rem 0 0;">'
            f'<span class="viewer-chip"><strong>{t("streamlit.viewer.quality_level")}</strong> {level_label}</span>'
            f'<span class="viewer-chip"><strong>{t("streamlit.viewer.enhancement")}</strong> {enh_label}</span>'
            f'<span class="viewer-chip"><strong>{t("streamlit.viewer.final_image")}</strong> {final_label}</span>'
            f"</div></div>",
            unsafe_allow_html=True,
        )
        if not competition:
            reason = (
                getattr(iq, "verification_reason", "")
                or getattr(iq, "rejection_reason", "")
                or ""
            )
            detail_lines = []
            ow = getattr(iq, "original_width", 0)
            oh = getattr(iq, "original_height", 0)
            out_w = getattr(iq, "output_width", 0)
            out_h = getattr(iq, "output_height", 0)
            if ow and oh:
                detail_lines.append(f"Original: {ow}×{oh}")
            if out_w and out_h:
                detail_lines.append(f"Output: {out_w}×{out_h}")
            detail_lines.append(
                f"Verified: {'Yes' if getattr(iq, 'enhancement_verified', False) or iq.enhancement_applied else 'No'}"
            )
            if getattr(iq, "super_resolution_used", False):
                detail_lines.extend(
                    [
                        f"Model: {getattr(iq, 'sr_model', '') or 'Real-ESRGAN'}",
                        f"Scale: {getattr(iq, 'sr_scale', 1)}×",
                        f"Processing: {getattr(iq, 'sr_device', '') or 'cpu'}",
                    ]
                )
            detail_lines.extend(
                [
                    f"Before quality (internal): {iq.before_quality:.0%}",
                    f"After quality (internal): {iq.after_quality:.0%}",
                    f"Quality delta (internal): {getattr(iq, 'quality_delta_percent', iq.improvement_percent):+.1f}%",
                    f"Operations: {', '.join(iq.enhancement_operations) or '—'}",
                ]
            )
            if reason:
                detail_lines.append(reason)
            if detail_lines:
                with st.expander(t("streamlit.viewer.enhancement_details"), expanded=False):
                    st.caption("\n".join(detail_lines))
        st.markdown("<div style='height:0.45rem'></div>", unsafe_allow_html=True)

    enhanced = result.enhanced_preview_path
    if (
        enhanced
        and Path(enhanced).is_file()
        and result.image_quality
        and result.image_quality.enhancement_applied
    ):
        with st.expander(t("streamlit.results.enhanced"), expanded=False):
            st.image(str(enhanced), use_container_width=True)

    technical_sections = [
        (t("section.environment"), format_environment(result)),
        (t("section.objects"), format_detected_objects(result)),
        (t("streamlit.section.object_details"), format_object_details(result)),
        (t("section.relationships"), format_relationships(result)),
        (t("section.activities"), format_activities(result)),
        (t("streamlit.section.color_palette"), format_color_palette(result)),
        (t("streamlit.section.scene_graph"), format_scene_graph(result)),
        (t("section.image_quality"), format_image_quality(result)),
        (t("section.quality"), format_quality_report(result)),
        (t("section.metrics"), format_execution_metrics(result)),
    ]

    with st.expander(t("streamlit.results.technical"), expanded=False):
        st.caption(t("streamlit.results.technical_hint"))
        if not competition:
            _render_quality_indicators(result)
        search = st.text_input(t("results.search"), key="tech_results_search")
        query = search.lower().strip()
        for title, body in technical_sections:
            if not body or body == "None verified.":
                continue
            if query and query not in title.lower() and query not in body.lower():
                continue
            with st.expander(title, expanded=False):
                st.markdown(
                    f'<div class="glass-card" style="padding:0.85rem 0.95rem;">'
                    f'<pre style="white-space:pre-wrap;margin:0;color:var(--text-secondary);'
                    f'font-family:Outfit,Segoe UI,sans-serif;font-size:0.92rem;line-height:1.55;">'
                    f"{_escape(body)}</pre></div>",
                    unsafe_allow_html=True,
                )

    if not competition:
        with st.expander(t("streamlit.results.debug"), expanded=False):
            st.caption(t("streamlit.results.debug_hint"))
            conf = format_caption_confidence(result)
            if conf:
                st.markdown(
                    f'<div class="glass-card"><pre style="white-space:pre-wrap;margin:0;color:#D7CFE8;">{_escape(conf)}</pre></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption(t("msg.no_sentence_confidence"))
    else:
        with st.expander(t("streamlit.results.debug"), expanded=False):
            st.caption(t("streamlit.results.debug_competition_hint"))
