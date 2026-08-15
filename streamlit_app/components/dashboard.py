"""Metrics dashboard with Plotly charts."""

from __future__ import annotations

import streamlit as st

from core.contracts.pipeline import PipelineResult
from streamlit_app.i18n import t

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _metric_cards(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="stat-card"><div class="stat-label">{_escape(label)}</div>'
        f'<div class="stat-value">{_escape(value)}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)


def render_dashboard(result: PipelineResult | None) -> None:
    st.markdown(
        f'<div class="hero-header" style="padding:1.1rem;">'
        f'<p class="hero-title gradient-header" style="font-size:1.8rem;">{t("streamlit.nav.dashboard")}</p>'
        f'<p class="hero-subtitle">{t("app.slogan")}</p></div>',
        unsafe_allow_html=True,
    )
    if result is None:
        st.markdown(
            f'<div class="glass-card"><p style="color:#D7CFE8;margin:0;">{t("streamlit.dashboard.empty")}</p></div>',
            unsafe_allow_html=True,
        )
        return

    metrics = result.metrics
    context = result.scene_context
    quality = result.quality_report

    _metric_cards(
        [
            (t("streamlit.metric.objects"), str(metrics.objects_detected)),
            (t("streamlit.metric.relations"), str(metrics.relationships_inferred)),
            (t("streamlit.metric.activities"), str(metrics.activities_inferred)),
            (t("streamlit.metric.caption_quality"), f"{quality.overall_quality:.0%}"),
        ]
    )
    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
    setting = (context.environment.setting or "—").strip()
    _metric_cards(
        [
            ("RAM", f"{metrics.peak_ram_mb:.0f} MB"),
            ("VRAM", f"{metrics.peak_vram_mb:.0f} MB"),
            (t("streamlit.metric.duration"), f"{metrics.total_duration_ms:.0f} ms"),
            (t("section.environment"), setting),
        ]
    )

    # Quality radar
    if go is not None:
        # Radar cannot draw N/A; omit unavailable metrics instead of faking 0%/100%.
        metric_pairs = [
            (t("streamlit.quality.evidence"), quality.evidence_consistency),
            (t("streamlit.quality.objects"), quality.object_coverage),
            (t("streamlit.quality.relationships"), quality.relationship_coverage),
            (t("streamlit.quality.activities"), quality.activity_coverage),
            (
                t("streamlit.quality.hallucination_safety"),
                None
                if quality.hallucination_risk is None
                else max(0.0, 1.0 - quality.hallucination_risk),
            ),
        ]
        categories = [name for name, value in metric_pairs if value is not None]
        values = [float(value) for _, value in metric_pairs if value is not None]
        if len(values) < 3:
            st.caption(t("streamlit.quality.evidence") + ": N/A metrics omitted from radar")
            return
        fig_radar = go.Figure(
            data=[
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    line=dict(color="#FF3FA4", width=2),
                    fillcolor="rgba(255, 63, 164, 0.28)",
                )
            ]
        )
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(36, 24, 67, 0.65)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(215,207,232,0.25)"),
                angularaxis=dict(gridcolor="rgba(215,207,232,0.2)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF", family="Outfit"),
            margin=dict(l=40, r=40, t=40, b=20),
            height=360,
            showlegend=False,
            title=t("streamlit.dashboard.quality_radar"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    if go is None or not metrics.stage_metrics:
        return

    stages = [item.stage.display_name for item in metrics.stage_metrics[:10]]
    durations = [item.duration_ms for item in metrics.stage_metrics[:10]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=durations,
                y=stages,
                orientation="h",
                marker=dict(color="rgba(255, 63, 164, 0.85)", line=dict(color="#FF3FA4", width=1)),
            )
        ]
    )
    fig.update_layout(
        title=t("streamlit.dashboard.pipeline_timing"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(36,24,67,0.85)",
        font=dict(color="#FFFFFF", family="Outfit"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
