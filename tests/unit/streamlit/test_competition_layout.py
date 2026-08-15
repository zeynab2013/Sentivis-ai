"""Regression: competition UI layout priorities (voice row + advanced viz)."""

from __future__ import annotations

from pathlib import Path


def test_voice_controls_not_cramped_in_caption_header() -> None:
    results = Path("streamlit_app/components/results.py").read_text(encoding="utf-8")
    tts = Path("streamlit_app/components/tts_control.py").read_text(encoding="utf-8")
    # Caption kicker must not share a [9, 2] squeeze with TTS controls.
    assert "st.columns([9, 2])" not in results
    assert "caption-tts-bar" in results
    assert "use_container_width=True" in tts
    assert "st.columns([1, 1]" in tts or 'st.columns([1, 1]' in tts


def test_advanced_visualization_demotes_overlay_controls() -> None:
    main = Path("streamlit_app/main.py").read_text(encoding="utf-8")
    assert "streamlit.viewer.advanced_viz" in main
    assert "st.expander" in main
    # Results column should be at least as wide as the image column ratio.
    assert "[0.85, 2.0, 2.05]" in main
    assert "_render_main_language_selector" in main
    assert 'nav_labels = [t("streamlit.nav.analyze")]' in main
    # Dashboard/Settings must not be routable in the competition UI.
    assert "render_dashboard" not in main
    assert 'nav == t("streamlit.nav.dashboard")' not in main
    assert 'nav == t("streamlit.nav.settings")' not in main


def test_advanced_viz_translation_key_present() -> None:
    import json

    en = json.loads(Path("translations/en.json").read_text(encoding="utf-8"))
    assert en["streamlit.viewer.advanced_viz"] == "Advanced Visualization"
    core = json.loads(Path("core/resources/translations/en.json").read_text(encoding="utf-8"))
    assert core["streamlit.viewer.advanced_viz"] == "Advanced Visualization"
