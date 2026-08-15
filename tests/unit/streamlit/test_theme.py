"""Unit tests for premium theme CSS."""

from streamlit_app.theme import THEME_CSS, inject_theme


def test_theme_css_contains_brand_tokens() -> None:
    assert "#FF4FAD" in THEME_CSS or "#FF3FA4" in THEME_CSS
    assert "#FF63B8" in THEME_CSS or "#FF73C2" in THEME_CSS
    assert "#0E0818" in THEME_CSS
    assert "#171028" in THEME_CSS
    assert "#21163A" in THEME_CSS
    assert "#FFFFFF" in THEME_CSS
    assert "#D2C9E6" in THEME_CSS or "#D7CFE8" in THEME_CSS
    assert "rgba(255, 79, 173" in THEME_CSS
    assert "Outfit" in THEME_CSS
    assert "Source Serif 4" in THEME_CSS
    assert "glass-card" in THEME_CSS
    assert "caption-panel" in THEME_CSS
    assert "caption-tts-bar" in THEME_CSS
    assert "hero-header" in THEME_CSS
    assert "confidence-track" in THEME_CSS
    assert "stat-card" in THEME_CSS
    assert "stat-grid" in THEME_CSS
    assert "section-heading" in THEME_CSS
    assert "va-suggestions" in THEME_CSS
    assert "export-panel" in THEME_CSS
    assert "white-space: nowrap" in THEME_CSS
    assert "word-break: normal" in THEME_CSS
    assert "overflow-wrap: break-word" in THEME_CSS or "overflow-wrap: normal" in THEME_CSS
    assert "break-all" not in THEME_CSS
    assert "#FFD84A" not in THEME_CSS
    assert "#FF4FD8" not in THEME_CSS


def test_inject_theme_is_callable() -> None:
    assert callable(inject_theme)
