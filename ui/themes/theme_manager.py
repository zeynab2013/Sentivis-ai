"""Theme application via token engine."""

from dataclasses import replace

from PySide6.QtWidgets import QApplication, QWidget

from core.config.theme_config import ThemeConfig
from ui.design import DARK_TOKENS, LIGHT_TOKENS
from ui.design.tokens import DesignTokens
from ui.themes.theme_engine import render_stylesheet


class ThemeManager:
    """Applies token-driven QSS themes to the application."""

    def __init__(self, config: ThemeConfig) -> None:
        self._config = config
        initial = "light" if "light" in config.name.lower() else "dark"
        base = LIGHT_TOKENS if initial == "light" else DARK_TOKENS
        self._active_tokens = self._merge_config(base)

    @property
    def active_tokens(self) -> DesignTokens:
        return self._active_tokens

    def apply(self, widget: QWidget) -> None:
        self.apply_tokens(self._active_tokens, widget)

    def apply_tokens(self, tokens: DesignTokens, widget: QWidget | None = None) -> None:
        self._active_tokens = tokens
        stylesheet = render_stylesheet(tokens)
        if widget is not None:
            widget.setStyleSheet(stylesheet)
            return
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.setStyleSheet(stylesheet)

    def apply_theme_name(self, theme_name: str, widget: QWidget) -> None:
        from ui.preferences.ui_preferences import load_high_contrast, load_large_font

        base = LIGHT_TOKENS if theme_name.lower() == "light" else DARK_TOKENS
        tokens = self._merge_config(base)
        if load_high_contrast():
            tokens = replace(
                tokens,
                background="#000000",
                surface="#0a0a0a",
                card="#111111",
                border="#ffffff",
                text_primary="#ffffff",
                text_secondary="#e0e0e0",
                focus_ring="#ffff00",
            )
        if load_large_font():
            tokens = replace(
                tokens,
                font_size_sm=tokens.font_size_sm + 2,
                font_size_md=tokens.font_size_md + 2,
                font_size_lg=tokens.font_size_lg + 2,
                font_size_xl=tokens.font_size_xl + 3,
            )
        self.apply_tokens(tokens, widget)

    def theme_name(self) -> str:
        return "light" if self._active_tokens.background == LIGHT_TOKENS.background else "dark"

    def _merge_config(self, tokens: DesignTokens) -> DesignTokens:
        return replace(
            tokens,
            font_family=self._config.font_family,
            font_size_md=self._config.font_size,
        )
