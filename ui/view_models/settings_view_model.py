"""Settings presentation and theme switching."""

from PySide6.QtCore import QObject, Signal

from ui.controllers.settings_controller import SettingsController
from ui.i18n.translator import get_translator
from ui.preferences.ui_preferences import (
    load_comparison_mode,
    load_competition_mode,
    load_enable_enhancement,
    load_enable_sam2,
    load_enable_super_resolution,
    load_high_contrast,
    load_language,
    load_large_font,
    save_comparison_mode,
    save_competition_mode,
    save_enable_enhancement,
    save_enable_sam2,
    save_enable_super_resolution,
    save_high_contrast,
    save_language,
    save_large_font,
)
from ui.themes.theme_manager import ThemeManager


class SettingsViewModel(QObject):
    """Exposes settings values and appearance changes to the UI."""

    settings_changed = Signal()

    def __init__(self, controller: SettingsController, theme_manager: ThemeManager) -> None:
        super().__init__()
        self._controller = controller
        self._theme_manager = theme_manager
        get_translator().set_language(load_language())

    @property
    def language_code(self) -> str:
        return get_translator().language

    @property
    def competition_mode_enabled(self) -> bool:
        return load_competition_mode()

    @property
    def high_contrast_enabled(self) -> bool:
        return load_high_contrast()

    @property
    def large_font_enabled(self) -> bool:
        return load_large_font()

    @property
    def enhancement_enabled(self) -> bool:
        return load_enable_enhancement()

    @property
    def super_resolution_enabled(self) -> bool:
        return load_enable_super_resolution()

    @property
    def sam2_enabled(self) -> bool:
        return load_enable_sam2()

    @property
    def comparison_mode_enabled(self) -> bool:
        return load_comparison_mode()

    @property
    def app_name(self) -> str:
        return self._controller.app_config.app_name

    @property
    def app_version(self) -> str:
        return self._controller.app_config.app_version

    @property
    def exports_dir(self) -> str:
        return str(self._controller.app_config.paths.exports_dir)

    @property
    def models_dir(self) -> str:
        return str(self._controller.app_config.paths.models_dir)

    @property
    def pipeline_timeout_seconds(self) -> int:
        return self._controller.app_config.hardware.pipeline_timeout_seconds

    @property
    def cpu_fallback_enabled(self) -> bool:
        return self._controller.app_config.hardware.cpu_fallback_enabled

    @property
    def log_level(self) -> str:
        return self._controller.app_config.logging.level

    @property
    def theme_name(self) -> str:
        return self._theme_manager.theme_name()

    @property
    def font_family(self) -> str:
        return self._controller.theme_config.font_family

    @property
    def font_size(self) -> int:
        return self._controller.theme_config.font_size

    def set_theme(self, theme_name: str, window: object) -> None:
        from PySide6.QtWidgets import QWidget

        if isinstance(window, QWidget):
            self._theme_manager.apply_theme_name(theme_name, window)
            self.settings_changed.emit()

    def set_language(self, code: str) -> None:
        import os

        normalized = (code or "en").lower().strip()
        os.environ["SENTIVIS_UI_LANGUAGE"] = normalized
        save_language(normalized)
        get_translator().set_language(normalized)
        try:
            from language.refinement.caption_refiner import clear_ui_language_cache
            from streamlit_app.catalog import get_catalog
            from streamlit_app.preferences import save_language as save_streamlit_language

            clear_ui_language_cache()
            get_catalog().set_language(normalized)
            save_streamlit_language(normalized)
        except Exception:  # noqa: BLE001
            pass
        self.settings_changed.emit()

    def set_competition_mode(self, enabled: bool) -> None:
        save_competition_mode(enabled)
        self.settings_changed.emit()

    def set_high_contrast(self, enabled: bool, window: object) -> None:
        from PySide6.QtWidgets import QWidget

        save_high_contrast(enabled)
        if isinstance(window, QWidget):
            self._theme_manager.apply_theme_name(self.theme_name, window)
        self.settings_changed.emit()

    def set_large_font(self, enabled: bool, window: object) -> None:
        from PySide6.QtWidgets import QWidget

        save_large_font(enabled)
        if isinstance(window, QWidget):
            self._theme_manager.apply_theme_name(self.theme_name, window)
        self.settings_changed.emit()

    def set_enhancement(self, enabled: bool) -> None:
        save_enable_enhancement(enabled)
        self.settings_changed.emit()

    def set_super_resolution(self, enabled: bool) -> None:
        save_enable_super_resolution(enabled)
        self.settings_changed.emit()

    def set_sam2(self, enabled: bool) -> None:
        save_enable_sam2(enabled)
        self.settings_changed.emit()

    def set_comparison_mode(self, enabled: bool) -> None:
        save_comparison_mode(enabled)
        self.settings_changed.emit()
