"""Professional settings dialog with categorized options."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.components.button import SentivisButton
from ui.i18n.translator import SUPPORTED_LANGUAGES, tr
from ui.view_models.settings_view_model import SettingsViewModel

_LANGUAGE_LABELS = {
    "en": "English",
    "fa": "فارسی",
    "es": "Español",
    "zh": "中文",
    "fr": "Français",
}


@dataclass(frozen=True)
class SettingField:
    """One setting row with description and current value."""

    label: str
    description: str
    value: str
    default: str


class SettingsDialog(QDialog):
    """Tabbed settings experience with descriptions and restore defaults."""

    SUPPORTED_FORMATS = (
        "TXT — full analysis report",
        "Markdown — structured report",
        "JSON — full analysis payload",
        "PDF — printable summary report",
    )

    def __init__(
        self,
        view_model: SettingsViewModel,
        parent: QWidget | None = None,
        *,
        presentation_mode: bool = False,
    ) -> None:
        super().__init__(parent)
        self._vm = view_model
        self.setWindowTitle(tr("settings.title"))
        self.resize(680, 560)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems([tr("settings.theme.dark"), tr("settings.theme.light")])
        self._theme_combo.setCurrentText(
            tr("settings.theme.light") if view_model.theme_name == "light" else tr("settings.theme.dark")
        )

        self._language_combo = QComboBox()
        for code in SUPPORTED_LANGUAGES:
            self._language_combo.addItem(_LANGUAGE_LABELS.get(code, code.upper()), code)
        current_index = max(0, SUPPORTED_LANGUAGES.index(view_model.language_code))
        self._language_combo.setCurrentIndex(current_index)

        self._competition_check = QCheckBox(tr("settings.competition_enable"))
        self._competition_check.setChecked(view_model.competition_mode_enabled)
        self._high_contrast_check = QCheckBox(tr("settings.accessibility.high_contrast"))
        self._high_contrast_check.setChecked(view_model.high_contrast_enabled)
        self._large_font_check = QCheckBox(tr("settings.accessibility.large_font"))
        self._large_font_check.setChecked(view_model.large_font_enabled)
        self._enhancement_check = QCheckBox(tr("settings.enhancement.enable"))
        self._enhancement_check.setChecked(view_model.enhancement_enabled)
        self._super_resolution_check = QCheckBox(tr("settings.enhancement.super_resolution"))
        self._super_resolution_check.setChecked(view_model.super_resolution_enabled)
        self._sam2_check = QCheckBox(tr("settings.enhancement.sam2"))
        self._sam2_check.setChecked(view_model.sam2_enabled)
        self._comparison_check = QCheckBox(tr("settings.enhancement.comparison"))
        self._comparison_check.setChecked(view_model.comparison_mode_enabled)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._build_category(tr("settings.general"), self._general_fields()), tr("settings.general"))
        tabs.addTab(
            self._build_category(
                tr("settings.appearance"),
                self._appearance_fields(),
                controls={
                    tr("settings.theme"): self._theme_combo,
                    tr("settings.language"): self._language_combo,
                },
            ),
            tr("settings.appearance"),
        )
        tabs.addTab(self._build_category(tr("settings.models"), self._model_fields()), tr("settings.models"))
        tabs.addTab(
            self._build_category(
                tr("settings.performance"),
                self._performance_fields(),
                controls={
                    tr("settings.enhancement.enable"): self._enhancement_check,
                    tr("settings.enhancement.super_resolution"): self._super_resolution_check,
                    tr("settings.enhancement.sam2"): self._sam2_check,
                    tr("settings.enhancement.comparison"): self._comparison_check,
                },
            ),
            tr("settings.performance"),
        )
        tabs.addTab(
            self._build_category(
                tr("settings.competition"),
                self._competition_fields(),
                controls={tr("settings.competition"): self._competition_check},
            ),
            tr("settings.competition"),
        )
        tabs.addTab(self._build_category(tr("settings.exports"), self._export_fields()), tr("settings.exports"))
        tabs.addTab(
            self._build_category(
                tr("settings.accessibility"),
                self._accessibility_fields(),
                controls={
                    tr("settings.accessibility.high_contrast"): self._high_contrast_check,
                    tr("settings.accessibility.large_font"): self._large_font_check,
                },
            ),
            tr("settings.accessibility"),
        )
        if not presentation_mode:
            tabs.addTab(
                self._build_category(tr("settings.diagnostics"), self._diagnostics_fields()),
                tr("settings.diagnostics"),
            )
            tabs.addTab(
                self._build_category(tr("settings.advanced"), self._advanced_fields()),
                tr("settings.advanced"),
            )
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        restore = SentivisButton(tr("button.restore_defaults"), variant="secondary")
        restore.clicked.connect(self._restore_defaults)
        row = QHBoxLayout()
        row.addWidget(restore)
        row.addStretch()
        row.addWidget(buttons)
        layout.addLayout(row)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def selected_theme(self) -> str:
        return "light" if self._theme_combo.currentText() == tr("settings.theme.light") else "dark"

    def selected_language(self) -> str:
        return str(self._language_combo.currentData())

    def selected_competition_mode(self) -> bool:
        return self._competition_check.isChecked()

    def selected_high_contrast(self) -> bool:
        return self._high_contrast_check.isChecked()

    def selected_large_font(self) -> bool:
        return self._large_font_check.isChecked()

    def selected_enhancement(self) -> bool:
        return self._enhancement_check.isChecked()

    def selected_super_resolution(self) -> bool:
        return self._super_resolution_check.isChecked()

    def selected_sam2(self) -> bool:
        return self._sam2_check.isChecked()

    def selected_comparison_mode(self) -> bool:
        return self._comparison_check.isChecked()

    def _build_category(
        self,
        title: str,
        fields: tuple[SettingField, ...],
        controls: dict[str, QWidget] | None = None,
    ) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel(title)
        heading.setProperty("class", "SectionTitle")
        layout.addWidget(heading)
        form = QFormLayout()
        for field in fields:
            block = QVBoxLayout()
            name = QLabel(field.label)
            name.setProperty("class", "SectionTitle")
            desc = QLabel(field.description)
            desc.setWordWrap(True)
            desc.setProperty("class", "StatusLabel")
            value = QLabel(f"{tr('settings.current')}: {field.value}\n{tr('settings.default')}: {field.default}")
            block.addWidget(name)
            block.addWidget(desc)
            control = controls.get(field.label) if controls else None
            if control is not None:
                block.addWidget(control)
            else:
                block.addWidget(value)
            wrapper = QWidget()
            wrapper.setLayout(block)
            form.addRow(wrapper)
        layout.addLayout(form)
        layout.addStretch()
        return page

    def _appearance_fields(self) -> tuple[SettingField, ...]:
        return (
            SettingField(
                tr("settings.theme"),
                tr("settings.theme.desc"),
                self._vm.theme_name.title(),
                tr("settings.theme.dark"),
            ),
            SettingField(
                tr("settings.language"),
                tr("settings.language.desc"),
                _LANGUAGE_LABELS.get(self._vm.language_code, self._vm.language_code),
                "English",
            ),
            SettingField(
                "Font",
                tr("settings.font.desc"),
                f"{self._vm.font_family} {self._vm.font_size}pt",
                "Segoe UI 10pt",
            ),
        )

    def _general_fields(self) -> tuple[SettingField, ...]:
        return (
            SettingField(
                tr("settings.general"),
                tr("settings.general.desc"),
                f"{self._vm.app_name} {self._vm.app_version}",
                f"{self._vm.app_name} {self._vm.app_version}",
            ),
        )

    def _model_fields(self) -> tuple[SettingField, ...]:
        return (
            SettingField(
                tr("settings.models"),
                tr("settings.models.desc"),
                self._vm.models_dir,
                self._vm.models_dir,
            ),
        )

    def _performance_fields(self) -> tuple[SettingField, ...]:
        return (
            SettingField(
                tr("settings.enhancement.enable"),
                tr("settings.enhancement.enable.desc"),
                tr("settings.enabled") if self._vm.enhancement_enabled else tr("settings.disabled"),
                tr("settings.enabled"),
            ),
            SettingField(
                tr("settings.enhancement.super_resolution"),
                tr("settings.enhancement.super_resolution.desc"),
                tr("settings.enabled") if self._vm.super_resolution_enabled else tr("settings.disabled"),
                tr("settings.disabled"),
            ),
            SettingField(
                tr("settings.enhancement.sam2"),
                tr("settings.enhancement.sam2.desc"),
                tr("settings.enabled") if self._vm.sam2_enabled else tr("settings.disabled"),
                tr("settings.enabled"),
            ),
            SettingField(
                tr("settings.enhancement.comparison"),
                tr("settings.enhancement.comparison.desc"),
                tr("settings.enabled") if self._vm.comparison_mode_enabled else tr("settings.disabled"),
                tr("settings.disabled"),
            ),
            SettingField(
                tr("settings.performance.timeout"),
                tr("settings.performance.timeout.desc"),
                str(self._vm.pipeline_timeout_seconds),
                "600",
            ),
            SettingField(
                tr("settings.performance.cpu_fallback"),
                tr("settings.performance.cpu_fallback.desc"),
                tr("settings.enabled") if self._vm.cpu_fallback_enabled else tr("settings.disabled"),
                tr("settings.enabled"),
            ),
        )

    def _competition_fields(self) -> tuple[SettingField, ...]:
        return (
            SettingField(
                tr("settings.competition"),
                tr("settings.competition.desc"),
                tr("settings.enabled") if self._vm.competition_mode_enabled else tr("settings.disabled"),
                tr("settings.disabled"),
            ),
        )

    def _export_fields(self) -> tuple[SettingField, ...]:
        formats = "\n".join(f"• {item}" for item in self.SUPPORTED_FORMATS)
        return (
            SettingField(
                tr("settings.exports.dir"),
                tr("settings.exports.dir.desc"),
                self._vm.exports_dir,
                self._vm.exports_dir,
            ),
            SettingField(
                tr("settings.exports.formats"),
                tr("settings.exports.formats.desc"),
                formats,
                formats,
            ),
        )

    def _accessibility_fields(self) -> tuple[SettingField, ...]:
        return (
            SettingField(
                tr("settings.accessibility.high_contrast"),
                tr("settings.accessibility.high_contrast.desc"),
                tr("settings.enabled") if self._vm.high_contrast_enabled else tr("settings.disabled"),
                tr("settings.disabled"),
            ),
            SettingField(
                tr("settings.accessibility.large_font"),
                tr("settings.accessibility.large_font.desc"),
                tr("settings.enabled") if self._vm.large_font_enabled else tr("settings.disabled"),
                tr("settings.disabled"),
            ),
        )

    def _diagnostics_fields(self) -> tuple[SettingField, ...]:
        return (
            SettingField(
                tr("settings.diagnostics.logging"),
                tr("settings.diagnostics.logging.desc"),
                self._vm.log_level,
                "INFO",
            ),
        )

    def _advanced_fields(self) -> tuple[SettingField, ...]:
        return (
            SettingField(
                tr("settings.advanced.config"),
                tr("settings.advanced.config.desc"),
                tr("settings.advanced.config.value"),
                tr("settings.advanced.config.value"),
            ),
        )

    def _restore_defaults(self) -> None:
        self._theme_combo.setCurrentText(tr("settings.theme.dark"))
        self._language_combo.setCurrentIndex(0)
        self._competition_check.setChecked(False)
        self._enhancement_check.setChecked(True)
        self._super_resolution_check.setChecked(False)
        self._sam2_check.setChecked(True)
        self._comparison_check.setChecked(False)
        self._high_contrast_check.setChecked(False)
        self._large_font_check.setChecked(False)
