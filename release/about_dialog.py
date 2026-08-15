"""Production About dialog for release engineering."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout, QWidget

from release.metadata import ReleaseInfo
from release.version import WEBSITE_PLACEHOLDER
from ui.branding.logo_provider import load_logo_pixmap
from ui.i18n.translator import tr


class AboutDialog(QDialog):
    """Production About dialog displaying release metadata."""

    def __init__(self, release_info: ReleaseInfo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("app.about.title"))
        self.setModal(True)
        self.resize(520, 460)

        layout = QVBoxLayout(self)
        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = load_logo_pixmap(96)
        if not pixmap.isNull():
            logo.setPixmap(pixmap)
        layout.addWidget(logo)

        title = QLabel(f"<h2>{release_info.application_name}</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(title)

        summary = QLabel(release_info.full_version_line)
        summary.setWordWrap(True)
        summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(summary)

        details = QLabel(self._detail_text(release_info))
        details.setWordWrap(True)
        details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(details)

        credits = QLabel(
            "<b>Credits</b><br>"
            "Sentivis AI Team · Ultralytics YOLO · Salesforce BLIP · Google Gemma<br>"
            "Built with PySide6, PyTorch, and Hugging Face Transformers"
        )
        credits.setTextFormat(Qt.TextFormat.RichText)
        credits.setWordWrap(True)
        layout.addWidget(credits)

        website = QLabel(f'<a href="{release_info.website}">{release_info.website or WEBSITE_PLACEHOLDER}</a>')
        website.setOpenExternalLinks(True)
        layout.addWidget(website)

        license_label = QLabel("License: Proprietary — see LICENSE and release/resources/THIRD_PARTY_NOTICES.md")
        license_label.setWordWrap(True)
        layout.addWidget(license_label)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _detail_text(release_info: ReleaseInfo) -> str:
        return (
            f"Build number: {release_info.build_number}\n"
            f"Architecture: v{release_info.architecture_version}\n"
            f"AI Pipeline: v{release_info.ai_pipeline_version}\n"
            f"Model Registry: v{release_info.model_registry_version}\n"
            f"Configuration: v{release_info.configuration_version}\n"
            f"Git commit: {release_info.git_commit}\n"
            f"Build time: {release_info.build_timestamp}\n"
            f"Profile: {release_info.build_profile}"
        )


def show_about_dialog(parent: QWidget | None, release_info: ReleaseInfo) -> None:
    """Display the production About dialog."""
    AboutDialog(release_info, parent).exec()
