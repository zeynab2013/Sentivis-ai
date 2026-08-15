"""Branded splash screen shown during startup."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ui.branding.logo_provider import load_logo_pixmap
from ui.i18n.translator import tr


class SplashScreenWidget(QWidget):
    """Minimal splash overlay with logo and loading text."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setProperty("class", "Card")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo = QLabel()
        pixmap = load_logo_pixmap(96)
        if not pixmap.isNull():
            logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel(tr("app.name"))
        title.setProperty("class", "BrandTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slogan = QLabel(tr("app.slogan"))
        slogan.setProperty("class", "BrandSlogan")
        slogan.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status = QLabel(tr("splash.loading"))
        status.setProperty("class", "StatusLabel")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addWidget(slogan)
        layout.addWidget(status)
        self.resize(420, 320)

    @staticmethod
    def show_brief(app: QGuiApplication, ms: int = 1200) -> None:
        """Display splash briefly without blocking startup pipeline."""
        splash = SplashScreenWidget()
        splash.show()
        app.processEvents()
        QTimer.singleShot(ms, splash.close)
