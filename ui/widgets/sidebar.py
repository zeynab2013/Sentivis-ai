"""Left navigation panel with project actions and history."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QListWidget, QStackedWidget, QVBoxLayout, QWidget

from ui.branding.logo_provider import load_logo_pixmap
from ui.components.button import SentivisButton
from ui.components.empty_state import EmptyStateWidget
from ui.components.status_badge import StatusBadge
from ui.i18n.translator import tr
from ui.models.operation_status import OperationStatus


class SidebarWidget(QWidget):
    """Navigation, workflow actions, session info, and session history."""

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("class", "Sidebar")
        self.setMinimumWidth(220)
        self.setMaximumWidth(260)
        layout = QVBoxLayout(self)

        self._logo = QLabel()
        self._logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = load_logo_pixmap(48)
        if not pixmap.isNull():
            self._logo.setPixmap(pixmap)
        layout.addWidget(self._logo)

        self._brand = QLabel(tr("app.name"))
        self._brand.setProperty("class", "BrandTitle")
        self._slogan = QLabel(tr("app.slogan"))
        self._slogan.setProperty("class", "BrandSlogan")
        layout.addWidget(self._brand)
        layout.addWidget(self._slogan)

        self._session_title = QLabel(tr("sidebar.session"))
        self._session_title.setProperty("class", "SectionTitle")
        layout.addWidget(self._session_title)
        self._session_info = QLabel(tr("sidebar.no_image"))
        self._session_info.setProperty("class", "StatusLabel")
        self._session_info.setWordWrap(True)
        layout.addWidget(self._session_info)

        self._nav_title = QLabel(tr("sidebar.navigation"))
        self._nav_title.setProperty("class", "SectionTitle")
        layout.addWidget(self._nav_title)

        self.open_button = SentivisButton(tr("button.open"), variant="secondary")
        self.random_button = SentivisButton(tr("button.random_image"), variant="secondary")
        self.clear_button = SentivisButton(tr("button.clear"), variant="secondary")
        self.analyze_button = SentivisButton(tr("button.analyze"), variant="primary")
        self.cancel_button = SentivisButton(tr("button.cancel"), variant="danger")
        self.settings_button = SentivisButton(tr("button.settings"), variant="secondary")
        self.presentation_button = SentivisButton(tr("button.presentation"), variant="secondary")
        self.open_button.setToolTip(tr("tooltip.open"))
        self.random_button.setToolTip(tr("tooltip.random_image"))
        self.clear_button.setToolTip(tr("tooltip.clear"))
        self.analyze_button.setToolTip(tr("tooltip.analyze"))
        self.cancel_button.setToolTip(tr("tooltip.cancel"))
        self.settings_button.setToolTip(tr("tooltip.settings"))
        self.presentation_button.setToolTip(tr("tooltip.presentation"))
        for button in (
            self.open_button,
            self.random_button,
            self.clear_button,
            self.analyze_button,
            self.cancel_button,
            self.settings_button,
            self.presentation_button,
        ):
            layout.addWidget(button)

        self._history_title = QLabel(tr("sidebar.recent"))
        self._history_title.setProperty("class", "SectionTitle")
        layout.addWidget(self._history_title)

        self._history_stack = QStackedWidget()
        self._history_empty = EmptyStateWidget(
            tr("sidebar.no_recent"),
            tr("sidebar.no_recent_hint"),
            action_hint=tr("sidebar.no_recent_action"),
        )
        self.history_list = QListWidget()
        self.history_list.setToolTip(tr("sidebar.history.tooltip"))
        self._history_stack.addWidget(self._history_empty)
        self._history_stack.addWidget(self.history_list)
        layout.addWidget(self._history_stack, stretch=1)

        self.status_badge = StatusBadge()
        self.status_label = QLabel("")
        self.status_label.setProperty("class", "StatusLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_badge)
        layout.addWidget(self.status_label)

    def retranslate_ui(self) -> None:
        pixmap = load_logo_pixmap(48)
        if not pixmap.isNull():
            self._logo.setPixmap(pixmap)
        self._brand.setText(tr("app.name"))
        self._slogan.setText(tr("app.slogan"))
        self._session_title.setText(tr("sidebar.session"))
        self._nav_title.setText(tr("sidebar.navigation"))
        self.open_button.setText(tr("button.open"))
        self.random_button.setText(tr("button.random_image"))
        self.clear_button.setText(tr("button.clear"))
        self.analyze_button.setText(tr("button.analyze"))
        self.cancel_button.setText(tr("button.cancel"))
        self.settings_button.setText(tr("button.settings"))
        self.presentation_button.setText(tr("button.presentation"))
        self.open_button.setToolTip(tr("tooltip.open"))
        self.random_button.setToolTip(tr("tooltip.random_image"))
        self.clear_button.setToolTip(tr("tooltip.clear"))
        self.analyze_button.setToolTip(tr("tooltip.analyze"))
        self.cancel_button.setToolTip(tr("tooltip.cancel"))
        self.settings_button.setToolTip(tr("tooltip.settings"))
        self.presentation_button.setToolTip(tr("tooltip.presentation"))
        self._history_title.setText(tr("sidebar.recent"))
        self.history_list.setToolTip(tr("sidebar.history.tooltip"))
        self._history_empty.set_content(
            tr("sidebar.no_recent"),
            tr("sidebar.no_recent_hint"),
            action_hint=tr("sidebar.no_recent_action"),
        )

    def bind_status(self, status: OperationStatus, message: str, device: str = "") -> None:
        self.status_badge.set_status(status, message)
        self.status_label.setText(f"{message}\n{device}" if device else message)

    def bind_session(
        self,
        *,
        image_name: str,
        duration_ms: float | None,
        competition_mode: bool,
        model_hint: str,
        presentation_mode: bool = False,
    ) -> None:
        if not image_name:
            self._session_info.setText("No image loaded\nOpen an image to begin.")
            return
        if presentation_mode:
            duration = f"{duration_ms:.0f} ms" if duration_ms is not None else "—"
            self._session_info.setText(f"Image: {image_name}\nAnalysis time: {duration}")
            return
        duration = f"{duration_ms:.0f} ms" if duration_ms is not None else "—"
        competition = "On" if competition_mode else "Off"
        model_line = model_hint or "Default pipeline models"
        self._session_info.setText(
            f"Image: {image_name}\n"
            f"Duration: {duration}\n"
            f"Competition mode: {competition}\n"
            f"Models: {model_line}"
        )

    def bind_history(self, entries: tuple[object, ...]) -> None:
        self.history_list.clear()
        if not entries:
            self._history_stack.setCurrentWidget(self._history_empty)
            return
        self._history_stack.setCurrentWidget(self.history_list)
        for entry in entries:
            image_name = getattr(entry, "image_name", str(entry))
            preview = getattr(entry, "caption_preview", "")
            analyzed_at = getattr(entry, "analyzed_at", "")
            duration_ms = getattr(entry, "duration_ms", 0.0)
            meta = f"{analyzed_at} · {duration_ms:.0f} ms" if analyzed_at else ""
            text = f"{image_name}\n{preview[:48]}" if preview else image_name
            if meta:
                text = f"{text}\n{meta}"
            self.history_list.addItem(text)
            if preview:
                list_item = self.history_list.item(self.history_list.count() - 1)
                if list_item is not None:
                    list_item.setToolTip(preview)
