"""Reusable collapsible section widget."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFrame, QHBoxLayout, QTextEdit, QToolButton, QVBoxLayout, QWidget

from ui.components.button import SentivisButton
from ui.components.skeleton import SkeletonBlock


class CollapsibleSection(QWidget):
    """Card-style section that expands and collapses."""

    copy_requested = Signal(str)

    def __init__(self, title: str, *, expanded: bool = True) -> None:
        super().__init__()
        self._title = title
        self._plain_text = ""
        self.setProperty("class", "Card")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 8, 8, 4)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle.clicked.connect(self._on_toggle)
        header_layout.addWidget(self._toggle, stretch=1)

        self._copy_button = SentivisButton("Copy", variant="secondary")
        self._copy_button.setToolTip(f"Copy {title} to clipboard")
        self._copy_button.clicked.connect(self._copy_text)
        header_layout.addWidget(self._copy_button)

        self._body = QFrame()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(8, 0, 8, 8)

        self._content = QTextEdit()
        self._content.setReadOnly(True)
        self._content.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._content.setMinimumHeight(72)
        self._skeleton = SkeletonBlock(lines=4)
        self._skeleton.hide()
        body_layout.addWidget(self._content)
        body_layout.addWidget(self._skeleton)

        root.addWidget(header)
        root.addWidget(self._body)
        self._body.setVisible(expanded)

    @property
    def section_title(self) -> str:
        return self._title

    @property
    def plain_text(self) -> str:
        return self._plain_text

    def set_title(self, title: str) -> None:
        self._title = title
        self._toggle.setText(title)
        self._copy_button.setToolTip(f"Copy {title} to clipboard")

    def set_text(self, text: str) -> None:
        self._plain_text = text
        self._content.setPlainText(text)
        self._skeleton.hide()
        self._content.show()

    def set_loading(self, active: bool) -> None:
        if active:
            self._content.hide()
            self._skeleton.show()
        else:
            self._skeleton.hide()
            self._content.show()

    def expand(self) -> None:
        self._toggle.setChecked(True)
        self._on_toggle(True)

    def collapse(self) -> None:
        self._toggle.setChecked(False)
        self._on_toggle(False)

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def matches_filter(self, query: str) -> bool:
        if not query:
            return True
        haystack = f"{self._title}\n{self._plain_text}".lower()
        return query.lower() in haystack

    def set_visible_for_filter(self, visible: bool) -> None:
        self.setVisible(visible)

    def _copy_text(self) -> None:
        if self._plain_text:
            QGuiApplication.clipboard().setText(self._plain_text)
            self.copy_requested.emit(self._title)

    def _on_toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
