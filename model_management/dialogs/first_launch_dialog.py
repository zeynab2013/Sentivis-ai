"""First-launch model download dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.constants.model_kinds import ModelKind
from model_management.auth import resolve_hf_token, store_hf_token
from model_management.service import ModelManagementService


def _format_bytes(num: int) -> str:
    if num >= 1024**3:
        return f"{num / (1024**3):.1f} GB"
    if num >= 1024**2:
        return f"{num / (1024**2):.1f} MB"
    return f"{num:,} bytes"


class FirstLaunchDialog(QDialog):
    """Professional first-launch experience for missing production models."""

    def __init__(self, service: ModelManagementService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = service
        self._selected: set[ModelKind] = {record.kind for record in service.missing_mandatory()}
        self._skipped = False
        self.setWindowTitle("Sentivis AI — Model Setup")
        self.setModal(True)
        self.resize(640, 520)

        layout = QVBoxLayout(self)
        heading = QLabel("<h2>Required AI Models</h2>")
        heading.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(heading)

        intro = QLabel(
            "Sentivis AI uses production-grade vision and language models. "
            "The following models must be installed before analysis can begin."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._model_list = QListWidget()
        for record in service.records:
            if not record.mandatory:
                continue
            size = _format_bytes(record.expected_size_bytes or 0)
            status = record.installation_status.value.replace("_", " ").title()
            self._model_list.addItem(f"{record.display_name} — {size} — {status}")
        layout.addWidget(self._model_list)

        summary = QLabel(self._summary_text())
        summary.setWordWrap(True)
        summary.setProperty("class", "StatusLabel")
        layout.addWidget(summary)
        self._summary = summary
        self._token_field: QLineEdit | None = None

        if not resolve_hf_token():
            token_row = QHBoxLayout()
            token_row.addWidget(QLabel("Hugging Face token (if required):"))
            self._token_field = QLineEdit()
            self._token_field.setEchoMode(QLineEdit.EchoMode.Password)
            self._token_field.setPlaceholderText("Optional — set HF_TOKEN environment variable")
            token_row.addWidget(self._token_field)
            layout.addLayout(token_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.hide()
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QDialogButtonBox()
        self._download_all = QPushButton("Download All")
        self._download_selected = QPushButton("Download Selected")
        self._skip = QPushButton("Skip")
        self._cancel = QPushButton("Cancel")
        buttons.addButton(self._download_all, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(self._download_selected, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self._skip, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.addButton(self._cancel, QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(buttons)

        self._download_all.clicked.connect(self._on_download_all)
        self._download_selected.clicked.connect(self._on_download_selected)
        self._skip.clicked.connect(self._on_skip)
        self._cancel.clicked.connect(self.reject)

        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._poll_progress)

    @property
    def skipped(self) -> bool:
        return self._skipped

    def _summary_text(self) -> str:
        download_bytes = self._service.estimated_download_bytes()
        free_bytes = self._service.free_disk_bytes()
        return (
            f"Total download size (estimate): {_format_bytes(download_bytes)}\n"
            f"Available disk space: {_format_bytes(free_bytes)}\n"
            f"Target hardware: Windows 11 · Python 3.10.11 · NVIDIA GPU recommended"
        )

    def _store_token_if_needed(self) -> None:
        if self._token_field is None:
            return
        token = self._token_field.text().strip()
        if token:
            store_hf_token(token)

    def _on_download_all(self) -> None:
        self._store_token_if_needed()
        self._begin_download(tuple(record.kind for record in self._service.missing_mandatory()))

    def _on_download_selected(self) -> None:
        self._store_token_if_needed()
        self._begin_download(tuple(self._selected))

    def _on_skip(self) -> None:
        self._skipped = True
        self.reject()

    def _begin_download(self, kinds: tuple[ModelKind, ...]) -> None:
        if not kinds:
            self.accept()
            return
        self._progress.show()
        self._download_all.setEnabled(False)
        self._download_selected.setEnabled(False)
        self._service.download_selected(kinds, on_progress=self._on_progress)
        self._timer.start()

    def _on_progress(self, progress: object) -> None:
        from model_management.download.progress import DownloadProgress

        if isinstance(progress, DownloadProgress):
            self._status.setText(progress.message or progress.state.value)
            if progress.percent is not None:
                self._progress.setRange(0, 100)
                self._progress.setValue(int(progress.percent))

    def _poll_progress(self) -> None:
        if not self._service.downloader.is_running():
            self._timer.stop()
            self._progress.hide()
            self._service.wait_for_downloads(timeout_seconds=1)
            self._service.refresh()
            if self._service.all_mandatory_ready():
                self.accept()
            else:
                self._status.setText("Some models could not be installed. Check network and disk space.")
                self._download_all.setEnabled(True)
                self._download_selected.setEnabled(True)
