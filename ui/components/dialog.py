"""Design-system dialog helpers."""

from PySide6.QtWidgets import QMessageBox, QWidget


class SentivisDialog:
    """Consistent dialog presentation using token-styled message boxes."""

    @staticmethod
    def information(parent: QWidget, title: str, summary: str, detail: str = "") -> None:
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(title)
        box.setText(summary)
        if detail:
            box.setInformativeText(detail)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    @staticmethod
    def warning(parent: QWidget, title: str, summary: str, detail: str = "") -> None:
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(summary)
        if detail:
            box.setInformativeText(detail)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    @staticmethod
    def error(parent: QWidget, title: str, summary: str, detail: str = "") -> None:
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(summary)
        if detail:
            box.setDetailedText(detail)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    @staticmethod
    def confirm(parent: QWidget, title: str, summary: str, detail: str = "") -> bool:
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        box.setText(summary)
        if detail:
            box.setInformativeText(detail)
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        return box.exec() == QMessageBox.StandardButton.Yes
