"""Professional error presentation for the desktop UI."""

from PySide6.QtWidgets import QWidget

from ui.components.dialog import SentivisDialog


def suggested_action_for_message(message: str) -> str:
    lower = message.lower()
    if "cancel" in lower:
        return "Load an image and start analysis again when ready."
    if "memory" in lower or "vram" in lower:
        return "Close other GPU applications, then retry with a smaller image or CPU fallback."
    if "image" in lower or "format" in lower:
        return "Choose a supported PNG, JPG, WEBP, or BMP file under the configured size limit."
    if "model" in lower or "load" in lower:
        return "Verify model files and network access, then restart the application."
    if "timeout" in lower:
        return "Retry with a smaller image or increase the pipeline timeout in configuration."
    return "Review the image and settings, then try again."


def recovery_status_for_message(message: str) -> str:
    lower = message.lower()
    if "cancel" in lower:
        return "Pipeline cancelled — no changes were saved."
    return "The application is ready for a new analysis attempt."


class ErrorDialog:
    """Shows user-safe error information without developer diagnostics."""

    @staticmethod
    def show_pipeline_error(parent: QWidget, problem: str) -> None:
        SentivisDialog.error(
            parent,
            "Sentivis AI",
            "Analysis could not be completed",
            f"{problem}\n\nSuggested action:\n{suggested_action_for_message(problem)}\n\n"
            f"Recovery status:\n{recovery_status_for_message(problem)}",
        )

    @staticmethod
    def show_export_error(parent: QWidget, problem: str) -> None:
        SentivisDialog.error(
            parent,
            "Sentivis AI",
            "Export failed",
            f"{problem}\n\nSuggested action:\nVerify the exports folder is writable and try again.\n\n"
            "Recovery status:\nYour analysis results remain available in the application.",
        )

    @staticmethod
    def show_warnings(parent: QWidget, warnings: tuple[str, ...]) -> None:
        if not warnings:
            return
        SentivisDialog.warning(
            parent,
            "Sentivis AI",
            "Analysis completed with notices",
            "\n".join(warnings[:5]),
        )

    @staticmethod
    def show_export_success(parent: QWidget, path: str) -> None:
        SentivisDialog.information(parent, "Sentivis AI", "Export saved", path)

    @staticmethod
    def confirm_overwrite(parent: QWidget, path: str) -> bool:
        return SentivisDialog.confirm(
            parent,
            "Sentivis AI",
            "File already exists",
            f"Overwrite {path}?",
        )
