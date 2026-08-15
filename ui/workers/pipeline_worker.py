"""Pipeline worker running off the UI thread."""

from PySide6.QtCore import QThread, Signal

from core.contracts.pipeline import PipelineRequest
from core.exceptions.base import SentivisError
from core.exceptions.service import CancelledError
from core.logging import get_logger
from services.interfaces.pipeline import IPipelineOrchestrator

logger = get_logger(__name__)


class PipelineWorker(QThread):
    """Executes pipeline analysis on a background thread."""

    progress = Signal(object)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, orchestrator: IPipelineOrchestrator, request: PipelineRequest) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._request = request

    def run(self) -> None:
        try:
            result = self._orchestrator.analyze(self._request)
            self.finished_ok.emit(result)
        except CancelledError as exc:
            logger.info("Pipeline cancelled: %s", exc.developer_detail)
            self.failed.emit(exc.user_message)
        except SentivisError as exc:
            logger.error("Pipeline failed: %s", exc.developer_detail)
            self.failed.emit(exc.user_message)
        except Exception as exc:
            logger.exception("Unexpected pipeline failure: %s", exc)
            self.failed.emit("Analysis failed unexpectedly.")
