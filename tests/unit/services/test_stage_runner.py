"""Unit tests for stage runner reliability."""

import pytest

from core.config.loader import load_app_config
from core.constants.pipeline_stages import PipelineStage
from core.exceptions.language import InferenceError
from core.exceptions.service import CancelledError
from services.memory.memory_manager import MemoryManager
from services.pipeline.cancellation import CancellationToken
from services.pipeline.progress_reporter import ProgressReporter
from services.pipeline.stage_runner import StageRunner


def test_stage_runner_propagates_cancellation() -> None:
    token = CancellationToken()
    token.cancel()
    runner = StageRunner(ProgressReporter(), token)
    with pytest.raises(CancelledError):
        runner.run(PipelineStage.VALIDATION, 1.0, "test", lambda: "ok")


def test_stage_runner_retries_recoverable_gpu_failure() -> None:
    attempts = {"count": 0}
    memory_manager = MemoryManager(load_app_config())

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise InferenceError(
                "Visual understanding failed during analysis.",
                "simulated gpu failure",
                stage=PipelineStage.BLIP_UNDERSTANDING,
                recoverable=True,
            )
        return "ok"

    runner = StageRunner(ProgressReporter(), CancellationToken(), memory_manager)
    assert runner.run(PipelineStage.BLIP_UNDERSTANDING, 50.0, "test", flaky) == "ok"
    assert attempts["count"] == 2
