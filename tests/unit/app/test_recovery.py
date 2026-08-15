"""Unit tests for startup recovery messages."""

from app.startup.recovery import recovery_message


def test_python_version_recovery_message() -> None:
    message = recovery_message("Python 3.10.x required (tested on 3.10.11); found 3.9")
    assert "Python 3.10.11" in message
