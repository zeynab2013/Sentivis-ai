"""Shared fixtures for acceptance tests."""

from __future__ import annotations

import os

os.environ.setdefault("SENTIVIS_TEST_MODE", "1")

from collections.abc import Generator
from pathlib import Path
from typing import cast

import pytest
from PySide6.QtWidgets import QApplication

from tests.acceptance.support.harness import AcceptanceApp, build_acceptance_app, shutdown_acceptance_app
from tests.acceptance.support.images import create_standard_image


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Single QApplication instance for the acceptance session."""
    app = QApplication.instance()
    if app is None:
        return QApplication([])
    return cast(QApplication, app)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Valid PNG image for acceptance workflows."""
    return create_standard_image(tmp_path / "acceptance_sample.png")


@pytest.fixture
def acceptance_app(qapp: QApplication) -> Generator[AcceptanceApp, None, None]:
    """Desktop application with stub pipeline models."""
    app = build_acceptance_app(stub_pipeline=True)
    yield app
    shutdown_acceptance_app(app)
