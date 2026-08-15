"""Tests for production portability and import safety."""

from __future__ import annotations

import importlib.util

from core.utils.paths import ensure_runtime_directories, project_root, runtime_directories
from streamlit_app.backend import StreamlitBackend
from streamlit_app.bootstrap import StreamlitBackend as LegacyBackend
from streamlit_app.diagnostics import build_readiness_report, check_dependencies
from streamlit_app.startup import initialize_backend


def test_project_root_points_at_repo() -> None:
    root = project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "config" / "app.default.toml").is_file()


def test_runtime_directories_created() -> None:
    root = project_root()
    created = ensure_runtime_directories(root)
    assert len(created) == len(runtime_directories(root))
    for directory in runtime_directories(root):
        assert directory.is_dir()


def test_streamlit_entry_does_not_shadow_app_package() -> None:
    root = project_root()
    entry = root / "streamlit_app" / "main.py"
    spec = importlib.util.spec_from_file_location("sentivis_streamlit_probe", entry)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import app.container

    assert app.container.ApplicationContext is not None


def test_bootstrap_reexport_matches_backend() -> None:
    assert StreamlitBackend is LegacyBackend


def test_dependency_checks_return_results() -> None:
    checks = check_dependencies()
    names = {item.name for item in checks}
    assert "Python" in names
    assert "PyTorch" in names
    assert "Streamlit" in names


def test_readiness_report_has_title() -> None:
    report = build_readiness_report()
    assert report.title in {"System Ready", "System Not Ready"}
    assert report.summary


def test_initialize_backend_is_callable() -> None:
    assert callable(initialize_backend)
