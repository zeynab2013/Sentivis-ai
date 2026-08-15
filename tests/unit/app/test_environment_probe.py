"""Unit tests for environment probing."""

from app.startup.environment_probe import probe_environment
from core.utils.paths import project_root


def test_environment_probe_succeeds_on_developer_machine() -> None:
    root = project_root()
    report = probe_environment(
        project_root=root,
        models_dir=root / "models",
        config_paths=(
            root / "config" / "app.default.toml",
            root / "config" / "models.default.toml",
            root / "config" / "analysis.default.toml",
            root / "config" / "themes.default.toml",
        ),
    )
    assert report.python_version
    assert report.operating_system
    assert report.config_files_ok
    assert report.models_dir_writable
    assert report.temp_dir_writable
