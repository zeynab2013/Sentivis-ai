"""Release builder includes runtime packages and translations."""

from __future__ import annotations

from pathlib import Path

from release.builder import ReleaseBuilder


def test_release_package_dirs_include_streamlit_and_models() -> None:
    packages = set(ReleaseBuilder._PACKAGE_DIRS)  # noqa: SLF001
    for required in (
        "app",
        "core",
        "streamlit_app",
        "model_management",
        "language",
        "vision",
        "analysis",
        "services",
        "ui",
        "release",
    ):
        assert required in packages


def test_release_builder_copies_translations(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[3]
    builder = ReleaseBuilder(root=root)
    # Build development profile into an isolated dist by monkeypatching dist root.
    builder._dist_root = tmp_path / "dist"  # noqa: SLF001
    output = builder.build("development")
    assert (output / "streamlit_app" / "main.py").is_file()
    assert (output / "model_management").is_dir()
    assert (output / "translations" / "en.json").is_file()
    assert (output / "config" / "app.default.toml").is_file()
    assert (output / "core" / "resources" / "translations" / "en.json").is_file()
