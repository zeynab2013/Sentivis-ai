"""Layered configuration loading for application startup."""

from __future__ import annotations

from dataclasses import dataclass

from core.config.analysis_config import AnalysisConfig
from core.config.app_config import AppConfig
from core.config.config_sources import ConfigSource, LoadedConfiguration
from core.config.layered_loader import load_layered_toml
from core.config.loader import (
    _build_analysis_config,
    _build_app_config,
    _build_model_config,
    _build_theme_config,
)
from core.config.model_config import ModelConfig
from core.config.schema_validator import (
    validate_analysis_config,
    validate_app_config,
    validate_model_config,
    validate_theme_config,
)
from core.config.theme_config import ThemeConfig
from core.config.user_config_paths import user_config_dir, user_override_path
from core.utils.paths import project_root


@dataclass(frozen=True)
class ApplicationSettings:
    """All configuration loaded with source metadata."""

    app_config: AppConfig
    model_config: ModelConfig
    analysis_config: AnalysisConfig
    theme_config: ThemeConfig
    sources: LoadedConfiguration


def load_application_settings() -> ApplicationSettings:
    """Load layered configuration: defaults → user overrides → validation."""
    root = project_root()
    user_dir = user_config_dir()
    user_dir.mkdir(parents=True, exist_ok=True)

    layers: list[ConfigSource] = []

    app_default = root / "config" / "app.default.toml"
    app_user = user_override_path("app.toml")
    app_data = load_layered_toml(app_default, app_user if app_user.is_file() else None)
    validate_app_config(app_data)
    layers.append(ConfigSource("app", str(app_default), "default"))
    if app_user.is_file():
        layers.append(ConfigSource("app", str(app_user), "user"))

    model_default = root / "config" / "models.default.toml"
    model_user = user_override_path("models.toml")
    model_data = load_layered_toml(model_default, model_user if model_user.is_file() else None)
    validate_model_config(model_data)
    layers.append(ConfigSource("models", str(model_default), "default"))
    if model_user.is_file():
        layers.append(ConfigSource("models", str(model_user), "user"))

    analysis_default = root / "config" / "analysis.default.toml"
    analysis_user = user_override_path("analysis.toml")
    analysis_data = load_layered_toml(analysis_default, analysis_user if analysis_user.is_file() else None)
    validate_analysis_config(analysis_data)
    layers.append(ConfigSource("analysis", str(analysis_default), "default"))
    if analysis_user.is_file():
        layers.append(ConfigSource("analysis", str(analysis_user), "user"))

    theme_default = root / "config" / "themes.default.toml"
    theme_user = user_override_path("themes.toml")
    theme_data = load_layered_toml(theme_default, theme_user if theme_user.is_file() else None)
    validate_theme_config(theme_data)
    layers.append(ConfigSource("theme", str(theme_default), "default"))
    if theme_user.is_file():
        layers.append(ConfigSource("theme", str(theme_user), "user"))

    return ApplicationSettings(
        app_config=_build_app_config(app_data),
        model_config=_build_model_config(model_data),
        analysis_config=_build_analysis_config(analysis_data),
        theme_config=_build_theme_config(theme_data),
        sources=LoadedConfiguration(tuple(layers)),
    )
