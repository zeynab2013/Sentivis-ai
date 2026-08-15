"""Dedicated runtime asset managers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.config.app_config import AppConfig
from core.config.user_config_paths import user_config_dir
from core.logging import get_logger
from core.utils.paths import project_root, resource_path

logger = get_logger(__name__)


class AssetCategory(str, Enum):  # noqa: UP042
    """Runtime asset categories managed by dedicated managers."""

    MODELS = "models"
    ICONS = "icons"
    THEMES = "themes"
    CONFIGURATION = "configuration"
    SAMPLES = "samples"
    EXPORT_TEMPLATES = "export_templates"
    LOGS = "logs"
    CACHE = "cache"
    TEMPORARY = "temporary"


@dataclass(frozen=True)
class AssetInventory:
    """Inventory summary for one asset category."""

    category: AssetCategory
    root: Path
    file_count: int
    total_bytes: int
    writable: bool
    detail: str


class BaseAssetManager(ABC):
    """Base class for category-specific asset managers."""

    category: AssetCategory

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def ensure_directory(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def inventory(self) -> AssetInventory:
        self.ensure_directory()
        files = [path for path in self._root.rglob("*") if path.is_file()]
        total_bytes = sum(path.stat().st_size for path in files if path.exists())
        writable = self._is_writable()
        return AssetInventory(
            category=self.category,
            root=self._root,
            file_count=len(files),
            total_bytes=total_bytes,
            writable=writable,
            detail=f"{len(files)} files under {self._root.name}",
        )

    @abstractmethod
    def verify(self) -> tuple[str, ...]:
        """Return warnings for this asset category."""

    def _is_writable(self) -> bool:
        try:
            probe = self.ensure_directory() / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except OSError:
            return False


class ModelsAssetManager(BaseAssetManager):
    category = AssetCategory.MODELS

    def verify(self) -> tuple[str, ...]:
        warnings: list[str] = []
        if not self._root.exists():
            warnings.append(f"Models directory does not exist: {self._root}")
        elif not self._is_writable():
            warnings.append(f"Models directory is not writable: {self._root}")
        return tuple(warnings)


class IconsAssetManager(BaseAssetManager):
    category = AssetCategory.ICONS

    def verify(self) -> tuple[str, ...]:
        if not self._root.exists():
            return (f"Icons directory will be created on demand: {self._root}",)
        return ()


class ThemesAssetManager(BaseAssetManager):
    category = AssetCategory.THEMES

    def verify(self) -> tuple[str, ...]:
        if not self._root.exists():
            return (f"Themes directory missing: {self._root}",)
        return ()


class ConfigurationAssetManager(BaseAssetManager):
    category = AssetCategory.CONFIGURATION

    def verify(self) -> tuple[str, ...]:
        warnings: list[str] = []
        root = project_root()
        required = (
            root / "config" / "app.default.toml",
            root / "config" / "models.default.toml",
            root / "config" / "analysis.default.toml",
            root / "config" / "themes.default.toml",
        )
        for path in required:
            if not path.is_file():
                warnings.append(f"Missing configuration file: {path.name}")
        if not user_config_dir().exists():
            warnings.append("User configuration directory has not been created yet")
        return tuple(warnings)


class SamplesAssetManager(BaseAssetManager):
    category = AssetCategory.SAMPLES

    def verify(self) -> tuple[str, ...]:
        if not any(self._root.glob("*")):
            return ("No sample images bundled yet",)
        return ()


class ExportTemplatesAssetManager(BaseAssetManager):
    category = AssetCategory.EXPORT_TEMPLATES

    def verify(self) -> tuple[str, ...]:
        if not self._root.exists():
            return ("Export templates directory will be created on demand",)
        return ()


class LogsAssetManager(BaseAssetManager):
    category = AssetCategory.LOGS

    def verify(self) -> tuple[str, ...]:
        if not self._is_writable():
            return (f"Logs directory is not writable: {self._root}",)
        return ()


class CacheAssetManager(BaseAssetManager):
    category = AssetCategory.CACHE

    def verify(self) -> tuple[str, ...]:
        if not self._is_writable():
            return (f"Cache directory is not writable: {self._root}",)
        return ()


class TemporaryAssetManager(BaseAssetManager):
    category = AssetCategory.TEMPORARY

    def verify(self) -> tuple[str, ...]:
        if not self._is_writable():
            return (f"Temporary directory is not writable: {self._root}",)
        return ()


@dataclass(frozen=True)
class RuntimeAssetBundle:
    """All dedicated runtime asset managers."""

    models: ModelsAssetManager
    icons: IconsAssetManager
    themes: ThemesAssetManager
    configuration: ConfigurationAssetManager
    samples: SamplesAssetManager
    export_templates: ExportTemplatesAssetManager
    logs: LogsAssetManager
    cache: CacheAssetManager
    temporary: TemporaryAssetManager

    def all_managers(self) -> tuple[BaseAssetManager, ...]:
        return (
            self.models,
            self.icons,
            self.themes,
            self.configuration,
            self.samples,
            self.export_templates,
            self.logs,
            self.cache,
            self.temporary,
        )

    def inventory(self) -> tuple[AssetInventory, ...]:
        return tuple(manager.inventory() for manager in self.all_managers())


def build_runtime_assets(app_config: AppConfig) -> RuntimeAssetBundle:
    """Construct asset managers from application configuration."""
    root = project_root()
    temp_root = Path(app_config.paths.cache_dir.parent / "tmp")
    return RuntimeAssetBundle(
        models=ModelsAssetManager(app_config.paths.models_dir),
        icons=IconsAssetManager(resource_path("icons")),
        themes=ThemesAssetManager(root / "ui" / "themes"),
        configuration=ConfigurationAssetManager(user_config_dir()),
        samples=SamplesAssetManager(resource_path("samples")),
        export_templates=ExportTemplatesAssetManager(resource_path("export_templates")),
        logs=LogsAssetManager(app_config.paths.logs_dir),
        cache=CacheAssetManager(app_config.paths.cache_dir),
        temporary=TemporaryAssetManager(temp_root),
    )
