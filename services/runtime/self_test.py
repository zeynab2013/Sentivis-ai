"""Runtime self-test and health scoring."""

from __future__ import annotations

from dataclasses import dataclass

from core.config.app_config import AppConfig
from core.logging import get_logger
from services.plugins.plugin_registry import PluginRegistry
from services.runtime.assets import RuntimeAssetBundle
from services.runtime.cache_maintenance import CacheMaintenanceService
from services.runtime.model_registry import CentralModelRegistry

logger = get_logger(__name__)


@dataclass(frozen=True)
class SelfTestCheck:
    """One self-test check result."""

    name: str
    passed: bool
    detail: str
    weight: int


@dataclass(frozen=True)
class SelfTestReport:
    """Aggregated runtime self-test outcome."""

    checks: tuple[SelfTestCheck, ...]
    health_score: int

    @property
    def passed(self) -> bool:
        return self.health_score >= 70

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(check.detail for check in self.checks if not check.passed)


class SelfTestRunner:
    """Verifies configuration, models, assets, logging, plugins, and permissions."""

    def __init__(
        self,
        app_config: AppConfig,
        model_registry: CentralModelRegistry,
        assets: RuntimeAssetBundle,
        plugin_registry: PluginRegistry,
        cache_maintenance: CacheMaintenanceService,
    ) -> None:
        self._app_config = app_config
        self._model_registry = model_registry
        self._assets = assets
        self._plugin_registry = plugin_registry
        self._cache_maintenance = cache_maintenance

    def run(self) -> SelfTestReport:
        checks: list[SelfTestCheck] = []
        checks.append(self._check_configuration())
        checks.extend(self._check_models())
        checks.extend(self._check_assets())
        checks.append(self._check_logging())
        checks.append(self._check_plugins())
        checks.append(self._check_permissions())
        score = self._score(checks)
        logger.info("Runtime self-test completed with health score %d", score)
        return SelfTestReport(checks=tuple(checks), health_score=score)

    def _check_configuration(self) -> SelfTestCheck:
        warnings = self._assets.configuration.verify()
        passed = not any("Missing configuration file" in item for item in warnings)
        detail = warnings[0] if warnings else "Configuration files present"
        return SelfTestCheck("configuration", passed, detail, weight=20)

    def _check_models(self) -> list[SelfTestCheck]:
        checks: list[SelfTestCheck] = []
        for record in self._model_registry.records:
            passed = record.runtime_status.value not in {"missing", "validation_failed", "unavailable"}
            detail = record.validation_detail or record.runtime_status.value
            checks.append(
                SelfTestCheck(
                    f"model:{record.kind.name.lower()}",
                    passed,
                    detail,
                    weight=15,
                )
            )
        return checks

    def _check_assets(self) -> list[SelfTestCheck]:
        checks: list[SelfTestCheck] = []
        for manager in self._assets.all_managers():
            warnings = manager.verify()
            blocking = tuple(
                warning
                for warning in warnings
                if "not writable" in warning.lower() or "missing:" in warning.lower()
            )
            checks.append(
                SelfTestCheck(
                    f"assets:{manager.category.value}",
                    not blocking,
                    blocking[0] if blocking else (warnings[0] if warnings else "OK"),
                    weight=5,
                )
            )
        return checks

    def _check_logging(self) -> SelfTestCheck:
        import logging

        root = logging.getLogger("sentivis")
        passed = bool(root.handlers) or self._app_config.logging.console_enabled
        detail = f"{len(root.handlers)} logging handlers configured"
        return SelfTestCheck("logging", passed, detail, weight=10)

    def _check_plugins(self) -> SelfTestCheck:
        plugins = self._plugin_registry.list_plugins()
        passed = len(plugins) >= 3
        detail = f"{len(plugins)} plugins registered"
        return SelfTestCheck("plugins", passed, detail, weight=10)

    def _check_permissions(self) -> SelfTestCheck:
        critical = {"cache", "logs", "models", "temporary"}
        writable = all(
            item.writable
            for item in self._assets.inventory()
            if item.category.value in critical
        )
        detail = "Runtime directories writable" if writable else "One or more runtime directories are not writable"
        return SelfTestCheck("permissions", writable, detail, weight=10)

    @staticmethod
    def _score(checks: list[SelfTestCheck]) -> int:
        total_weight = sum(check.weight for check in checks)
        earned = sum(check.weight for check in checks if check.passed)
        if total_weight == 0:
            return 0
        return round((earned / total_weight) * 100)
