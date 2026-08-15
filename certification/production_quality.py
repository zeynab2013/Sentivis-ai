"""Production quality scanning."""

from __future__ import annotations

import ast
import importlib
import re
from dataclasses import dataclass
from pathlib import Path

from core.utils.paths import project_root


@dataclass(frozen=True)
class QualityFinding:
    """One production quality issue."""

    severity: str
    path: str
    detail: str


@dataclass(frozen=True)
class ProductionQualityReport:
    """Scan results for production readiness."""

    findings: tuple[QualityFinding, ...]
    removed_artifacts: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not any(item.severity == "error" for item in self.findings)


class ProductionQualityScanner:
    """Detects task markers, missing packages, and critical runtime gaps."""

    _SCAN_DIRS = (
        "app",
        "core",
        "vision",
        "language",
        "analysis",
        "services",
        "ui",
        "streamlit_app",
        "model_management",
        "release",
        "certification",
        "acceptance",
    )
    _REQUIRED_PACKAGES = (
        "app",
        "core",
        "vision",
        "language",
        "analysis",
        "services",
        "ui",
        "streamlit_app",
        "model_management",
        "release",
    )
    _REQUIRED_IMPORTS = (
        "app.main",
        "core.config.loader",
        "core.resources",
        "streamlit_app.main",
        "analysis.pose.pose_estimator",
        "analysis.ocr.text_extractor",
        "services.pipeline.orchestrator",
        "language.refinement.caption_refiner",
        "release.builder",
        "model_management",
    )
    _REQUIRED_DATA = (
        "config/app.default.toml",
        "config/models.default.toml",
        "config/analysis.default.toml",
        "translations/en.json",
        "core/resources/translations/en.json",
        "streamlit_app/main.py",
    )
    _SKIP_PARTS = {".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "__pycache__"}
    # Built without embedding literal marker tokens in this source file.
    _TASK_MARKER = re.compile(
        r"\b("
        + "|".join(
            (
                "TO" + "DO",
                "FIX" + "ME",
                "X" * 3,
                "HA" + "CK",
            )
        )
        + r")\b",
        re.IGNORECASE,
    )
    _DEBUG_PRINT = re.compile(r"\bprint\s*\(")

    _REMOVABLE_ARTIFACTS = (
        "logs/application.log",
        "logs/pipeline.log",
        "logs/error.log",
        "logs/benchmark.log",
        "logs/startup-diagnostics.json",
        "logs/startup-diagnostics.txt",
        "cert_tmp",
        "cert_exports",
    )

    def scan(self, root: Path | None = None, *, cleanup: bool = True) -> ProductionQualityReport:
        project = root or project_root()
        findings: list[QualityFinding] = []
        removed: list[str] = []

        findings.extend(self._check_required_layout(project))
        findings.extend(self._check_imports())
        findings.extend(self._check_translations_loadable())
        findings.extend(self._check_language_streamlit_leak(project))

        for package in self._SCAN_DIRS:
            package_path = project / package
            if not package_path.is_dir():
                findings.append(
                    QualityFinding("error", package, "Required package directory missing")
                )
                continue
            for path in package_path.rglob("*.py"):
                if any(part in self._SKIP_PARTS for part in path.parts):
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError as exc:
                    findings.append(
                        QualityFinding("warning", str(path.relative_to(project)), f"Unreadable: {exc}")
                    )
                    continue
                if self._TASK_MARKER.search(text):
                    findings.append(
                        QualityFinding("error", str(path.relative_to(project)), "Task marker found")
                    )
                if (
                    self._DEBUG_PRINT.search(text)
                    and "if __name__" not in text
                    and path.name not in {"__main__.py"}
                ):
                    findings.append(
                        QualityFinding(
                            "warning",
                            str(path.relative_to(project)),
                            "print() statement in production code",
                        )
                    )
                # Syntax gate for scanned modules.
                try:
                    ast.parse(text, filename=str(path))
                except SyntaxError as exc:
                    findings.append(
                        QualityFinding(
                            "error",
                            str(path.relative_to(project)),
                            f"SyntaxError: {exc.msg} (line {exc.lineno})",
                        )
                    )

        for relative in self._REMOVABLE_ARTIFACTS:
            target = project / relative
            if target.exists() and cleanup:
                try:
                    if target.is_dir():
                        import shutil

                        shutil.rmtree(target, ignore_errors=True)
                    else:
                        target.unlink(missing_ok=True)
                    removed.append(relative)
                except OSError:
                    continue

        return ProductionQualityReport(findings=tuple(findings), removed_artifacts=tuple(removed))

    def _check_required_layout(self, project: Path) -> list[QualityFinding]:
        findings: list[QualityFinding] = []
        for package in self._REQUIRED_PACKAGES:
            if not (project / package).is_dir():
                findings.append(QualityFinding("error", package, "Required package missing"))
        for relative in self._REQUIRED_DATA:
            if not (project / relative).is_file():
                findings.append(QualityFinding("error", relative, "Required runtime file missing"))
        return findings

    def _check_imports(self) -> list[QualityFinding]:
        findings: list[QualityFinding] = []
        for module_name in self._REQUIRED_IMPORTS:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 — import health is the check
                findings.append(
                    QualityFinding(
                        "error",
                        module_name,
                        f"Import failed: {type(exc).__name__}: {exc}",
                    )
                )
        return findings

    def _check_translations_loadable(self) -> list[QualityFinding]:
        findings: list[QualityFinding] = []
        try:
            from core.resources import load_translation_catalog

            catalog = load_translation_catalog("en")
            if not catalog:
                findings.append(
                    QualityFinding("error", "translations/en.json", "English catalog empty/unreadable")
                )
        except Exception as exc:  # noqa: BLE001
            findings.append(
                QualityFinding("error", "core.resources", f"Translation loader failed: {exc}")
            )
        return findings

    def _check_language_streamlit_leak(self, project: Path) -> list[QualityFinding]:
        """Domain packages must not import streamlit_app (architecture boundary)."""
        findings: list[QualityFinding] = []
        pattern = re.compile(r"^\s*(from|import)\s+streamlit_app\b", re.MULTILINE)
        for package in ("language", "core", "analysis", "vision", "services"):
            root = project / package
            if not root.is_dir():
                continue
            for path in root.rglob("*.py"):
                if any(part in self._SKIP_PARTS for part in path.parts):
                    continue
                text = path.read_text(encoding="utf-8")
                if pattern.search(text):
                    findings.append(
                        QualityFinding(
                            "error",
                            str(path.relative_to(project)),
                            "Domain package imports streamlit_app (architecture leak)",
                        )
                    )
        return findings
