"""Runtime resource audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.config.loader import load_app_config
from core.utils.paths import project_root, resource_path
from services.runtime.assets import build_runtime_assets


@dataclass(frozen=True)
class ResourceFinding:
    """One resource audit finding."""

    category: str
    path: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ResourceAuditReport:
    """Complete runtime resource audit."""

    findings: tuple[ResourceFinding, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.findings)


class ResourceAuditor:
    """Audits icons, themes, templates, config, docs, and bundled assets."""

    _REQUIRED_DOCS = (
        "INSTALLATION_GUIDE.md",
        "QUICK_START_GUIDE.md",
        "USER_MANUAL.md",
        "TROUBLESHOOTING_GUIDE.md",
        "RELEASE_NOTES.md",
        "SYSTEM_REQUIREMENTS.md",
        "KNOWN_LIMITATIONS.md",
        "DIRECTORY_STRUCTURE.md",
    )

    def audit(self, root: Path | None = None) -> ResourceAuditReport:
        project = root or project_root()
        findings: list[ResourceFinding] = []

        icon = resource_path("icons", "app_icon.svg")
        findings.append(
            ResourceFinding(
                "icons",
                str(icon),
                icon.is_file(),
                "Application icon present" if icon.is_file() else "Missing app_icon.svg",
            )
        )

        theme_engine = project / "ui" / "themes" / "theme_engine.py"
        findings.append(
            ResourceFinding(
                "themes",
                str(theme_engine),
                theme_engine.is_file(),
                "Theme engine present",
            )
        )

        for name in ("samples/README.txt", "export_templates/README.txt"):
            path = resource_path(*name.split("/"))
            findings.append(
                ResourceFinding(
                    "templates" if "export" in name else "samples",
                    str(path),
                    path.is_file(),
                    path.name,
                )
            )

        for name in (
            "app.default.toml",
            "models.default.toml",
            "analysis.default.toml",
            "themes.default.toml",
        ):
            path = project / "config" / name
            findings.append(
                ResourceFinding(
                    "configuration",
                    str(path),
                    path.is_file(),
                    name,
                )
            )

        release_resources = (
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "installer_manifest.json",
        )
        for name in release_resources:
            path = project / "release" / "resources" / name
            findings.append(
                ResourceFinding(
                    "release_resources",
                    str(path),
                    path.is_file(),
                    name,
                )
            )

        docs_dir = project / "docs"
        for name in self._REQUIRED_DOCS:
            path = docs_dir / name
            findings.append(
                ResourceFinding(
                    "documentation",
                    str(path),
                    path.is_file(),
                    name,
                )
            )

        assets = build_runtime_assets(load_app_config())
        for manager in assets.all_managers():
            inventory = manager.inventory()
            findings.append(
                ResourceFinding(
                    manager.category.value,
                    str(inventory.root),
                    inventory.writable or manager.category.value in {"icons", "samples", "export_templates"},
                    f"{inventory.file_count} files, writable={inventory.writable}",
                )
            )

        return ResourceAuditReport(findings=tuple(findings))
