"""Production certification orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from certification.build_certification import BuildCertificationReport, BuildCertifier
from certification.health_report import ProjectHealthReport, build_health_report
from certification.installation_validation import InstallationValidationReport, InstallationValidator
from certification.production_quality import ProductionQualityReport, ProductionQualityScanner
from certification.resource_audit import ResourceAuditor, ResourceAuditReport
from certification.system_verification import SystemVerificationReport, SystemVerifier
from core.utils.paths import project_root


@dataclass(frozen=True)
class CertificationReport:
    """Complete production certification outcome."""

    timestamp: str
    system: SystemVerificationReport
    build: BuildCertificationReport
    installation: InstallationValidationReport
    resources: ResourceAuditReport
    quality: ProductionQualityReport
    health: ProjectHealthReport

    @property
    def passed(self) -> bool:
        return self.health.production_ready

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "passed": self.passed,
            "overall_score": self.health.overall_score,
            "system_checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.system.checks
            ],
            "build_profiles": [
                {
                    "profile": r.profile,
                    "validation_passed": r.validation_passed,
                    "build_passed": r.build_passed,
                    "detail": r.detail,
                }
                for r in self.build.results
            ],
            "quality_findings": [
                {"severity": f.severity, "path": f.path, "detail": f.detail}
                for f in self.quality.findings
            ],
            "removed_artifacts": list(self.quality.removed_artifacts),
        }


class ProductionCertifier:
    """Runs full production certification suite."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or project_root()

    def certify(self, *, build: bool = True, cleanup: bool = True) -> CertificationReport:
        workspace = self._root / "cert_tmp"
        workspace.mkdir(parents=True, exist_ok=True)

        system = SystemVerifier().run(workspace)
        build_report = BuildCertifier().certify(build=build)
        install_root = self._root / "cert_install_sim"
        if install_root.exists():
            import shutil

            shutil.rmtree(install_root, ignore_errors=True)
        installation = InstallationValidator().run(install_root)
        quality = ProductionQualityScanner().scan(self._root, cleanup=cleanup)
        resources = ResourceAuditor().audit(self._root)
        health = build_health_report(system, build_report, installation, resources, quality)

        return CertificationReport(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
            system=system,
            build=build_report,
            installation=installation,
            resources=resources,
            quality=quality,
            health=health,
        )

    def write_reports(self, report: CertificationReport, docs_dir: Path | None = None) -> tuple[Path, Path]:
        docs = docs_dir or (self._root / "docs")
        docs.mkdir(parents=True, exist_ok=True)
        json_path = docs / "production_certification.json"
        json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        health_path = docs / "PROJECT_HEALTH_REPORT.md"
        health_path.write_text(report.health.to_markdown(), encoding="utf-8")
        return json_path, health_path
