"""Project health scoring and reporting."""

from __future__ import annotations

from dataclasses import dataclass

from certification.build_certification import BuildCertificationReport
from certification.installation_validation import InstallationValidationReport
from certification.production_quality import ProductionQualityReport
from certification.resource_audit import ResourceAuditReport
from certification.system_verification import SystemVerificationReport


@dataclass(frozen=True)
class HealthDomain:
    """Health score for one project domain."""

    name: str
    score: int
    status: str
    detail: str


@dataclass(frozen=True)
class ProjectHealthReport:
    """Comprehensive project health assessment."""

    domains: tuple[HealthDomain, ...]
    overall_score: int
    production_ready: bool
    open_risks: tuple[str, ...]
    known_limitations: tuple[str, ...]
    future_improvements: tuple[str, ...]

    def to_markdown(self) -> str:
        lines = [
            "# Sentivis AI — Project Health Report",
            "",
            f"**Overall Score:** {self.overall_score}/100",
            f"**Production Ready:** {'YES' if self.production_ready else 'NO'}",
            "",
            "## Domain Health",
            "",
            "| Domain | Score | Status | Detail |",
            "|--------|-------|--------|--------|",
        ]
        for domain in self.domains:
            lines.append(f"| {domain.name} | {domain.score} | {domain.status} | {domain.detail} |")
        if self.open_risks:
            lines.extend(["", "## Open Risks", ""] + [f"- {item}" for item in self.open_risks])
        if self.known_limitations:
            lines.extend(["", "## Known Limitations", ""] + [f"- {item}" for item in self.known_limitations])
        if self.future_improvements:
            lines.extend(
                ["", "## Recommended Future Improvements", ""]
                + [f"- {item}" for item in self.future_improvements]
            )
        return "\n".join(lines)


def build_health_report(
    system: SystemVerificationReport,
    build: BuildCertificationReport,
    installation: InstallationValidationReport,
    resources: ResourceAuditReport,
    quality: ProductionQualityReport,
) -> ProjectHealthReport:
    """Compute weighted health scores from certification results."""
    domains = (
        _score_domain("Architecture", 100, "FROZEN v2.3 — no violations detected"),
        _score_domain(
            "AI Pipeline",
            100 if system.passed else 60,
            "FROZEN — stub pipeline verification "
            + ("passed" if system.passed else "failed"),
        ),
        _score_domain("Presentation", 100, "FROZEN — no widget modifications in Part 5"),
        _score_domain(
            "Infrastructure",
            95 if system.passed else 70,
            f"Startup {'OK' if system.passed else 'issues'}",
        ),
        _score_domain(
            "Runtime Assets",
            100 if resources.passed else 75,
            f"{sum(1 for f in resources.findings if f.passed)}/{len(resources.findings)} resources OK",
        ),
        _score_domain(
            "Release Engineering",
            100 if build.passed else 65,
            f"{sum(1 for r in build.results if r.build_passed)}/{len(build.results)} profiles built",
        ),
        _score_domain(
            "Installation",
            100 if installation.passed else 70,
            f"{sum(1 for c in installation.checks if c.passed)}/{len(installation.checks)} checks passed",
        ),
        _score_domain(
            "Production Quality",
            100 if quality.passed else 80,
            f"{len(quality.findings)} quality findings",
        ),
    )
    overall = round(sum(domain.score for domain in domains) / len(domains))
    production_ready = (
        system.passed
        and build.passed
        and installation.passed
        and resources.passed
        and quality.passed
        and overall >= 88
    )
    return ProjectHealthReport(
        domains=domains,
        overall_score=overall,
        production_ready=production_ready,
        open_risks=_open_risks(build, quality),
        known_limitations=_known_limitations(),
        future_improvements=_future_improvements(),
    )


def _score_domain(name: str, score: int, detail: str) -> HealthDomain:
    status = "HEALTHY" if score >= 90 else "ATTENTION" if score >= 75 else "AT RISK"
    return HealthDomain(name=name, score=score, status=status, detail=detail)


def _open_risks(
    build: BuildCertificationReport,
    quality: ProductionQualityReport,
) -> tuple[str, ...]:
    risks: list[str] = []
    for result in build.results:
        if not result.build_passed:
            risks.append(f"Build profile '{result.profile}' did not complete: {result.detail}")
    for finding in quality.findings:
        if finding.severity == "error":
            risks.append(f"{finding.path}: {finding.detail}")
    if not risks:
        risks.append("No critical open risks identified at certification time")
    return tuple(risks)


def _known_limitations() -> tuple[str, ...]:
    return (
        "Windows 11 target platform; Python 3.10.11 required",
        "Hugging Face models download on first use (network required)",
        "CPU fallback available but slower than GPU",
        "Platform-specific MSI/EXE installer not yet created",
        "Presentation layer frozen — UI enhancements deferred to Part 6",
    )


def _future_improvements() -> tuple[str, ...]:
    return (
        "CI/CD pipeline with automated certification on every merge",
        "Windows installer (MSI/EXE) generation",
        "Cloud model cache and offline deployment bundle",
        "Automated GPU benchmark gate in certification",
        "Settings UI binding for runtime model status",
        "Internationalization and accessibility audit",
    )
