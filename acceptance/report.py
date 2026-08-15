"""Generate TEST_REPORT.md from pytest JUnit output."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class TestCaseResult:
    """One executed test case."""

    classname: str
    name: str
    time_seconds: float
    passed: bool
    message: str


@dataclass(frozen=True)
class AcceptanceReport:
    """Aggregated acceptance test results."""

    total: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    cases: tuple[TestCaseResult, ...]

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.total > 0


def parse_junit(path: Path) -> AcceptanceReport:
    """Parse pytest --junitxml output."""
    tree = ET.parse(path)
    root = tree.getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        return AcceptanceReport(0, 0, 0, 0, 0.0, ())

    cases: list[TestCaseResult] = []
    for case in suite.findall("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        time_seconds = float(case.get("time", "0") or 0)
        failure = case.find("failure")
        error = case.find("error")
        skipped_elem = case.find("skipped")
        if failure is not None:
            cases.append(
                TestCaseResult(classname, name, time_seconds, False, failure.get("message", failure.text or ""))
            )
        elif error is not None:
            cases.append(
                TestCaseResult(classname, name, time_seconds, False, error.get("message", error.text or ""))
            )
        elif skipped_elem is not None:
            cases.append(TestCaseResult(classname, name, time_seconds, True, "skipped"))
        else:
            cases.append(TestCaseResult(classname, name, time_seconds, True, ""))

    total = int(suite.get("tests", len(cases)))
    failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    passed = total - failed - skipped
    duration = float(suite.get("time", "0") or 0)
    return AcceptanceReport(total, passed, failed, skipped, duration, tuple(cases))


def render_markdown(report: AcceptanceReport) -> str:
    """Render acceptance report as markdown."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")  # noqa: UP017
    status = "PASSED" if report.success else "FAILED"
    lines = [
        "# Sentivis AI — Acceptance Test Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Status:** {status}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total tests | {report.total} |",
        f"| Passed | {report.passed} |",
        f"| Failed | {report.failed} |",
        f"| Skipped | {report.skipped} |",
        f"| Duration | {report.duration_seconds:.2f}s |",
        "",
        "## Results by Category",
        "",
    ]

    categories: dict[str, list[TestCaseResult]] = {}
    for case in report.cases:
        category = case.classname.split(".")[-1] if case.classname else "general"
        categories.setdefault(category, []).append(case)

    for category, cases in sorted(categories.items()):
        passed_count = sum(1 for case in cases if case.passed and case.message != "skipped")
        lines.append(f"### {category} ({passed_count}/{len(cases)} passed)")
        lines.append("")
        lines.append("| Test | Result | Time (s) |")
        lines.append("|------|--------|----------|")
        for case in cases:
            if case.message == "skipped":
                result = "SKIP"
            elif case.passed:
                result = "PASS"
            else:
                result = "FAIL"
            lines.append(f"| `{case.name}` | {result} | {case.time_seconds:.3f} |")
        lines.append("")

    failures = [case for case in report.cases if not case.passed and case.message != "skipped"]
    if failures:
        lines.extend(["## Failures", ""])
        for case in failures:
            lines.append(f"### `{case.classname}.{case.name}`")
            lines.append("")
            lines.append("```")
            lines.append(case.message.strip() or "No message")
            lines.append("```")
            lines.append("")

    lines.extend(
        [
            "## Manual Checklist",
            "",
            "See [TEST_CHECKLIST.md](../TEST_CHECKLIST.md) for manual verification steps not covered by automation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(junit_path: Path, output_path: Path) -> AcceptanceReport:
    """Parse JUnit XML and write markdown report."""
    report = parse_junit(junit_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(report), encoding="utf-8")
    return report
