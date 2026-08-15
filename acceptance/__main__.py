"""Run acceptance tests and generate TEST_REPORT.md."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from acceptance.report import write_report


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    junit_path = root / "acceptance_results.xml"
    report_path = root / "docs" / "TEST_REPORT.md"

    print("Running Sentivis AI acceptance test suite…")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/acceptance",
            "-v",
            "--tb=short",
            f"--junitxml={junit_path}",
        ],
        cwd=root,
        check=False,
    )

    if junit_path.is_file():
        report = write_report(junit_path, report_path)
        print(f"Report written to {report_path}")
        print(f"Results: {report.passed}/{report.total} passed, {report.failed} failed")
        if not report.success:
            return 1
    else:
        print("JUnit results not found; report not generated.", file=sys.stderr)
        return result.returncode or 1

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
