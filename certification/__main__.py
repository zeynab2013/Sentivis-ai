"""CLI entry point for production certification."""

from __future__ import annotations

import argparse
import sys

from certification.certifier import ProductionCertifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Certify Sentivis AI for production deployment")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building all release profiles (validation only for builds)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not remove development log artifacts",
    )
    args = parser.parse_args(argv)

    certifier = ProductionCertifier()
    report = certifier.certify(build=not args.skip_build, cleanup=not args.no_cleanup)
    json_path, health_path = certifier.write_reports(report)

    print(f"Certification {'PASSED' if report.passed else 'FAILED'}")
    print(f"Overall score: {report.health.overall_score}/100")
    print(f"Reports: {json_path.name}, {health_path.name}")

    if not report.passed:
        if not report.system.passed:
            print("System verification failed", file=sys.stderr)
        if not report.build.passed:
            print("Build certification failed", file=sys.stderr)
        if not report.installation.passed:
            print("Installation validation failed", file=sys.stderr)
        if not report.resources.passed:
            print("Resource audit failed", file=sys.stderr)
        if not report.quality.passed:
            print("Production quality scan failed", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
