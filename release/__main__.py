"""CLI entry point for release builds."""

from __future__ import annotations

import argparse
import sys

from release.build_profiles import get_profile
from release.builder import BuildError, available_profiles, build_profile
from release.metadata import load_release_info
from release.validator import BuildValidator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Sentivis AI release artifacts")
    parser.add_argument(
        "profile",
        choices=available_profiles(),
        help="Build profile to produce",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run build validation without creating artifacts",
    )
    args = parser.parse_args(argv)

    profile = get_profile(args.profile)
    release_info = load_release_info(build_profile=profile.name.value)
    report = BuildValidator().validate(profile, release_info)

    for warning in report.warnings:
        print(f"WARNING: {warning}")
    if not report.passed:
        for error in report.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Build validation passed.")
    if args.validate_only:
        return 0

    try:
        output = build_profile(args.profile)
    except BuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Build complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
