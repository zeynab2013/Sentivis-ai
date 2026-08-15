"""Gate UI/presentation changes against perception unit regressions.

Runs focused unit suites that cover caption quality dimensions requested for
competition validation. Exit code 1 on failure (caller should not ship UI).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUITES = [
    "tests/unit/analysis",
    "tests/unit/language",
    "tests/unit/vision",
    "tests/unit/streamlit",
    "tests/unit/ui/test_design_system.py",
]


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", *SUITES]
    print("Running perception + UI gate:")
    print(" ", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        print("GATE FAILED — perception/UI regressions detected. Do not ship.")
        return completed.returncode
    print("GATE PASSED — no perception/UI unit regressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
