"""Application entry point — launches Streamlit UI with visible CMD logs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from app.console_bootstrap import (
        prepare_windows_console,
        print_runtime_diagnostics,
        print_startup_banner,
    )

    prepare_windows_console()

    app_path = root / "streamlit_app" / "main.py"
    port = int(os.environ.get("SENTIVIS_PORT", os.environ.get("STREAMLIT_SERVER_PORT", "8501")))

    print_startup_banner(root=root, app_path=app_path, port=port)
    try:
        print_runtime_diagnostics(root=root)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not print runtime diagnostics: {exc}", flush=True)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "dark",
        "--logger.level",
        "info",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Ensure Streamlit child does not hide Python logging from this console.
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    try:
        completed = subprocess.run(cmd, cwd=str(root), env=env, check=False)
    except KeyboardInterrupt:
        print("\nSentivis AI interrupted by user.", flush=True)
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to start Streamlit: {exc}", flush=True)
        raise SystemExit(1) from exc

    if completed.returncode != 0:
        print(
            f"ERROR: Streamlit exited with code {completed.returncode}. "
            f"See logs under {root / 'logs'}.",
            flush=True,
        )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
