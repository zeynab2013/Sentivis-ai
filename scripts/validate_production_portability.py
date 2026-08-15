"""Production portability and import validation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utils.paths import ensure_runtime_directories, project_root  # noqa: E402
from streamlit_app.diagnostics import build_readiness_report, check_dependencies  # noqa: E402


def _simulate_streamlit_module_shadow() -> tuple[bool, str]:
    """Verify ``streamlit_app/main.py`` does not shadow the ``app`` package."""

    script_dir = str((ROOT / "streamlit_app").resolve())
    probe_path = ROOT / "streamlit_app" / "main.py"
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("sentivis_streamlit_entry", probe_path)
    if spec is None or spec.loader is None:
        return False, "Could not load streamlit entry spec"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return False, f"Streamlit entry import failed: {exc}"
    try:
        import app.container  # noqa: F401
    except Exception as exc:
        return False, f"app package shadowing detected: {exc}"
    return True, "Streamlit entry loads without shadowing app package"


def _import_backend_chain() -> tuple[bool, str]:
    try:
        from streamlit_app.backend import StreamlitBackend  # noqa: F401
        from streamlit_app.bootstrap import StreamlitBackend as LegacyBackend  # noqa: F401
        from streamlit_app.startup import initialize_backend  # noqa: F401

        assert StreamlitBackend is LegacyBackend
        return True, "StreamlitBackend import chain OK"
    except Exception as exc:
        return False, str(exc)


def _run_pytest() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit", "tests/acceptance", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
    summary = " | ".join(tail)
    return result.returncode == 0, summary


def main() -> int:
    ensure_runtime_directories(ROOT)
    readiness = build_readiness_report(ROOT)
    dependencies = check_dependencies()
    shadow_ok, shadow_detail = _simulate_streamlit_module_shadow()
    import_ok, import_detail = _import_backend_chain()
    tests_ok, tests_detail = _run_pytest()

    payload = {
        "validated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "project_root": str(project_root()),
        "readiness": {
            "ready": readiness.ready,
            "title": readiness.title,
            "summary": readiness.summary,
            "offline_mode": readiness.offline_mode,
            "gpu_available": readiness.gpu_available,
        },
        "dependencies": [asdict(item) for item in dependencies],
        "checks": {
            "no_app_shadowing": {"passed": shadow_ok, "detail": shadow_detail},
            "import_chain": {"passed": import_ok, "detail": import_detail},
            "pytest": {"passed": tests_ok, "detail": tests_detail},
        },
    }

    validation_dir = ROOT / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "portability_validation.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    circular_md = [
        "# Circular Import Fix",
        "",
        "## Root cause",
        "",
        "Streamlit executed `streamlit_app/app.py` as module name `app`, which shadowed",
        "the real `app` package. When `streamlit_app/bootstrap.py` imported",
        "`from app.container import ...`, Python resolved the partially initialized",
        "Streamlit script instead of the application package.",
        "",
        "## Fix",
        "",
        "- Renamed entry module to `streamlit_app/main.py`",
        "- Split adapter into `streamlit_app/backend.py` and `streamlit_app/startup.py`",
        "- Kept `streamlit_app/bootstrap.py` as a thin re-export only",
        "- Added `streamlit_app/runtime.py` to configure `sys.path` before backend imports",
        "",
        "## Measured validation",
        "",
        f"- Import chain: **{'PASS' if import_ok else 'FAIL'}** — {import_detail}",
        f"- No app shadowing: **{'PASS' if shadow_ok else 'FAIL'}** — {shadow_detail}",
        "",
    ]
    (docs / "CIRCULAR_IMPORT_FIX.md").write_text("\n".join(circular_md), encoding="utf-8")

    portability_md = [
        "# Portability Report",
        "",
        f"**Validated:** {payload['validated_at']}",
        f"**Project root:** `{payload['project_root']}`",
        "",
        "## Runtime directories",
        "",
        "Auto-created on launch: `models/`, `cache/`, `logs/`, `exports/`, `tmp/`, `assets/user/`, `.sentivis/`",
        "",
        "## Path resolution",
        "",
        "- Project root detected via `pyproject.toml` / `config/app.default.toml` markers",
        "- Override with environment variable `SENTIVIS_PROJECT_ROOT`",
        "- All configured paths resolved relative to project root",
        "",
        "## Readiness",
        "",
        f"- **{readiness.title}** — {readiness.summary}",
        f"- Offline mode: {readiness.offline_mode}",
        f"- GPU available: {readiness.gpu_available}",
        "",
        "## Dependencies",
        "",
        "| Package | Available | Version | Required |",
        "|---------|-----------|---------|----------|",
    ]
    for item in dependencies:
        portability_md.append(
            f"| {item.name} | {'Yes' if item.available else 'No'} | {item.version} | "
            f"{'Yes' if item.required else 'No'} |"
        )
    (docs / "PORTABILITY_REPORT.md").write_text("\n".join(portability_md), encoding="utf-8")

    runtime_md = [
        "# Runtime Validation",
        "",
        f"**Validated:** {payload['validated_at']}",
        "",
        "## Checks",
        "",
        f"- App package shadowing: **{'PASS' if shadow_ok else 'FAIL'}**",
        f"- Import chain: **{'PASS' if import_ok else 'FAIL'}**",
        f"- Pytest: **{'PASS' if tests_ok else 'FAIL'}**",
        f"- System readiness: **{readiness.title}**",
        "",
        "### Pytest summary",
        "",
        tests_detail,
        "",
    ]
    (docs / "RUNTIME_VALIDATION.md").write_text("\n".join(runtime_md), encoding="utf-8")

    judge_md = [
        "# Judge Deployment Guide",
        "",
        "## Requirements",
        "",
        "- Windows 10 or 11",
        "- Python 3.10.11",
        "- Copy the entire project folder to any drive or user profile",
        "",
        "## One-command launch",
        "",
        "```bash",
        "pip install -e .",
        "sentivis-ai",
        "```",
        "",
        "Alternative:",
        "",
        "```bash",
        "streamlit run streamlit_app/main.py",
        "```",
        "",
        "## First launch",
        "",
        "The application automatically creates runtime folders and runs a startup self-test.",
        "Open the sidebar **Startup diagnostics** panel for **System Ready** or **System Not Ready**.",
        "",
        "## Models",
        "",
        "Place weights in the portable `models/` directory. YOLO, BLIP, Gemma, SAM2, and RealESRGAN",
        "are discovered automatically from `models/` and configured search paths.",
        "",
        "## Offline use",
        "",
        "The UI starts without internet. Features that require downloads are disabled with warnings.",
        "",
        "## Measured status on validation machine",
        "",
        f"- {readiness.title}: {readiness.summary}",
        f"- Pytest: {'PASS' if tests_ok else 'FAIL'}",
        "",
    ]
    (docs / "JUDGE_DEPLOYMENT_GUIDE.md").write_text("\n".join(judge_md), encoding="utf-8")

    print("Validation complete.")
    print(f"  Readiness: {readiness.title}")
    print(f"  Shadowing: {'PASS' if shadow_ok else 'FAIL'}")
    print(f"  Imports: {'PASS' if import_ok else 'FAIL'}")
    print(f"  Pytest: {'PASS' if tests_ok else 'FAIL'}")
    return 0 if shadow_ok and import_ok and tests_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
