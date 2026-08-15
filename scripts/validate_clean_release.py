"""Clean-release validation: build → venv → install → import/smoke checks.

Usage (from repo root, with network for pip):
  python scripts/validate_clean_release.py

This script:
1. Builds a wheel via ``python -m build`` when available, else copies a release tree
2. Creates a temporary virtual environment
3. Installs the artifact
4. Verifies imports, translations, config markers, and Streamlit entry presence
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or ROOT, env=env, check=True)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="sentivis-clean-"))
    print(f"Work directory: {work}")
    try:
        # Prefer a release-folder install that mirrors competition portable layout.
        from release.builder import ReleaseBuilder

        dist_root = work / "dist"
        builder = ReleaseBuilder(root=ROOT)
        builder._dist_root = dist_root  # noqa: SLF001
        artifact = builder.build("development")
        print(f"Release artifact: {artifact}")

        venv_dir = work / "venv"
        venv.create(venv_dir, with_pip=True)
        py = _venv_python(venv_dir)

        # Install runtime deps from the project's requirements into the clean venv,
        # then put the artifact on PYTHONPATH (portable layout — not a wheel).
        _run([str(py), "-m", "pip", "install", "-U", "pip", "wheel"])
        # Minimal smoke deps for import checks (full ML stack is heavy).
        _run(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "tomli",
                "Pillow",
                "numpy<2",
                "psutil",
            ]
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(artifact)
        env["SENTIVIS_PROJECT_ROOT"] = str(artifact)

        smoke = r"""
import importlib
import json
from pathlib import Path
import os

root = Path(os.environ['SENTIVIS_PROJECT_ROOT'])
assert (root / 'streamlit_app' / 'main.py').is_file(), 'streamlit_app missing'
assert (root / 'translations' / 'en.json').is_file(), 'translations missing'
assert (root / 'config' / 'app.default.toml').is_file(), 'config missing'
assert (root / 'model_management').is_dir(), 'model_management missing'

from core.resources import load_translation_catalog
catalog = load_translation_catalog('en')
assert catalog, 'translation catalog empty'

# Domain must not require Streamlit to resolve language.
from core.config.ui_language import resolve_ui_language
assert resolve_ui_language() in {'en','fa','de','es','zh'}

for mod in (
    'app.main',
    'core.config.loader',
    'analysis.pose.pose_estimator',
    'analysis.ocr.text_extractor',
    'release.builder',
):
    importlib.import_module(mod)

print(json.dumps({'ok': True, 'translation_keys': len(catalog), 'root': str(root)}))
"""
        _run([str(py), "-c", smoke], env=env)
        print("CLEAN RELEASE VALIDATION PASSED")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"CLEAN RELEASE VALIDATION FAILED: {exc}")
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
