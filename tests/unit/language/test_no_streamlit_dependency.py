"""Architecture boundary: language must not import streamlit_app."""

from __future__ import annotations

import ast
from pathlib import Path

from core.config.ui_language import (
    clear_ui_language_provider,
    register_ui_language_provider,
    resolve_ui_language,
)
from language.refinement.caption_refiner import active_ui_language, clear_ui_language_cache


ROOT = Path(__file__).resolve().parents[3]


def test_language_package_has_no_streamlit_app_imports() -> None:
    language_root = ROOT / "language"
    offenders: list[str] = []
    for path in language_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "streamlit_app" or alias.name.startswith("streamlit_app."):
                        offenders.append(str(path.relative_to(ROOT)))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "streamlit_app" or node.module.startswith("streamlit_app."):
                    offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_ui_language_provider_abstraction(monkeypatch) -> None:
    monkeypatch.delenv("SENTIVIS_UI_LANGUAGE", raising=False)
    clear_ui_language_cache()
    clear_ui_language_provider()
    register_ui_language_provider(lambda: "de")
    try:
        assert resolve_ui_language() == "de"
        assert active_ui_language() == "de"
    finally:
        clear_ui_language_provider()
        clear_ui_language_cache()
