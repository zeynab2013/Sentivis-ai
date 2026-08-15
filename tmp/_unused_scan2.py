"""Refined unused-module scan with relative-import resolution."""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["analysis", "language", "vision", "services", "ui", "app", "core", "scripts"]
IGNORE_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "models",
    "tmp",
    ".pytest_cache",
    "node_modules",
    "realesrgan",
    "cache",
    "logs",
    "exports",
    "temp",
    "cert_tmp",
    "cert_install_sim",
    "sentivis_ai.egg-info",
    ".sentivis",
}
SEARCH_SUFFIXES = {
    ".py",
    ".toml",
    ".yml",
    ".yaml",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
    ".json",
    ".bat",
    ".ps1",
    ".sh",
    ".rst",
}


def should_skip(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)


def file_to_mod(path: Path) -> str:
    if path.name == "__init__.py":
        return ".".join(path.relative_to(ROOT).parent.parts)
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def resolve_relative(importer: Path, module: str | None, level: int) -> str | None:
    """Resolve relative import to absolute dotted module."""
    pkg_parts = list(importer.relative_to(ROOT).parent.parts)
    # level=1 means current package; level=2 means parent, etc.
    if level > len(pkg_parts):
        return None
    base = pkg_parts[: len(pkg_parts) - (level - 1)] if level >= 1 else pkg_parts
    if module:
        return ".".join(base + module.split("."))
    return ".".join(base) if base else None


def collect_candidates() -> list[dict]:
    out: list[dict] = []
    for pkg in PACKAGES:
        base = ROOT / pkg
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            if should_skip(p):
                continue
            out.append(
                {
                    "path": p,
                    "rel": p.relative_to(ROOT).as_posix(),
                    "mod": file_to_mod(p),
                    "is_init": p.name == "__init__.py",
                    "stem": p.stem,
                    "pkg": pkg,
                }
            )
    return out


def main() -> None:
    candidates = collect_candidates()
    by_mod = {c["mod"]: c for c in candidates}

    contents: dict[Path, str] = {}
    for p in ROOT.rglob("*"):
        if not p.is_file() or should_skip(p):
            continue
        if p.suffix.lower() in SEARCH_SUFFIXES or p.name in {
            "Dockerfile",
            "Makefile",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
        }:
            try:
                contents[p] = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass

    import_map: dict[str, set[str]] = defaultdict(set)
    symbol_map: dict[str, set[str]] = defaultdict(set)  # ClassName / func -> files
    path_mentions: dict[str, set[str]] = defaultdict(set)

    for sp, text in contents.items():
        relsp = sp.relative_to(ROOT).as_posix()
        if sp.suffix != ".py":
            # non-py path mentions only
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_map[alias.name].add(relsp)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    abs_mod = resolve_relative(sp, node.module, node.level)
                    if abs_mod:
                        import_map[abs_mod].add(relsp)
                        for alias in node.names:
                            if alias.name != "*":
                                import_map[f"{abs_mod}.{alias.name}"].add(relsp)
                                symbol_map[alias.name].add(relsp)
                elif node.module:
                    import_map[node.module].add(relsp)
                    for alias in node.names:
                        if alias.name != "*":
                            import_map[f"{node.module}.{alias.name}"].add(relsp)
                            symbol_map[alias.name].add(relsp)

        # dynamic import strings
        for m in re.finditer(
            r"""(?:importlib\.import_module|__import__)\(\s*['"]([\w.]+)['"]""",
            text,
        ):
            import_map[m.group(1)].add(relsp)

    # path string mentions across all files
    for c in candidates:
        rel = c["rel"]
        name = c["path"].name
        for sp, text in contents.items():
            if sp.resolve() == c["path"].resolve():
                continue
            relsp = sp.relative_to(ROOT).as_posix()
            if rel in text or f"scripts/{name}" in text:
                path_mentions[c["mod"]].add(relsp)

    # Extract top-level definitions for symbol cross-check
    defined_symbols: dict[str, list[str]] = defaultdict(list)  # symbol -> mods defining it
    for c in candidates:
        if c["is_init"]:
            continue
        text = contents.get(c["path"], "")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined_symbols[node.name].append(c["mod"])

    print("=== NON-INIT ZERO / NEAR-ZERO (code refs only, excluding docs-only) ===\n")
    safeish = []
    for c in sorted(candidates, key=lambda x: x["rel"]):
        if c["is_init"]:
            continue
        mod = c["mod"]
        code_refs = set()
        for r in import_map.get(mod, set()) | path_mentions.get(mod, set()):
            # classify
            code_refs.add(r)

        # filter self
        code_refs = {r for r in code_refs if r != c["rel"]}

        py_refs = {r for r in code_refs if r.endswith(".py")}
        non_test_py = {
            r
            for r in py_refs
            if not r.startswith("tests/")
            and not r.startswith("tmp/")
            and not r.startswith("scripts/")
        }
        test_py = {r for r in py_refs if r.startswith("tests/")}
        script_py = {r for r in py_refs if r.startswith("scripts/")}
        doc_refs = {r for r in code_refs if not r.endswith(".py")}

        # Symbol-name weak signal (only if uniquely defined)
        weak_symbol = set()
        text = contents.get(c["path"], "")
        try:
            tree = ast.parse(text)
            tops = [
                n.name
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
        except SyntaxError:
            tops = []
        for sym in tops:
            defs = defined_symbols.get(sym, [])
            if len(defs) == 1 and defs[0] == mod:
                for r in symbol_map.get(sym, set()):
                    if r != c["rel"]:
                        weak_symbol.add(r)

        if not py_refs:
            status = "NO_PY_REFS"
            safeish.append(c)
            print(f"{status}\t{c['rel']}")
            print(f"  docs/config mentions: {sorted(doc_refs) or '-'}")
            print(f"  unique-symbol imports: {sorted(weak_symbol) or '-'}")
            print()

    print("\n=== PACKAGE __init__ with zero package-name imports (informational, NOT delete) ===")
    init_zero = 0
    for c in candidates:
        if not c["is_init"]:
            continue
        refs = {r for r in import_map.get(c["mod"], set()) if r != c["rel"]}
        if not refs:
            init_zero += 1
    print(f"count={init_zero} (package markers; do not delete)\n")

    # Spotlight specific suspicious from first scan
    spotlight = [
        "language.blip.vision_language_service",
        "language.interfaces.vision_model",
        "language.blip.observation_mapper",
        "analysis.activity.activity_analyzer",
        "vision.tracking.noop_tracker",
        "app.desktop_main",
        "services.benchmark.benchmark_runner",
        "language.tts.speech_service",
        "language.tts.audio_utils",
        "ui.components.button",
        "ui.components.card",
        "ui.components.empty_state",
        "ui.components.notification",
        "ui.components.progress",
        "ui.components.scroll_panel",
        "ui.branding.logo_provider",
        "ui.themes.theme_engine",
        "ui.preferences.ui_preferences",
        "core.utils.paths",
        "core.utils.images",
        "core.utils.timing",
        "analysis.relationships.relation_metrics",
        "services.runtime.model_status",
        "services.runtime.model_record",
        "services.pipeline.progress_reporter",
        "services.pipeline.cancellation",
        "services.models.device_selector",
        "services.models.model_validator",
        "services.export.export_manager",
        "app.plugin_bootstrap",
        "language.florence.florence_engine",
        "language.gemma.gemma_engine",
        "language.prompts.context_caption",
        "language.prompts.prompt_builder",
    ]
    print("=== SPOTLIGHT ===")
    for mod in spotlight:
        c = by_mod.get(mod)
        if not c:
            print(f"MISSING\t{mod}")
            continue
        refs = sorted({r for r in import_map.get(mod, set()) | path_mentions.get(mod, set()) if r != c["rel"]})
        print(f"{mod}")
        print(f"  refs={refs or '-'}")


if __name__ == "__main__":
    main()
