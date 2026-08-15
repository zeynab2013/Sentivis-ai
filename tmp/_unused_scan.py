"""Scan for demonstrably unused Python modules in target packages."""
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


def collect_candidates() -> list[dict]:
    out: list[dict] = []
    for pkg in PACKAGES:
        base = ROOT / pkg
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            if should_skip(p):
                continue
            rel = p.relative_to(ROOT).as_posix()
            if p.name == "__init__.py":
                mod = ".".join(p.relative_to(ROOT).parent.parts)
                is_init = True
            else:
                mod = ".".join(p.relative_to(ROOT).with_suffix("").parts)
                is_init = False
            out.append(
                {
                    "path": p,
                    "rel": rel,
                    "mod": mod,
                    "is_init": is_init,
                    "stem": p.stem,
                    "pkg": pkg,
                }
            )
    return out


def collect_search_files() -> list[Path]:
    files: list[Path] = []
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
            files.append(p)
    return files


def parse_imports(text: str, relsp: str, import_map: dict[str, set[str]]) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # fallback regex
        for m in re.finditer(
            r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", text, re.M
        ):
            mod = m.group(1) or m.group(2)
            if mod:
                import_map[mod].add(relsp)
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_map[alias.name].add(relsp)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # absolute
                if node.level == 0:
                    import_map[node.module].add(relsp)
                    for alias in node.names:
                        if alias.name != "*":
                            import_map[f"{node.module}.{alias.name}"].add(relsp)


def main() -> None:
    candidates = collect_candidates()
    search_files = collect_search_files()
    contents: dict[Path, str] = {}
    for p in search_files:
        try:
            contents[p] = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass

    import_map: dict[str, set[str]] = defaultdict(set)
    string_mod_refs: dict[str, set[str]] = defaultdict(set)
    path_refs: dict[str, set[str]] = defaultdict(set)
    stem_script_refs: dict[str, set[str]] = defaultdict(set)

    dyn_patterns = [
        re.compile(r"""(?:importlib\.import_module|__import__)\(\s*['"]([\w.]+)['"]"""),
        re.compile(r"""(?:import_module)\(\s*['"]([\w.]+)['"]"""),
        re.compile(
            r"""(?:register_plugin|load_plugin|get_plugin|create_engine|get_adapter)\([^)]*['"]([\w.\-]+)['"]"""
        ),
        re.compile(r"""['"]((?:analysis|language|vision|services|ui|app|core|scripts)[\w.]*)['"]"""),
    ]

    for sp, text in contents.items():
        relsp = sp.relative_to(ROOT).as_posix()
        if sp.suffix == ".py":
            parse_imports(text, relsp, import_map)
        for pat in dyn_patterns:
            for m in pat.finditer(text):
                string_mod_refs[m.group(1)].add(relsp)

    # Build reverse index of path mentions
    for c in candidates:
        rel = c["rel"]
        win = rel.replace("/", "\\")
        name = c["path"].name
        for sp, text in contents.items():
            if sp.resolve() == c["path"].resolve():
                continue
            relsp = sp.relative_to(ROOT).as_posix()
            if rel in text or win in text or f"scripts/{name}" in text:
                path_refs[c["mod"]].add(relsp)
            if c["pkg"] == "scripts" and name in text:
                # require nearby scripts context or exact filename
                if name in text:
                    stem_script_refs[c["mod"]].add(relsp)

    # Also check __all__ and package re-exports by reading __init__ files
    init_exports: dict[str, set[str]] = defaultdict(set)
    for c in candidates:
        if not c["is_init"]:
            continue
        text = contents.get(c["path"], "")
        # from .foo import bar
        for m in re.finditer(r"from\s+\.(\w+)\s+import", text):
            child = f"{c['mod']}.{m.group(1)}" if c["mod"] else m.group(1)
            init_exports[child].add(c["rel"])
        for m in re.finditer(r"__all__\s*=\s*\[([^\]]+)\]", text, re.S):
            for name in re.findall(r"['\"](\w+)['\"]", m.group(1)):
                child = f"{c['mod']}.{name}" if c["mod"] else name
                init_exports[child].add(c["rel"])

    zero = []
    low = []
    for c in candidates:
        mod = c["mod"]
        refs: set[str] = set()
        refs |= import_map.get(mod, set())
        refs |= string_mod_refs.get(mod, set())
        refs |= path_refs.get(mod, set())
        refs |= init_exports.get(mod, set())
        if c["pkg"] == "scripts":
            refs |= stem_script_refs.get(mod, set())

        # Also: if something imports a symbol that matches this module's
        # parent package importing this file via relative - covered by init_exports

        # Remove self-references
        refs = {r for r in refs if r != c["rel"]}

        # Parent package init importing us counts
        # already in init_exports / import_map

        c["refs"] = sorted(refs)
        c["ref_count"] = len(refs)

        if not refs:
            zero.append(c)
        elif len(refs) <= 3:
            low.append(c)

    print(f"Candidates: {len(candidates)}")
    print(f"Search files: {len(contents)}")
    print("\n=== ZERO REFERENCES ===")
    for c in sorted(zero, key=lambda x: x["rel"]):
        print(f"ZERO\t{c['rel']}\t{c['mod']}\tinit={c['is_init']}")

    print("\n=== LOW REFS (1-3) for manual review ===")
    for c in sorted(low, key=lambda x: (x["ref_count"], x["rel"])):
        print(f"LOW{c['ref_count']}\t{c['rel']}\trefs={c['refs']}")

    # Dump import_map size for sanity
    print(f"\nUnique imported names: {len(import_map)}")
    print(f"Zero count: {len(zero)}")


if __name__ == "__main__":
    main()
