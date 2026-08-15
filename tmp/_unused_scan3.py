"""Find non-init modules with zero import references (strict)."""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["analysis", "language", "vision", "services", "ui", "app", "core"]
IGNORE = {
    ".venv", "venv", "__pycache__", ".git", "models", "tmp", ".pytest_cache",
    "realesrgan", "cache", "logs", "exports", "temp", "cert_tmp",
    "cert_install_sim", "sentivis_ai.egg-info", ".sentivis",
}


def skip(p: Path) -> bool:
    return any(x in IGNORE for x in p.parts)


def resolve_rel(importer: Path, module: str | None, level: int) -> str | None:
    pkg = list(importer.relative_to(ROOT).parent.parts)
    if level > len(pkg):
        return None
    base = pkg[: len(pkg) - (level - 1)]
    if module:
        return ".".join(base + module.split("."))
    return ".".join(base) if base else None


def main() -> None:
    texts: dict[Path, str] = {}
    for p in ROOT.rglob("*"):
        if not p.is_file() or skip(p):
            continue
        if p.suffix.lower() in {".py", ".toml", ".md", ".yml", ".yaml", ".txt", ".json", ".rst"}:
            try:
                texts[p] = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass

    cands: list[tuple[str, Path]] = []
    for pkg in PACKAGES:
        for p in (ROOT / pkg).rglob("*.py"):
            if skip(p) or p.name == "__init__.py":
                continue
            mod = ".".join(p.relative_to(ROOT).with_suffix("").parts)
            cands.append((mod, p))

    imap: dict[str, set[str]] = defaultdict(set)
    for sp, text in texts.items():
        if sp.suffix != ".py":
            continue
        rel = sp.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imap[a.name].add(rel)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    absmod = resolve_rel(sp, node.module, node.level)
                    if absmod:
                        imap[absmod].add(rel)
                elif node.module:
                    imap[node.module].add(rel)

        for m in re.finditer(
            r"(?:importlib\.import_module|__import__)\(\s*[\"']([\w.]+)[\"']",
            text,
        ):
            imap[m.group(1)].add(rel)

    # entry points / config string module refs
    for sp, text in texts.items():
        rel = sp.relative_to(ROOT).as_posix()
        if sp.suffix not in {".toml", ".md", ".yml", ".yaml", ".json", ".txt"} and "pyproject" not in sp.name:
            # also scan py for quoted module paths
            pass
        for mod, p in cands:
            # entry-point style app.foo:main
            if re.search(rf"(?<![\w.]){re.escape(mod)}(?::[\w]+)?(?![\w.])", text):
                # avoid counting self file content defining the module path in docstrings lightly
                if sp.resolve() == p.resolve():
                    continue
                # require import-like or entrypoint-like context for .py files
                if sp.suffix == ".py":
                    if re.search(
                        rf"(?:from|import)\s+{re.escape(mod)}\b|[\"']{re.escape(mod)}[\"']",
                        text,
                    ):
                        imap[mod].add(rel)
                else:
                    imap[mod].add(rel)

    print("STRICT_ZERO_CODE_AND_CONFIG")
    for mod, p in sorted(cands):
        self_rel = p.relative_to(ROOT).as_posix()
        refs = {r for r in imap.get(mod, set()) if r != self_rel}
        if not refs:
            print(self_rel)

    print("\nONLY_INIT_OR_TESTS")
    for mod, p in sorted(cands):
        self_rel = p.relative_to(ROOT).as_posix()
        refs = {r for r in imap.get(mod, set()) if r != self_rel}
        if not refs:
            continue
        non_doc = {r for r in refs if r.endswith(".py")}
        prod = {
            r
            for r in non_doc
            if not r.startswith("tests/")
            and not r.startswith("scripts/")
            and not r.endswith("__init__.py")
        }
        if not prod and non_doc:
            print(f"{self_rel}\trefs={sorted(refs)}")


if __name__ == "__main__":
    main()
